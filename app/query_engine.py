"""
Natural-language query engine.

Rather than asking the LLM to write and `eval()` arbitrary pandas/Python code
(a real security and reliability hazard - a hallucinated or malicious
snippet could do anything), we ask it to translate the question into a small,
whitelisted JSON "query spec": which columns to filter on, how to aggregate,
how to sort. That spec is then executed by hand-written, unit-tested pandas
code (`execute_query`). This keeps the LLM's job narrow (translation, which
LLMs are good at) and keeps execution deterministic and safe.

Pipeline for each question:
  1. Build a prompt containing the schema + the question + few-shot examples.
  2. Ask the LLM for a JSON query spec (with one retry-with-feedback if the
     first response doesn't parse or fails validation).
  3. Execute the spec against the in-memory DataFrame.
  4. Ask the LLM to phrase the numeric/tabular result as a short natural
     language answer (falls back to a templated sentence if this step fails
     or the LLM is unavailable, so a broken "phrasing" step never hides a
     result the user actually asked for).
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from app.llm_providers import LLMError, LLMProvider

logger = logging.getLogger(__name__)

ALLOWED_COLUMNS = {
    "ticket_id", "created_at", "category", "priority", "status",
    "response_time_hrs", "resolution_time_hrs", "agent_id",
    "customer_rating", "issue_summary", "age_hrs",
}
ALLOWED_FILTER_OPS = {"==", "!=", ">", ">=", "<", "<=", "in", "contains"}
ALLOWED_AGG_FUNCS = {"mean", "sum", "count", "min", "max", "median", "nunique"}
ALLOWED_OPERATIONS = {"count", "aggregate", "groupby_agg", "list"}

SYSTEM_PROMPT = (
    "You are a translator from natural-language questions about a customer "
    "support ticket dataset into a strict JSON query spec. You NEVER answer "
    "the question directly and you NEVER write pandas/SQL/Python code. You "
    "ONLY output a single JSON object, with no markdown fences and no "
    "commentary before or after it."
)

FEW_SHOT = """
Examples:

Question: How many tickets are currently open?
JSON: {"operation": "count", "filters": [{"column": "status", "op": "==", "value": "Open"}]}

Question: How many critical tickets are unresolved?
JSON: {"operation": "count", "filters": [{"column": "priority", "op": "==", "value": "Critical"}, {"column": "status", "op": "!=", "value": "Resolved"}]}

Question: Which agent has the lowest average customer rating?
JSON: {"operation": "groupby_agg", "filters": [], "group_by": "agent_id", "agg_column": "customer_rating", "agg_func": "mean", "sort": {"by": "value", "ascending": true}, "limit": 1}

Question: What is the average customer rating for Technical category tickets?
JSON: {"operation": "aggregate", "filters": [{"column": "category", "op": "==", "value": "Technical"}], "agg_column": "customer_rating", "agg_func": "mean"}

Question: Which agent resolved the most tickets?
JSON: {"operation": "groupby_agg", "filters": [{"column": "status", "op": "==", "value": "Resolved"}], "group_by": "agent_id", "agg_column": "ticket_id", "agg_func": "count", "sort": {"by": "value", "ascending": false}, "limit": 1}

Question: Show me all Critical tickets not resolved within 12 hours.
JSON: {"operation": "list", "filters": [{"column": "priority", "op": "==", "value": "Critical"}, {"column": "status", "op": "!=", "value": "Resolved"}, {"column": "age_hrs", "op": ">", "value": 12}], "limit": 50}

Question: List billing tickets handled by AGT-04.
JSON: {"operation": "list", "filters": [{"column": "category", "op": "==", "value": "Billing"}, {"column": "agent_id", "op": "==", "value": "AGT-04"}], "limit": 50}
"""

QUERY_SPEC_FIELDS = """
JSON schema for the query spec:
{
  "operation": "count" | "aggregate" | "groupby_agg" | "list",
  "filters": [ {"column": <col>, "op": "==" | "!=" | ">" | ">=" | "<" | "<=" | "in" | "contains", "value": <val>} , ... ],
  "group_by": <col> | null,        // required for groupby_agg
  "agg_column": <col> | null,      // required for aggregate & groupby_agg
  "agg_func": "mean"|"sum"|"count"|"min"|"max"|"median"|"nunique" | null,
  "sort": {"by": "value", "ascending": true|false} | null,   // only for groupby_agg
  "limit": <int> | null            // caps rows returned for "list", or groups for groupby_agg
}
Only use columns from the schema below. Use "in" with a list value for
membership checks, and "contains" for free-text substring search on
issue_summary. Always respond with ONLY the JSON object.
"""


class QuerySpecError(ValueError):
    """Raised when an LLM-produced query spec is malformed or unsafe."""


@dataclass
class QueryResult:
    question: str
    spec: dict[str, Any]
    raw_result: Any  # scalar, dict, or list[dict]
    answer_text: str
    used_llm_fallback: bool = False
    warnings: list[str] = field(default_factory=list)


def _extract_json(text: str) -> dict[str, Any]:
    """Pull the first {...} JSON object out of a possibly messy LLM response."""
    text = text.strip()
    # Strip common markdown code fences.
    text = re.sub(r"^```(json)?", "", text.strip(), flags=re.IGNORECASE).strip()
    text = re.sub(r"```$", "", text.strip()).strip()
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise QuerySpecError(f"No JSON object found in LLM output: {text!r}")
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise QuerySpecError(f"LLM output was not valid JSON: {exc}") from exc


def validate_spec(spec: dict[str, Any]) -> dict[str, Any]:
    """Whitelist-validate a query spec. Raises QuerySpecError if unsafe/invalid."""
    if not isinstance(spec, dict):
        raise QuerySpecError("Query spec must be a JSON object.")

    operation = spec.get("operation")
    if operation not in ALLOWED_OPERATIONS:
        raise QuerySpecError(f"Unknown or missing operation: {operation!r}")

    filters = spec.get("filters") or []
    if not isinstance(filters, list):
        raise QuerySpecError("filters must be a list.")
    for f in filters:
        if not isinstance(f, dict) or {"column", "op", "value"} - f.keys():
            raise QuerySpecError(f"Malformed filter: {f!r}")
        if f["column"] not in ALLOWED_COLUMNS:
            raise QuerySpecError(f"Unknown column in filter: {f['column']!r}")
        if f["op"] not in ALLOWED_FILTER_OPS:
            raise QuerySpecError(f"Unknown filter operator: {f['op']!r}")

    if operation in ("aggregate", "groupby_agg"):
        agg_col = spec.get("agg_column")
        if agg_col not in ALLOWED_COLUMNS:
            raise QuerySpecError(f"Unknown or missing agg_column: {agg_col!r}")
        agg_func = spec.get("agg_func")
        if agg_func not in ALLOWED_AGG_FUNCS:
            raise QuerySpecError(f"Unknown or missing agg_func: {agg_func!r}")

    if operation == "groupby_agg":
        group_by = spec.get("group_by")
        if group_by not in ALLOWED_COLUMNS:
            raise QuerySpecError(f"Unknown or missing group_by column: {group_by!r}")

    limit = spec.get("limit")
    if limit is not None and (not isinstance(limit, int) or limit <= 0):
        raise QuerySpecError("limit must be a positive integer if provided.")

    return spec


def _apply_filters(df: pd.DataFrame, filters: list[dict[str, Any]]) -> pd.DataFrame:
    for f in filters:
        col, op, val = f["column"], f["op"], f["value"]
        series = df[col]
        if op == "==":
            df = df[series == val]
        elif op == "!=":
            df = df[series != val]
        elif op == ">":
            df = df[series > val]
        elif op == ">=":
            df = df[series >= val]
        elif op == "<":
            df = df[series < val]
        elif op == "<=":
            df = df[series <= val]
        elif op == "in":
            values = val if isinstance(val, list) else [val]
            df = df[series.isin(values)]
        elif op == "contains":
            df = df[series.astype("string").str.contains(str(val), case=False, na=False)]
    return df


def execute_query(df: pd.DataFrame, spec: dict[str, Any]) -> Any:
    """Execute a validated query spec against the ticket DataFrame."""
    operation = spec["operation"]
    filtered = _apply_filters(df, spec.get("filters") or [])

    if operation == "count":
        return {"count": int(len(filtered))}

    if operation == "aggregate":
        agg_col, agg_func = spec["agg_column"], spec["agg_func"]
        series = filtered[agg_col].dropna()
        if series.empty:
            return {"value": None, "n": 0}
        value = getattr(series, agg_func)()
        return {"value": round(float(value), 3) if pd.notna(value) else None, "n": int(len(series))}

    if operation == "groupby_agg":
        group_by, agg_col, agg_func = spec["group_by"], spec["agg_column"], spec["agg_func"]
        grouped = filtered.dropna(subset=[agg_col]).groupby(group_by)[agg_col]
        result = getattr(grouped, agg_func)().reset_index(name="value")
        counts = filtered.groupby(group_by).size().reset_index(name="n")
        result = result.merge(counts, on=group_by, how="left")

        sort = spec.get("sort")
        if sort:
            result = result.sort_values("value", ascending=bool(sort.get("ascending", True)))
        limit = spec.get("limit")
        if limit:
            result = result.head(limit)
        return result.round(3).to_dict(orient="records")

    if operation == "list":
        limit = spec.get("limit") or 50
        cols = [c for c in df.columns if c != "age_hrs"]
        return filtered[cols].head(limit).to_dict(orient="records")

    raise QuerySpecError(f"Unhandled operation: {operation}")  # pragma: no cover


def _templated_answer(question: str, spec: dict[str, Any], result: Any) -> str:
    """A deterministic, always-available fallback phrasing (no LLM needed)."""
    operation = spec["operation"]
    if operation == "count":
        return f"There are {result['count']} matching ticket(s)."
    if operation == "aggregate":
        if result["value"] is None:
            return "No matching rows had a value for that column."
        return f"The {spec['agg_func']} of {spec['agg_column']} is {result['value']} (n={result['n']})."
    if operation == "groupby_agg":
        if not result:
            return "No matching groups were found."
        lines = [f"{r[spec['group_by']]}: {r['value']} (n={r['n']})" for r in result]
        return "; ".join(lines)
    if operation == "list":
        return f"Found {len(result)} matching ticket(s) (showing up to the limit)."
    return "Done."


def _phrase_answer(llm: LLMProvider, question: str, spec: dict, result: Any) -> str:
    """Ask the LLM to turn the raw result into a short natural-language answer."""
    prompt = (
        "Answer the user's question in one or two short sentences, using ONLY "
        "the data below. Do not invent numbers that aren't present. Be direct "
        "and quote the relevant figures.\n\n"
        f"Question: {question}\n"
        f"Query performed: {json.dumps(spec)}\n"
        f"Result data (JSON): {json.dumps(result, default=str)}\n"
    )
    try:
        text = llm.complete(prompt, system="You are a precise data analyst assistant.")
        return text.strip()
    except LLMError as exc:
        logger.warning("LLM phrasing step failed, falling back to template: %s", exc)
        return _templated_answer(question, spec, result)


def answer_question(
    df: pd.DataFrame,
    question: str,
    llm: LLMProvider,
    schema_description: str,
    phrase_with_llm: bool = True,
) -> QueryResult:
    """Full NL -> spec -> execution -> NL-answer pipeline for one question."""
    warnings: list[str] = []
    prompt = (
        f"{schema_description}\n\n{QUERY_SPEC_FIELDS}\n{FEW_SHOT}\n"
        f"Question: {question}\nJSON:"
    )

    spec: dict[str, Any] | None = None
    last_error: str | None = None
    used_fallback = False

    for attempt in range(2):  # one shot + one retry-with-feedback
        try:
            raw = llm.complete(
                prompt if attempt == 0 else f"{prompt}\n\n(Your previous answer was invalid: "
                f"{last_error}. Return ONLY corrected JSON.)",
                system=SYSTEM_PROMPT,
            )
            candidate = _extract_json(raw)
            spec = validate_spec(candidate)
            break
        except (LLMError, QuerySpecError) as exc:
            last_error = str(exc)
            logger.warning("Query-spec attempt %d failed: %s", attempt + 1, exc)

    if spec is None:
        warnings.append(
            f"LLM could not produce a valid query after retries ({last_error}); "
            "used a safe default (unfiltered ticket count)."
        )
        spec = {"operation": "count", "filters": []}
        used_fallback = True

    result = execute_query(df, spec)

    if phrase_with_llm and not used_fallback:
        answer_text = _phrase_answer(llm, question, spec, result)
    else:
        answer_text = _templated_answer(question, spec, result)

    return QueryResult(
        question=question,
        spec=spec,
        raw_result=result,
        answer_text=answer_text,
        used_llm_fallback=used_fallback,
        warnings=warnings,
    )
