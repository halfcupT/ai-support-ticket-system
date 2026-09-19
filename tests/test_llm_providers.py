import json

from app.llm_providers import RuleBasedProvider
from app.query_engine import FEW_SHOT, QUERY_SPEC_FIELDS, SYSTEM_PROMPT


def build_prompt(question: str) -> str:
    # Mirrors the real prompt shape built in query_engine.answer_question,
    # which matters here because the few-shot block itself contains several
    # "Question: ..." lines that a naive parser could latch onto instead of
    # the real, final question.
    return f"schema\n\n{QUERY_SPEC_FIELDS}\n{FEW_SHOT}\nQuestion: {question}\nJSON:"


def test_rule_based_provider_uses_the_final_question_not_a_few_shot_example():
    provider = RuleBasedProvider()
    prompt = build_prompt("How many critical tickets are unresolved?")
    spec = json.loads(provider.complete(prompt, system=SYSTEM_PROMPT))
    assert spec["operation"] == "count"
    filter_cols = {f["column"] for f in spec["filters"]}
    assert filter_cols == {"priority", "status"}


def test_rule_based_provider_open_count():
    provider = RuleBasedProvider()
    prompt = build_prompt("How many tickets are currently open?")
    spec = json.loads(provider.complete(prompt, system=SYSTEM_PROMPT))
    assert spec == {
        "operation": "count",
        "filters": [{"column": "status", "op": "==", "value": "Open"}],
    }


def test_rule_based_provider_lowest_avg_rating_agent():
    provider = RuleBasedProvider()
    prompt = build_prompt("Which agent has the lowest average customer rating?")
    spec = json.loads(provider.complete(prompt, system=SYSTEM_PROMPT))
    assert spec["operation"] == "groupby_agg"
    assert spec["sort"]["ascending"] is True
