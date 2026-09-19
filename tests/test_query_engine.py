import json

import pandas as pd
import pytest

from app.llm_providers import LLMProvider
from app.query_engine import (
    QuerySpecError,
    answer_question,
    execute_query,
    validate_spec,
)


@pytest.fixture
def df():
    return pd.DataFrame(
        {
            "ticket_id": ["TKT-1", "TKT-2", "TKT-3", "TKT-4"],
            "created_at": pd.to_datetime(
                ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"]
            ),
            "category": ["Billing", "Technical", "Technical", "General"],
            "priority": ["High", "Critical", "Low", "Critical"],
            "status": ["Open", "Open", "Resolved", "Escalated"],
            "response_time_hrs": [1.0, 2.0, 3.0, 4.0],
            "resolution_time_hrs": [None, None, 5.0, None],
            "agent_id": ["AGT-1", "AGT-2", "AGT-1", "AGT-2"],
            "customer_rating": [None, None, 4, None],
            "issue_summary": ["a", "b", "c", "d"],
            "age_hrs": [100.0, 80.0, 60.0, 10.0],
        }
    )


class FakeLLM(LLMProvider):
    name = "fake"

    def __init__(self, response: str):
        self.response = response

    def complete(self, prompt: str, system: str | None = None) -> str:
        return self.response


def test_validate_spec_rejects_unknown_column():
    with pytest.raises(QuerySpecError):
        validate_spec({"operation": "count", "filters": [{"column": "nope", "op": "==", "value": 1}]})


def test_validate_spec_rejects_unknown_operation():
    with pytest.raises(QuerySpecError):
        validate_spec({"operation": "drop_table", "filters": []})


def test_execute_count(df):
    spec = {"operation": "count", "filters": [{"column": "status", "op": "==", "value": "Open"}]}
    result = execute_query(df, spec)
    assert result == {"count": 2}


def test_execute_aggregate(df):
    spec = {
        "operation": "aggregate",
        "filters": [{"column": "category", "op": "==", "value": "Technical"}],
        "agg_column": "response_time_hrs",
        "agg_func": "mean",
    }
    result = execute_query(df, spec)
    assert result["value"] == pytest.approx(2.5)
    assert result["n"] == 2


def test_execute_groupby_agg_sorted(df):
    spec = {
        "operation": "groupby_agg",
        "filters": [],
        "group_by": "agent_id",
        "agg_column": "response_time_hrs",
        "agg_func": "mean",
        "sort": {"by": "value", "ascending": False},
        "limit": 1,
    }
    result = execute_query(df, spec)
    assert len(result) == 1
    assert result[0]["agent_id"] == "AGT-2"  # (2+4)/2=3.0 > AGT-1's (1+3)/2=2.0


def test_execute_list_respects_filters_and_limit(df):
    spec = {
        "operation": "list",
        "filters": [{"column": "priority", "op": "==", "value": "Critical"}],
        "limit": 1,
    }
    result = execute_query(df, spec)
    assert len(result) == 1
    assert result[0]["priority"] == "Critical"


def test_answer_question_end_to_end(df):
    spec = {"operation": "count", "filters": [{"column": "status", "op": "==", "value": "Open"}]}
    fake = FakeLLM(json.dumps(spec))
    result = answer_question(
        df=df,
        question="How many tickets are open?",
        llm=fake,
        schema_description="irrelevant for this fake",
        phrase_with_llm=False,
    )
    assert result.raw_result == {"count": 2}
    assert not result.used_llm_fallback
    assert "2" in result.answer_text


def test_answer_question_falls_back_on_garbage_llm_output(df):
    fake = FakeLLM("this is not json at all")
    result = answer_question(
        df=df,
        question="anything",
        llm=fake,
        schema_description="irrelevant",
        phrase_with_llm=False,
    )
    assert result.used_llm_fallback
    assert result.warnings
