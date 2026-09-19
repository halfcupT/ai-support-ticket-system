import pandas as pd
import pytest

from app import anomaly


@pytest.fixture
def df():
    # 6 resolved Technical tickets: 5 cluster around 4-6h, one huge outlier.
    rows = []
    for i, res_time in enumerate([4.0, 5.0, 4.5, 5.5, 4.8, 60.0]):
        rows.append(
            {
                "ticket_id": f"TKT-{i}",
                "created_at": pd.Timestamp("2024-01-01"),
                "category": "Technical",
                "priority": "Low",
                "status": "Resolved",
                "response_time_hrs": 1.0,
                "resolution_time_hrs": res_time,
                "agent_id": "AGT-1",
                "customer_rating": 4,
                "issue_summary": "x",
                "age_hrs": 200.0,
            }
        )
    # An unresolved Critical ticket open way past SLA.
    rows.append(
        {
            "ticket_id": "TKT-SLA",
            "created_at": pd.Timestamp("2024-01-01"),
            "category": "Technical",
            "priority": "Critical",
            "status": "Open",
            "response_time_hrs": 1.0,
            "resolution_time_hrs": None,
            "agent_id": "AGT-1",
            "customer_rating": None,
            "issue_summary": "urgent",
            "age_hrs": 48.0,
        }
    )
    # A resolved ticket rated poorly.
    rows.append(
        {
            "ticket_id": "TKT-UNHAPPY",
            "created_at": pd.Timestamp("2024-01-01"),
            "category": "General",
            "priority": "Low",
            "status": "Resolved",
            "response_time_hrs": 1.0,
            "resolution_time_hrs": 3.0,
            "agent_id": "AGT-2",
            "customer_rating": 1,
            "issue_summary": "meh",
            "age_hrs": 10.0,
        }
    )
    return pd.DataFrame(rows)


def test_detect_resolution_time_outliers_flags_the_spike(df):
    outliers = anomaly.detect_resolution_time_outliers(df)
    flagged_ids = {a.ticket_id for a in outliers}
    assert "TKT-5" in flagged_ids  # the 60h ticket
    assert "TKT-0" not in flagged_ids


def test_detect_sla_breaches_flags_critical_open_ticket(df):
    breaches = anomaly.detect_sla_breaches(df)
    assert len(breaches) == 1
    assert breaches[0].ticket_id == "TKT-SLA"
    assert breaches[0].severity == "high"


def test_detect_low_satisfaction(df):
    low = anomaly.detect_low_satisfaction_clusters(df)
    assert len(low) == 1
    assert low[0].ticket_id == "TKT-UNHAPPY"


def test_detect_all_combines_and_sorts_by_severity(df):
    all_anomalies = anomaly.detect_all(df)
    severities = [a.severity for a in all_anomalies]
    # high-severity anomalies should come first
    assert severities == sorted(severities, key=lambda s: {"high": 0, "medium": 1, "low": 2}[s])


def test_summarize(df):
    all_anomalies = anomaly.detect_all(df)
    summary = anomaly.summarize(all_anomalies)
    assert summary["total"] == len(all_anomalies)
    assert sum(summary["by_type"].values()) == summary["total"]
