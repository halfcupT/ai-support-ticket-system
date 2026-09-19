"""
Anomaly detection over the ticket dataset.

Two complementary approaches are combined, matching the two examples given
in the assignment brief:

1. Statistical outliers - tickets whose response/resolution time is
   abnormally high relative to *their own category*, using the IQR
   ("Tukey's fences") method. Category-relative (rather than global)
   thresholds avoid flagging every Technical ticket just because Technical
   issues legitimately take longer than a "how do I change my password"
   General ticket.
2. Rule-based SLA breaches - unresolved High/Critical tickets older than a
   configurable threshold (default 24h). This is a business rule, not a
   statistical judgement, so it's implemented as an explicit rule rather
   than forced through the outlier detector.

Both are cheap, transparent, and explainable — important for an "anomaly"
feature that a support manager will actually act on; a black-box ML model
would be overkill for 500 rows and harder to justify in the walkthrough.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from app import config


@dataclass
class Anomaly:
    ticket_id: str
    type: str
    detail: str
    severity: str  # "low" | "medium" | "high"
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ticket_id": self.ticket_id,
            "type": self.type,
            "detail": self.detail,
            "severity": self.severity,
            **self.data,
        }


def _iqr_outliers(df: pd.DataFrame, column: str, group_col: str, multiplier: float) -> pd.DataFrame:
    """Return rows whose `column` is above Q3 + multiplier*IQR within their group."""
    flagged_frames = []
    for group_value, group_df in df.dropna(subset=[column]).groupby(group_col):
        series = group_df[column]
        if len(series) < 4:
            continue  # not enough data for a meaningful IQR in this group
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        upper_bound = q3 + multiplier * iqr
        outliers = group_df[group_df[column] > upper_bound].copy()
        if not outliers.empty:
            outliers["_upper_bound"] = upper_bound
            outliers["_group_median"] = series.median()
            flagged_frames.append(outliers)
    if not flagged_frames:
        return df.iloc[0:0]
    return pd.concat(flagged_frames)


def detect_resolution_time_outliers(
    df: pd.DataFrame, multiplier: float = config.IQR_MULTIPLIER
) -> list[Anomaly]:
    resolved = df[df["status"] == "Resolved"]
    outliers = _iqr_outliers(resolved, "resolution_time_hrs", "category", multiplier)
    anomalies = []
    for _, row in outliers.iterrows():
        anomalies.append(
            Anomaly(
                ticket_id=row["ticket_id"],
                type="long_resolution_time",
                detail=(
                    f"Resolution took {row['resolution_time_hrs']:.1f}h, well above the "
                    f"typical {row['_group_median']:.1f}h for {row['category']} tickets "
                    f"(threshold {row['_upper_bound']:.1f}h)."
                ),
                severity="high" if row["resolution_time_hrs"] > row["_upper_bound"] * 1.5 else "medium",
                data={
                    "category": row["category"],
                    "resolution_time_hrs": float(row["resolution_time_hrs"]),
                    "category_median_hrs": round(float(row["_group_median"]), 2),
                    "threshold_hrs": round(float(row["_upper_bound"]), 2),
                },
            )
        )
    return anomalies


def detect_response_time_outliers(
    df: pd.DataFrame, multiplier: float = config.IQR_MULTIPLIER
) -> list[Anomaly]:
    outliers = _iqr_outliers(df, "response_time_hrs", "category", multiplier)
    anomalies = []
    for _, row in outliers.iterrows():
        anomalies.append(
            Anomaly(
                ticket_id=row["ticket_id"],
                type="slow_first_response",
                detail=(
                    f"First response took {row['response_time_hrs']:.1f}h, above the "
                    f"typical {row['_group_median']:.1f}h for {row['category']} tickets."
                ),
                severity="medium",
                data={
                    "category": row["category"],
                    "response_time_hrs": float(row["response_time_hrs"]),
                    "category_median_hrs": round(float(row["_group_median"]), 2),
                    "threshold_hrs": round(float(row["_upper_bound"]), 2),
                },
            )
        )
    return anomalies


def detect_sla_breaches(
    df: pd.DataFrame,
    priorities: set[str] = config.SLA_PRIORITIES,
    hours: float = config.SLA_BREACH_HOURS,
) -> list[Anomaly]:
    unresolved = df[df["status"].isin(["Open", "Escalated"])]
    breaches = unresolved[
        unresolved["priority"].isin(priorities) & (unresolved["age_hrs"] > hours)
    ]
    anomalies = []
    for _, row in breaches.iterrows():
        anomalies.append(
            Anomaly(
                ticket_id=row["ticket_id"],
                type="unresolved_sla_breach",
                detail=(
                    f"{row['priority']} priority ticket has been {row['status']} for "
                    f"{row['age_hrs']:.1f}h (SLA threshold: {hours:.0f}h)."
                ),
                severity="high" if row["priority"] == "Critical" else "medium",
                data={
                    "priority": row["priority"],
                    "status": row["status"],
                    "age_hrs": round(float(row["age_hrs"]), 2),
                    "sla_threshold_hrs": hours,
                },
            )
        )
    return anomalies


def detect_low_satisfaction_clusters(df: pd.DataFrame, rating_threshold: int = 2) -> list[Anomaly]:
    """Bonus signal: resolved tickets rated very poorly despite being 'Resolved'."""
    low = df[(df["status"] == "Resolved") & (df["customer_rating"] <= rating_threshold)]
    anomalies = []
    for _, row in low.iterrows():
        anomalies.append(
            Anomaly(
                ticket_id=row["ticket_id"],
                type="low_satisfaction_despite_resolution",
                detail=(
                    f"Marked Resolved but customer rated it {int(row['customer_rating'])}/5 "
                    f"(agent {row['agent_id']})."
                ),
                severity="low",
                data={"customer_rating": int(row["customer_rating"]), "agent_id": row["agent_id"]},
            )
        )
    return anomalies


def detect_all(
    df: pd.DataFrame,
    types: set[str] | None = None,
) -> list[Anomaly]:
    """
    Run all anomaly detectors and return a combined, ticket-sorted list.
    `types` optionally restricts to a subset:
      {"long_resolution_time", "slow_first_response", "unresolved_sla_breach",
       "low_satisfaction_despite_resolution"}
    """
    detectors = {
        "long_resolution_time": detect_resolution_time_outliers,
        "slow_first_response": detect_response_time_outliers,
        "unresolved_sla_breach": detect_sla_breaches,
        "low_satisfaction_despite_resolution": detect_low_satisfaction_clusters,
    }
    selected = types or set(detectors)
    results: list[Anomaly] = []
    for name, fn in detectors.items():
        if name in selected:
            results.extend(fn(df))

    severity_rank = {"high": 0, "medium": 1, "low": 2}
    results.sort(key=lambda a: (severity_rank.get(a.severity, 3), a.ticket_id))
    return results


def summarize(anomalies: list[Anomaly]) -> dict[str, Any]:
    by_type: dict[str, int] = {}
    by_severity: dict[str, int] = {}
    for a in anomalies:
        by_type[a.type] = by_type.get(a.type, 0) + 1
        by_severity[a.severity] = by_severity.get(a.severity, 0) + 1
    return {"total": len(anomalies), "by_type": by_type, "by_severity": by_severity}
