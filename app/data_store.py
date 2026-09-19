"""
Loads the support ticket CSV into a validated, typed pandas DataFrame and
keeps it in memory. For a dataset this size (500 rows) an in-memory DataFrame
queried with pandas is simpler, faster, and easier to reason about than
standing up a database — see README "Architecture" section for the trade-off
discussion (and how this would change at larger scale).
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path

import pandas as pd

from app import config

logger = logging.getLogger(__name__)

CATEGORIES = {"Billing", "Technical", "General"}
PRIORITIES = {"Low", "Medium", "High", "Critical"}
STATUSES = {"Open", "Resolved", "Escalated"}

REQUIRED_COLUMNS = [
    "ticket_id",
    "created_at",
    "category",
    "priority",
    "status",
    "response_time_hrs",
    "resolution_time_hrs",
    "agent_id",
    "customer_rating",
    "issue_summary",
]


class DataStoreError(RuntimeError):
    """Raised when the ticket CSV cannot be loaded or fails validation."""


class TicketDataStore:
    """Thread-safe, lazily-loaded, in-memory view of the ticket dataset."""

    def __init__(self, csv_path: Path | str | None = None):
        self._csv_path = Path(csv_path or config.CSV_PATH)
        self._df: pd.DataFrame | None = None
        self._lock = threading.Lock()
        self._reference_time: pd.Timestamp | None = None

    def load(self, force: bool = False) -> pd.DataFrame:
        """Load (or reload) the CSV. Safe to call repeatedly / concurrently."""
        with self._lock:
            if self._df is not None and not force:
                return self._df

            if not self._csv_path.exists():
                raise DataStoreError(f"Ticket CSV not found at {self._csv_path}")

            try:
                df = pd.read_csv(self._csv_path)
            except Exception as exc:  # noqa: BLE001 - surfaced as a clear error
                raise DataStoreError(f"Failed to parse CSV: {exc}") from exc

            missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
            if missing:
                raise DataStoreError(f"CSV is missing required columns: {missing}")

            df = self._clean(df)
            self._df = df
            # "Now", for the purposes of SLA/age calculations, is anchored to
            # the most recent ticket in the dataset rather than wall-clock
            # time, since the dataset is static/historical. This is called
            # out explicitly in the README as a documented assumption.
            self._reference_time = df["created_at"].max()
            logger.info(
                "Loaded %d tickets from %s (reference 'now' = %s)",
                len(df),
                self._csv_path,
                self._reference_time,
            )
            return self._df

    @staticmethod
    def _clean(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
        n_bad_dates = df["created_at"].isna().sum()
        if n_bad_dates:
            logger.warning("%d rows had unparseable created_at values", n_bad_dates)

        for col in ("response_time_hrs", "resolution_time_hrs"):
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df["customer_rating"] = pd.to_numeric(df["customer_rating"], errors="coerce")

        for col in ("category", "priority", "status", "agent_id", "ticket_id", "issue_summary"):
            df[col] = df[col].astype("string").str.strip()

        # Flag (but do not silently drop) rows with categorical values
        # outside the documented schema - useful for data-quality visibility.
        bad_category = ~df["category"].isin(CATEGORIES) & df["category"].notna()
        bad_priority = ~df["priority"].isin(PRIORITIES) & df["priority"].notna()
        bad_status = ~df["status"].isin(STATUSES) & df["status"].notna()
        for name, mask in (
            ("category", bad_category),
            ("priority", bad_priority),
            ("status", bad_status),
        ):
            if mask.any():
                logger.warning("%d rows have an unexpected %s value", mask.sum(), name)

        # Business-rule consistency: resolved tickets should have a
        # resolution time & rating; open/escalated ones should not. We don't
        # correct the data (that's a business decision) but we do compute a
        # helper column used by anomaly detection.
        now = df["created_at"].max()
        df["age_hrs"] = (now - df["created_at"]).dt.total_seconds() / 3600.0

        return df

    @property
    def reference_time(self) -> pd.Timestamp:
        self.load()
        return self._reference_time

    def get(self) -> pd.DataFrame:
        return self.load()

    def schema_description(self) -> str:
        """A compact, LLM-friendly description of the columns and their domains."""
        return (
            "Table `tickets` columns:\n"
            "- ticket_id (string, unique id, e.g. 'TKT-001')\n"
            "- created_at (datetime)\n"
            "- category (string, one of: Billing, Technical, General)\n"
            "- priority (string, one of: Low, Medium, High, Critical)\n"
            "- status (string, one of: Open, Resolved, Escalated)\n"
            "- response_time_hrs (float, hours to first response)\n"
            "- resolution_time_hrs (float, hours to resolution; null if not resolved)\n"
            "- agent_id (string, e.g. 'AGT-03')\n"
            "- customer_rating (integer 1-5; null if not resolved)\n"
            "- issue_summary (free text)\n"
            "- age_hrs (float, derived: hours between created_at and the dataset's most "
            "recent ticket, used as a proxy for 'now')"
        )


# Module-level singleton used by the FastAPI app / Streamlit UI so the CSV is
# parsed once per process rather than per-request.
store = TicketDataStore()
