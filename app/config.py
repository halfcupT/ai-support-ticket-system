"""
Central configuration for the AI Support Ticket Intelligence system.

Everything is driven by environment variables so the system can be pointed at
either a local Ollama model (fully free, fully offline) or the Groq free-tier
API (free, but requires an API key and internet access) without touching code.
"""
from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# --- Data -------------------------------------------------------------
CSV_PATH = Path(os.getenv("TICKETS_CSV_PATH", BASE_DIR / "data" / "support_tickets.csv"))

# --- LLM provider -------------------------------------------------------
# "ollama"  -> local model via Ollama's REST API (default, zero cost, no signup)
# "groq"    -> Groq's OpenAI-compatible free-tier API (needs GROQ_API_KEY)
# "none"    -> disables the LLM and falls back to a deterministic rule-based
#              parser. Only used so the system still runs end-to-end (e.g. in
#              CI or offline demos) if nobody has an LLM configured; the
#              assignment's NL understanding requirement is met by the
#              ollama/groq paths.
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").lower()

# Ollama settings
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:1b")

# Groq settings
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
GROQ_BASE_URL = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")

# Generic LLM behaviour
LLM_TIMEOUT_SECONDS = float(os.getenv("LLM_TIMEOUT_SECONDS", "30"))
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "2"))

# --- Anomaly detection thresholds --------------------------------------
# Multiplier applied to the IQR when flagging resolution/response time outliers.
IQR_MULTIPLIER = float(os.getenv("IQR_MULTIPLIER", "1.5"))
# Tickets at these priorities that are still open/escalated past this many
# hours are flagged as SLA-breach anomalies.
SLA_PRIORITIES = {"High", "Critical"}
SLA_BREACH_HOURS = float(os.getenv("SLA_BREACH_HOURS", "24"))
