# AI Support Ticket Intelligence

An AI-powered system for a 500-row customer support ticket dataset that:

1. Ingests the CSV and makes it queryable.
2. Answers natural-language questions about the data using an LLM.
3. Detects and flags anomalies (abnormal resolution times, unresolved high-priority tickets past SLA, etc.).
4. Exposes everything through **both** a REST API (FastAPI) and a minimal UI (Streamlit).

Built for the DOTMappers AI Engineer technical assessment.

---

## 1. Quick start

### Option A — Docker Compose (recommended, truly zero-cost, no signup)

This spins up a local **Ollama** LLM (no API key, no internet needed after the first
model pull), the FastAPI backend, and the Streamlit UI, with one command:

```bash
docker-compose up --build
```

Then open:
- UI: http://localhost:8501
- API docs (Swagger): http://localhost:8000/docs

First run will take a minute or two while Ollama pulls `llama3.2:1b` (~1.3GB). Subsequent
runs reuse the cached model via the `ollama_models` volume.

### Option B — Run locally without Docker

```bash
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Pick ONE LLM backend:
# (1) Local Ollama (free, offline, recommended)
#     Install from https://ollama.com, then:
ollama pull llama3.2:1b
ollama serve   # in a separate terminal, if not already running as a service

# (2) OR Groq free tier (no local install, but needs internet + a free key)
#     Get a key at https://console.groq.com/keys, then:
cp .env.example .env
#     edit .env: LLM_PROVIDER=groq, GROQ_API_KEY=...

# Start both services with one command:
./run.sh
# or run them separately:
#   uvicorn app.main:app --reload --port 8000
#   streamlit run ui/streamlit_app.py
```

### Option C — No LLM available at all

Set `LLM_PROVIDER=none` (see `.env.example`). This swaps in a small deterministic,
keyword-based query translator (`RuleBasedProvider` in `app/llm_providers.py`) so the
rest of the pipeline — data loading, safe query execution, anomaly detection, API, UI —
still runs end to end for a quick look. **This mode does not use an LLM** and only
covers question shapes similar to the assignment's own sample queries; it exists purely
as an offline fallback/demo mode, not as the system's actual NL-understanding solution.

### Running the tests

```bash
pytest -q
```

---

## 2. Architecture overview

```
                       ┌─────────────────┐
                       │  support_tickets │
                       │      .csv        │
                       └────────┬─────────┘
                                │  pandas.read_csv + validation
                                ▼
                       ┌─────────────────┐
                       │  TicketDataStore │  (app/data_store.py)
                       │  in-memory       │  - typed/cleaned DataFrame
                       │  DataFrame       │  - schema description for the LLM
                       └────────┬─────────┘
                                │
              ┌─────────────────┼─────────────────┐
              ▼                                    ▼
   ┌─────────────────────┐              ┌───────────────────────┐
   │   Query Engine       │              │   Anomaly Detection    │
   │ (app/query_engine.py)│              │   (app/anomaly.py)     │
   │                       │              │  - IQR outliers        │
   │ NL question           │              │    (per-category)      │
   │   → LLM → JSON spec   │              │  - SLA-breach rules     │
   │   → validated & safely│              │  - low-satisfaction     │
   │     executed on the   │              │    clusters             │
   │     DataFrame          │              └───────────────────────┘
   │   → LLM phrases the   │
   │     final answer       │
   └──────────┬────────────┘
              │
    ┌─────────┴──────────┐
    ▼                     ▼
┌─────────┐        ┌──────────────┐
│ FastAPI │◀──────▶│ Streamlit UI  │  (talks to the API over HTTP,
│ (main.py)│  HTTP  │ (streamlit_  │   not by importing app/ directly)
└─────────┘        │  app.py)      │
                    └──────────────┘

LLM backend is pluggable (app/llm_providers.py): Ollama (local) | Groq (free tier)
| none (rule-based fallback) — selected via the LLM_PROVIDER env var only.
```

### Key design decisions

**Why an in-memory pandas DataFrame instead of a database?**
500 rows fits comfortably in memory and pandas gives expressive, fast filtering/
aggregation with no extra moving parts. If the dataset grew to millions of rows or
needed concurrent writes, I'd move to DuckDB (keeps the pandas-like query ergonomics,
scales to much larger data, zero server to run) or Postgres if true multi-writer
concurrency were needed.

**Why translate NL → a whitelisted JSON query spec instead of having the LLM write
pandas/SQL code directly?**
Letting an LLM generate arbitrary code and `eval()`/`exec()` it is a real security
and reliability hazard — a hallucinated or adversarial prompt could do anything the
Python process can do. Instead, the LLM's job is narrowed to *translation*: turn the
question into a small JSON object (`operation`, `filters`, `group_by`, `agg_column`,
`agg_func`, `sort`, `limit`) built only from a whitelist of real columns/operators
(`app/query_engine.py::validate_spec`). That spec is then executed by hand-written,
unit-tested pandas code. This keeps the LLM doing what LLMs are good at (language
understanding) while keeping execution deterministic, safe, and easy to unit test —
and it's easy to explain and defend in a walkthrough.

The trade-off: this approach can't answer *arbitrary* analytical questions the schema
doesn't anticipate (e.g. a complex multi-hop question spanning several aggregations).
For a support-ticket Q&A tool this is an acceptable and, I'd argue, a *safer* scope
than free-form code generation.

**Two-step LLM usage.** The LLM is used twice per query: (1) to produce the structured
query spec, and (2) to phrase the raw numeric/tabular result as a natural sentence. Step
2 is optional (`phrase_with_llm=false` skips it) and always has a deterministic templated
fallback, so a flaky phrasing call never hides an answer the user actually asked for.

**Retry-with-feedback.** If the LLM's first JSON response fails to parse or fails
validation, the query engine retries once, feeding the validation error back into the
prompt (`app/query_engine.py::answer_question`). If both attempts fail, the system falls
back to a safe default query (an unfiltered count) rather than crashing, and reports
this clearly in the response (`used_llm_fallback`, `warnings`).

**Anomaly detection is rule-based/statistical, not ML.** For 500 rows, a transparent
IQR-outlier + explicit-SLA-rule approach is more explainable and easier to tune than
training a model, and matches exactly the two examples in the assignment brief
(abnormally long resolution times; unresolved high-priority tickets past a time
threshold). Resolution/response time outliers are computed **per category** (Tukey's
fences, 1.5×IQR by default) rather than globally, since Technical tickets legitimately
take longer than a "how do I export a CSV" General ticket — a global threshold would
just flag every Technical ticket.

**"Now" is anchored to the dataset, not wall-clock time.** The dataset is historical
(all tickets are from Jan–Mar 2024), so "unresolved for more than 24 hours" is computed
relative to the most recent `created_at` timestamp in the CSV, not today's date. This is
called out explicitly here and in `app/data_store.py` since it's a meaningful assumption
that would need revisiting for a live system.

**UI calls the API over HTTP rather than importing it.** This exercises the actual REST
API deliverable (rather than bypassing it) and means the UI and API could scale or be
deployed independently later.

---

## 3. Models / tools used

| Component        | Choice                                   | Why |
|-------------------|-------------------------------------------|-----|
| API framework      | FastAPI + Uvicorn                          | async, automatic OpenAPI docs, pydantic validation |
| Data layer         | pandas                                     | simplest fit for a 500-row static CSV |
| LLM (default)      | Ollama, `llama3.2:1b`                      | free, local, no signup, small enough to run on a laptop CPU |
| LLM (alternative)  | Groq free tier, `llama-3.1-8b-instant`     | free, fast, no local install, if the evaluator prefers not to run Ollama |
| UI                 | Streamlit                                  | fastest way to a usable multi-tab UI (query, anomalies, dashboard, browse) |
| Tests              | pytest                                     | unit tests for the safe-execution layer and anomaly detectors |

Swapping between Ollama/Groq/none is a single environment variable
(`LLM_PROVIDER`) — see `.env.example`.

---

## 4. Example queries & outputs

All of these were run against the actual dataset via `POST /query` (rule-based
provider shown for reproducibility here; phrasing is more natural with a real LLM —
see the "phrase_with_llm" note below).

**"How many tickets are currently open?"**
```json
{"answer": "There are 111 matching ticket(s).", "result": {"count": 111}}
```

**"How many critical tickets are unresolved?"**
```json
{"answer": "There are 31 matching ticket(s).",
 "result": {"count": 31},
 "query_spec": {"operation": "count",
   "filters": [{"column": "priority", "op": "==", "value": "Critical"},
               {"column": "status", "op": "!=", "value": "Resolved"}]}}
```

**"Which agent has the lowest average customer rating?"**
```json
{"answer": "AGT-08: 3.48 (n=42)",
 "result": [{"agent_id": "AGT-08", "value": 3.48, "n": 42}]}
```

**"What is the average customer rating for Technical category tickets?"**
```json
{"answer": "The mean of customer_rating is 3.74 (n=104).",
 "result": {"value": 3.74, "n": 104}}
```

**"Are there any anomalies in resolution times this week?"** → `GET /anomalies?types=long_resolution_time`
```json
{"summary": {"total": 22, "by_type": {"long_resolution_time": 22}, "by_severity": {"high": 9, "medium": 13}},
 "anomalies": [{"ticket_id": "TKT-023", "type": "long_resolution_time",
   "detail": "Resolution took 76.1h, well above the typical 11.4h for Billing tickets (threshold 44.5h).",
   "severity": "high"}, "..."]}
```

With a real LLM configured (`phrase_with_llm=true`, the default), the `answer` field
reads more like a sentence a person wrote, e.g. *"Yes — 31 critical tickets are still
unresolved, which is a meaningful chunk of your open queue."* rather than the templated
fallback shown above; the underlying `result`/`query_spec` are unaffected either way.

---

## 5. API reference

| Method | Path         | Purpose |
|--------|--------------|---------|
| GET    | `/health`    | Liveness + whether the configured LLM is reachable |
| POST   | `/query`     | `{"question": "...", "phrase_with_llm": true}` → NL answer + structured result |
| GET    | `/anomalies` | Optional `?types=long_resolution_time&types=unresolved_sla_breach` |
| GET    | `/stats`     | Aggregate counts/averages, used by the UI dashboard |
| GET    | `/tickets`   | Paginated raw listing with `status`/`category`/`priority` filters |

Full interactive docs at `/docs` (Swagger) once the API is running.

---

## 6. Known limitations

- **Query coverage is bounded by the JSON query-spec grammar.** Multi-step or highly
  compound questions (e.g. "compare this month's Billing rating to last month's
  Technical rating and tell me which agent improved the most") aren't supported — the
  spec supports one filter/aggregate/group-by pass per question, not query composition.
- **"Now" is dataset-relative, not wall-clock**, as noted above — a deliberate choice
  for this static historical dataset, but would need to change for a live feed.
- **The rule-based fallback (`LLM_PROVIDER=none`) is not a real NL understanding
  system** — it recognizes a handful of literal question shapes and exists only as an
  offline safety net, not as the assignment's LLM solution.
- **`llama3.2:1b` is a small model.** It's fast and free but occasionally produces
  slightly malformed JSON on more unusual phrasings; the one-shot retry-with-feedback
  handles most of these, and any unrecovered failure falls back to a safe default query
  rather than an error. A larger local model (e.g. `llama3.1:8b`) or Groq's hosted model
  will be noticeably more robust at the cost of more RAM / needing internet, respectively.
- **No authentication/rate limiting** — this is a local assessment prototype; CORS is
  wide open and there's no API key on the endpoints. Would need both before any real
  deployment.
- **No persistence layer for anomalies** — they're recomputed on every request rather
  than stored/tracked over time (no "this anomaly was already seen" de-duplication).
- **Single CSV snapshot** — there's no ingestion pipeline for streaming/incrementally
  arriving tickets; `TicketDataStore.load()` re-reads the whole file if called with
  `force=True`, but there's no file-watching or webhook-triggered reload.

## 7. What I'd improve with more time

- Swap the JSON query-spec grammar for a small set of composable primitives (so two
  aggregations could be chained) if broader query coverage were needed.
- Add response caching for repeated identical questions.
- Add a lightweight eval harness (a fixed set of Q&A pairs with expected `query_spec`
  shapes) to catch prompt-drift regressions when changing models.
- Track anomalies over time (persist to SQLite) to distinguish "new since last check"
  from "still outstanding."
