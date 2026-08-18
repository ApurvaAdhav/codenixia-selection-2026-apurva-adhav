# AI Business Intelligence & Decision Support Assistant

Built for the **CODENIXIA AI/ML Internship Challenge**.

A business manager uploads sales data, and the system tells them **what
changed, why it changed, and what to do about it** — grounded entirely in
real computed numbers, with a conversational follow-up interface.

```
CSV/Excel Data → Python/Pandas cleaning → ML analysis (Isolation Forest +
forecast) → AI Agent (tool routing) → RAG (FAISS, local KB) → LLM →
Structured insights + recommendations → Follow-up questions
```

---

## 1. Folder structure

```
biassistant/
├── app/
│   ├── main.py                 # FastAPI entrypoint
│   ├── config.py                # All env-var configuration in one place
│   ├── logging_config.py        # Shared logging setup
│   ├── api/
│   │   ├── routes.py             # /health /upload /analyze /ask
│   │   └── schemas.py            # Pydantic request/response models
│   ├── data/
│   │   ├── processing.py         # CSV/Excel loading, cleaning, validation
│   │   ├── analytics.py          # analyze_sales/region/products logic
│   │   ├── anomaly.py            # Isolation Forest anomaly detection
│   │   ├── forecast.py           # Linear trend forecasting
│   │   └── store.py              # In-memory "active dataset" singleton
│   ├── agent/
│   │   ├── core.py               # THE AGENT: routing + memory + LLM glue
│   │   ├── tools.py              # The 6 tool functions + registry
│   │   └── llm_client.py         # Gemini API (google-genai) wrapper + fallback
│   └── rag/
│       └── retriever.py          # FAISS + TF-IDF local knowledge search
├── data/
│   └── knowledge_base/           # Policies, KPI defs, incident reports (.md)
├── .streamlit/config.toml        # Pins the app's existing color theme
├── streamlit_app.py              # Dashboard (KPIs, charts, chat)
├── tests/                        # pytest suite (44 tests)
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
├── README.md / AI_USAGE.md / DECISION_LOG.md / DEBUGGING_REPORT.md
```

---

## 2. Installation & run commands

```bash
# 1. Create venv and install deps
python3 -m venv .venv
source .venv/bin/activate         # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env
# Edit .env and set GEMINI_API_KEY (optional — app works without it,
# using the deterministic template fallback described below).

# 3. Run the FastAPI backend
uvicorn app.main:app --reload --port 8000
# → API docs at http://localhost:8000/docs

# 4. Run the Streamlit dashboard (separate terminal)
streamlit run streamlit_app.py
# → Dashboard at http://localhost:8501
```

> **No sample/demo data is bundled.** Both the API and the dashboard start
> with an empty active dataset - upload a CSV/Excel file (via the sidebar,
> or `POST /upload`) before `/analyze` or `/ask` will work. Required
> columns: `Date, Region, Product, Category, Customer_Segment, Quantity,
> Revenue, Cost, Profit`.

The Streamlit app calls the Python modules directly (no HTTP hop needed for
the demo); the FastAPI server is available independently for programmatic
API access, and is what you'd point a real frontend at.

---

## 3. Docker commands

```bash
# Build
docker build -t bi-assistant .

# Run the API only
docker run --rm -p 8000:8000 --env-file .env bi-assistant

# Run the dashboard only
docker run --rm -p 8501:8501 --env-file .env bi-assistant \
    streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0

# Or run BOTH together with docker-compose
docker compose up --build
# → API:       http://localhost:8000/docs
# → Dashboard: http://localhost:8501
```

> **Note:** Docker wasn't available in the sandbox this project was built
> in, so the image build itself wasn't executed there. The Dockerfile
> installs the exact same `requirements.txt` that was verified working
> directly (see `DEBUGGING_REPORT.md`), so it should build cleanly on any
> machine with Docker + internet access. Please verify with `docker build`
> on your machine and let me know if anything needs adjusting.

---

## 4. Test commands

```bash
pytest                      # run all 44 tests
pytest -v                   # verbose
pytest tests/test_agent.py  # just the agent/multi-turn tests
pytest --tb=short           # shorter tracebacks (already default via pytest.ini)
```

All 44 tests pass without any API key or network access — they exercise
the deterministic fallback path, which is also the path that proves
numbers are never invented (see `test_agent.py::test_tool_results_are_grounded_not_invented`).

---

## 5. Sample questions to try

Upload a CSV/Excel file first (no dataset is bundled). Then, in the
Streamlit "Ask the Assistant" tab (buttons are pre-wired for these):

1. **"Why did sales drop?"** → structured first answer (see format below)
2. **"What about Region West?"** → follow-up, reuses period context
3. **"Which product caused it?"** → follow-up, reuses region context
4. **"What is expected next month?"** → forecast, still region-scoped

Other things worth trying:
- "Show me anomalies"
- "How is Product C doing?"
- "What's our profit margin policy?" (pure RAG/policy question)

Example first-turn answer format (required by the challenge spec):
```
Sales dropped: 5.0%
Main contributors: Region West -29%, Region East -4%, Product A -10%, Product B -4%
Key insight: Region West is the single largest driver of the change (-29.0%, -8700 revenue impact).
Detected anomaly: Product A in West below normal on 2024-12-05
Recommendation: Review the top contributor above and check recent incident reports for a known cause.
```

---

## 6. Component explanations (plain language)

| Component | What it does | File(s) |
|---|---|---|
| **Data processing** | Loads CSV/Excel, validates required columns, parses types, drops bad rows, recomputes missing Profit, dedupes. | `app/data/processing.py` |
| **Anomaly detection** | Isolation Forest per Region+Product, run on **detrended** daily revenue (% deviation from a 14-day rolling baseline) so trend/seasonality doesn't drown out real anomalies. Only reports anomalies from the last 60 days. | `app/data/anomaly.py` |
| **Forecasting** | Linear regression (`day_index → revenue`) projected forward 30 days, with an R² confidence note. | `app/data/forecast.py` |
| **Analytics** | Period-over-period comparison (last N days vs prior N days) at the overall / region / product level, ranked by dollar impact. | `app/data/analytics.py` |
| **AI Agent** | Deterministic tool router: extracts region/product/intent from the question (using actual values in the dataset), decides which tools to call, always queries RAG, then asks the LLM to narrate — or falls back to a template if the LLM is unavailable. Holds per-session conversation memory. | `app/agent/core.py` |
| **Tools** | The 6 required tools (`analyze_sales`, `analyze_region`, `analyze_products`, `detect_anomalies`, `predict_next_month`, `search_business_knowledge`), each a thin wrapper with a description used by the router. | `app/agent/tools.py` |
| **RAG** | Local knowledge base (policies, KPI definitions, incident reports) chunked by section, embedded with **TF-IDF** (not a neural model — see `DECISION_LOG.md`), indexed with **FAISS** for cosine-similarity search. | `app/rag/retriever.py`, `data/knowledge_base/*.md` |
| **LLM + fallback** | Calls Google's Gemini API with a strict "only use these numbers" system prompt. If unavailable (no key, timeout, error), a deterministic template builds the same structured answer directly from tool results — same numbers, less prose polish. | `app/agent/llm_client.py`, `app/agent/core.py::_template_fallback` |
| **FastAPI** | `/health`, `/upload`, `/analyze`, `/ask` — thin routes, all logic lives in `app/data`/`app/agent`. | `app/api/routes.py` |
| **Streamlit dashboard** | KPI cards, 4 charts (trend, by region, by product, top movers), anomaly table, forecast panel, and a chat tab wired to the same agent. | `streamlit_app.py` |

---

## 7. How this satisfies the challenge requirements

| # | Requirement | Where |
|---|---|---|
| 1 | Upload dataset with required columns (no bundled sample data — upload is required) | Sidebar uploader (Streamlit) or `POST /upload` (API); validated in `app/data/processing.py` |
| 2 | Pandas cleaning/validation | `app/data/processing.py` — type coercion, dedup, missing-value handling, Profit recompute |
| 3 | Isolation Forest anomaly detection | `app/data/anomaly.py` |
| 4 | Simple forecasting | `app/data/forecast.py` — linear regression |
| 5 | AI Agent with the 6 named tools | `app/agent/tools.py` (`TOOL_REGISTRY`), routed in `app/agent/core.py` |
| 6 | RAG with local knowledge base | `app/rag/retriever.py` + `data/knowledge_base/*.md`, FAISS index |
| 7 | LLM never invents numbers | Strict system prompt in `core.py::SYSTEM_PROMPT` + fallback template that only ever echoes tool_results |
| 8 | Multi-turn context ("Why did sales drop?" → "What about Region West?" → ...) | `ConversationSession` in `app/agent/core.py`, tested in `tests/test_agent.py` |
| 9 | Structured first answer | `SYSTEM_PROMPT` rule #3 (LLM path) + `_template_fallback` (no-LLM path) |
| 10 | Streamlit dashboard: KPIs, charts, AI analysis | `streamlit_app.py` |
| 11 | FastAPI: `/health /upload /analyze /ask` | `app/api/routes.py` |
| 12 | Logging, error handling, LLM fallback | `app/logging_config.py`, try/except + `HTTPException` throughout routes, `LLMUnavailableError` fallback |
| 13 | pytest tests | `tests/` — 44 tests across processing, analytics, anomaly, forecast, RAG, agent, API |
| 14 | Dockerfile + `.env.example` | `Dockerfile`, `docker-compose.yml`, `.env.example` |
| 15 | README / AI_USAGE / DECISION_LOG / DEBUGGING_REPORT | This file + the other three in the repo root |

---

## 8. Code/functions worth understanding for the interview

These are the pieces most likely to come up in a technical walkthrough:

1. **`app/agent/core.py::ask()`** — the whole pipeline in one function:
   entity extraction → tool routing → RAG → LLM/fallback → memory update.
   Start here to explain the architecture.
2. **`app/agent/core.py::_extract_entities()`** — how the agent figures out
   what a follow-up question is "about" using word-boundary regex against
   real dataset values, and the `_is_followup()` heuristic for context
   inheritance. (This is where a real bug was caught and fixed — see
   `DEBUGGING_REPORT.md` — good story for an interview.)
3. **`app/agent/core.py::_template_fallback()`** — proves the "never
   invents numbers" requirement even without any LLM: every line is built
   directly from `tool_results`.
4. **`app/data/anomaly.py::detect_anomalies()`** — the detrending step
   (rolling 14-day baseline → % residual → Isolation Forest on residuals)
   is a deliberate design choice worth explaining: raw revenue trends and
   weekly seasonality would otherwise swamp real anomalies.
5. **`app/data/analytics.py::_split_periods()` / `_contributor_breakdown()`**
   — the period-over-period comparison + "rank by absolute dollar impact,
   not %" logic that powers all three `analyze_*` tools.
6. **`app/rag/retriever.py::KnowledgeBaseRetriever`** — TF-IDF + FAISS
   IndexFlatIP over L2-normalized vectors = cosine similarity, entirely
   local/offline.
7. **`app/agent/tools.py::TOOL_REGISTRY`** — the single place that defines
   what tools exist; add a 7th tool here without touching `core.py`'s
   control flow.
