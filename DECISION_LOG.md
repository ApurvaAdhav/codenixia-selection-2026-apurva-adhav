# DECISION_LOG.md

Key design decisions, what alternatives were considered, and why. Written
so you can defend or challenge any of these choices in a technical
interview.

---

## 1. Deterministic rule-based tool router instead of an LLM-driven ReAct agent

**Decision**: the agent decides which tools to call using plain Python
logic (keyword matching + entity extraction against actual dataset values
in `app/agent/core.py::_route_tools`), not by asking an LLM to choose tools
via function-calling in a loop.

**Alternatives considered**: Anthropic's native tool-use API (LLM decides
which tool to call, in a loop, potentially multiple round trips), or a
framework like LangChain/LlamaIndex agents.

**Why the simpler approach won**:
- The challenge explicitly says "keep it simple" and "do NOT use... complex
  frameworks" / "multi-agent architecture."
- It's fully unit-testable with zero network calls or API costs — you can
  assert `"analyze_region" in tool_calls` without ever hitting an LLM.
- It's deterministic and debuggable: given a question, you can predict
  exactly which tools will run by reading one function, not by trusting an
  LLM's judgment call that might vary run to run.
- It still satisfies "AI Agent with tools" — the LLM's role is narrowed to
  what LLMs are actually good at (turning structured data into readable
  prose), while tool selection — which benefits from precision — is done
  precisely.

**Trade-off accepted**: this router can't handle genuinely novel phrasings
outside its keyword/entity patterns as gracefully as a true LLM-driven
agent could. Mitigated by: `analyze_sales` always runs as a baseline, and
RAG search always runs regardless of intent, so even an unmatched question
still gets *some* grounded context.

---

## 2. TF-IDF instead of a neural embedding model for RAG

**Decision**: `app/rag/retriever.py` uses scikit-learn's `TfidfVectorizer`
+ FAISS `IndexFlatIP`, not `sentence-transformers`.

**Why**: originally built with `sentence-transformers` (all-MiniLM-L6-v2).
During build/testing, the sandbox's network policy blocked the
HuggingFace Hub download required to fetch that model. Rather than treating
this purely as a sandbox limitation to route around, it was evaluated on
its own merits and kept as the permanent choice:
- **Zero external dependency at runtime** — no model download, no
  HuggingFace account/rate limits, no risk of the app failing on a machine
  without internet access to HF. This directly serves the challenge's
  "keep it simple" and offline-friendly local-FAISS requirement.
- The knowledge base is small (3 markdown files, ~20 chunks) — TF-IDF's
  weaker semantic matching (vs neural embeddings) is a non-issue at this
  scale; testing showed it correctly retrieves the exact matching incident
  report for realistic queries (see `tests/test_rag.py`).
- No GPU/large dependency footprint (`sentence-transformers` pulls in
  `torch`, which is large and slow to install).

**Trade-off accepted**: TF-IDF won't generalize to paraphrased or
semantically-similar-but-lexically-different queries as well as a neural
embedding model would. If the knowledge base grows much larger or queries
get more paraphrased, swapping in a neural embedding model only requires
changing `KnowledgeBaseRetriever._embed()` — the rest of the app (which only
calls `.search()`) doesn't need to change.

---

## 3. Anomaly detection: detrend before Isolation Forest, and only report recent anomalies

**Decision**: `app/data/anomaly.py` computes each day's % deviation from a
14-day trailing rolling average ("residual"), runs Isolation Forest on that
residual series (not raw revenue), and only reports anomalies from the
last 60 days.

**Why**: raw daily revenue has a real upward trend (~15%/year in the
sample data) and weekly seasonality (weekday vs weekend). Feeding raw
levels into Isolation Forest confuses "revenue is naturally higher in
December" with a true anomaly, and (during testing) buried the real
injected anomaly (Product A/West decline) under noisy single-day spikes
scattered across the whole year. Detrending isolates genuine deviations
from a series' own recent behavior. The recency filter (report only the
last 60 days) matches the actual business question — "what's wrong right
now" — rather than surfacing any outlier day from a year ago.

**Trade-off accepted**: a 14-day rolling window needs at least ~15 data
points per (Region, Product) group to produce a residual series; sparser
datasets will have fewer groups eligible for anomaly detection (handled
gracefully — those groups are just skipped, not errored).

---

## 4. Per-(Region, Product) models instead of one global anomaly model

**Decision**: Isolation Forest is fit separately for each Region×Product
combination.

**Why**: a single global model would treat "Product A is just bigger than
Product E" as an outlier, since it would be comparing absolute revenue
levels across series with very different scales. Per-group modeling means
each series is only ever compared to its own history.

**Trade-off accepted**: more models to fit (one per group instead of one
total), but each is cheap (a few hundred data points, `n_estimators=100`),
so this has no real performance cost at this data scale.

---

## 5. Simple linear regression for forecasting, not Prophet/ARIMA

**Decision**: `app/data/forecast.py` fits `sklearn.linear_model.LinearRegression`
on `day_index → revenue` and projects forward.

**Why**: the challenge asks for "simple sales forecasting" and explicitly
discourages unnecessary complexity/frameworks. Linear regression is
trivial to explain in an interview ("ordinary least squares on day index
vs revenue"), has zero exotic dependencies, and the R² value it reports
gives an honest, simple confidence signal — "this trend explains X% of the
variance" — without pretending to model seasonality or holidays that a
heavier model (Prophet, ARIMA) would attempt.

**Trade-off accepted**: no seasonality modeling, so forecasts for highly
seasonal series will be less accurate than a purpose-built time-series
model. Explicitly surfaced to the user via the `confidence_note` field
(e.g. "Low confidence" when R² < 0.3) rather than hidden.

---

## 6. In-memory session store and single "active dataset" (no database)

**Decision**: conversation history (`app/agent/core.py::_sessions`) and the
currently active dataset (`app/data/store.py`) are plain module-level
Python dicts/variables, not backed by Redis/Postgres/etc.

**Why**: the challenge explicitly says no authentication, keep it simple,
and this is a single-process demo/interview project, not a multi-tenant
production service. A database would add setup complexity and a moving
part to explain with no corresponding benefit at this scope.

**Trade-off accepted**: state is lost on process restart, and there's
exactly one "active dataset" per running process (uploading a new file
replaces it globally, not per-user). Documented explicitly in both
`store.py`'s docstring and here, so it's a known, deliberate limitation,
not an oversight — the natural next step for production use would be to
key both stores by user/session ID and back them with a real store.

---

## 7. Rule-based entity extraction using the ACTUAL dataset's values

**Decision**: `_extract_entities()` looks for region/product mentions by
checking the question text against the real, unique values present in the
uploaded dataset (`df["Region"].unique()`), not a hardcoded list.

**Why**: this makes the agent automatically work correctly with any
uploaded dataset's actual region/product names, without hardcoding "North/
South/East/West" anywhere in the agent logic — upload a dataset with
different region names and entity extraction still works.

**Bug caught and fixed because of this design** (full story in
`DEBUGGING_REPORT.md`): plain substring matching (`"product c" in
question.lower()`) falsely matched inside "product **c**aused" — fixed
with word-boundary regex (`\bproduct c\b`).

---

## 8. Post-handoff changes: Gemini provider swap + sample-data removal

**Decision**: two changes made after the entries above were written.

1. **LLM provider: Anthropic → Google Gemini (`google-genai`).** Isolated
   entirely to `app/agent/llm_client.py` (rewritten) + `GEMINI_*` config in
   `app/config.py`/`.env.example`. `LLMClient.generate()`,
   `LLMUnavailableError`, and `get_llm_client()` kept the exact same
   signatures on purpose, so `app/agent/core.py`, `app/agent/tools.py`, and
   everything else in the agent needed zero changes - they only ever talk
   to this interface, never to a vendor SDK.
2. **Sample/demo data removed.** `data/sample/sales_data.csv` and
   `scripts/generate_sample_data.py` were deleted; `app/data/store.py` no
   longer auto-loads anything - `get_active_dataset()` returns `None` until
   a real file is uploaded. The API (`/analyze`, `/ask`) and the Streamlit
   dashboard both now require an explicit upload first (400 / empty-state
   respectively). `tests/test_api.py` was updated accordingly (uploads a
   small seed CSV via a fixture instead of relying on startup auto-load).

**Also added**: a "Key insight" step in the required structured-answer
format (`SYSTEM_PROMPT` + `_template_fallback` in `app/agent/core.py`) -
one sentence identifying the single largest dollar-impact driver across
region+product contributors, grounded in the same tool-result fields
`Main contributors` already uses.

**What did NOT change**: `app/data/analytics.py`, `app/data/anomaly.py`,
`app/data/forecast.py`, `app/agent/tools.py`, `app/rag/retriever.py` are
byte-identical to before. `core.py`'s routing/entity-extraction/session
logic is untouched - only the two prompt/template strings above changed.
