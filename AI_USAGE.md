# AI_USAGE.md

Transparency note on how Claude (Anthropic) was used to build this project,
as requested for the CODENIXIA AI/ML Internship Challenge.

## What was AI-generated vs human-directed

This entire codebase was written by Claude (Sonnet) acting as a coding
agent, working from a detailed specification provided by the user (the
challenge PDF requirements, restated as an explicit list of features,
constraints, and the exact workflow diagram). The user directed the work
at the architecture/requirements level; Claude made the implementation
decisions within those constraints and verified its own output by running
the code, not just writing it.

## How the work was done, step by step

1. **Planning**: broke the workflow (`CSV → Pandas → ML → Agent → RAG →
   LLM → Insights`) into modules that map 1:1 onto the required folder
   structure, so each challenge requirement has one obvious file.
2. **Bottom-up build**: data cleaning → analytics → anomaly detection →
   forecasting → RAG → agent → LLM client → FastAPI → Streamlit, so each
   layer could be tested in isolation before the next layer depended on it.
3. **Synthetic data generation**: wrote `scripts/generate_sample_data.py`
   to produce a realistic dataset with a *deliberately injected* anomaly
   (Product A / Region West demand drop), so the anomaly detector,
   forecaster, and agent would all have a real, consistent signal to find
   and explain — rather than hoping a random dataset happened to contain
   an interesting story.
4. **Verification, not just generation**: after writing each module, Claude
   actually *ran* it (via `bash_tool`) — loading the dataset, calling
   `detect_anomalies()`, running the full multi-turn conversation, hitting
   the live FastAPI endpoints with curl, and running the Streamlit server
   to confirm it served valid HTML with no exceptions in its logs. This
   caught two real bugs before the user ever saw the code (see
   `DEBUGGING_REPORT.md`).
5. **Adaptation to environment constraints**: the original plan used
   `sentence-transformers` (a neural embedding model) for RAG. When the
   sandbox's network policy blocked HuggingFace downloads, Claude switched
   to a local TF-IDF vectorizer instead of just working around the
   sandbox — this is documented as a permanent architectural decision in
   `DECISION_LOG.md` because it's arguably the *better* choice for this
   project's "keep it simple, fully local" requirement anyway, not just a
   sandbox workaround.
6. **Test-writing**: pytest tests were written after the implementation,
   using hand-crafted fixtures with mathematically exact expected values
   (e.g. a fixture where Region West's revenue is precisely halved) so
   assertions could check exact percentages rather than loose bounds.

## Specific ways Claude (the LLM) is used *inside* the running application

This is distinct from Claude being used to *build* the project — the app
itself also calls an LLM (Google's Gemini API, via `google-genai`) at runtime:

- **Narrative synthesis only, never number generation.** The agent
  (`app/agent/core.py`) always computes real numbers first via pandas/
  scikit-learn tool calls, then hands those numbers to the LLM in a JSON
  blob with a system prompt that explicitly forbids inventing any figure
  not present in that blob. The LLM's job is to turn correct numbers into
  readable prose, not to reason about the data itself.
- **Grounding via RAG, not memory.** When the agent needs policy or
  historical-incident context, it retrieves it from the local FAISS index
  and passes the retrieved text to the LLM — the LLM is not asked to recall
  company policy from its own training data.
- **Fallback guarantees correctness over fluency.** If the LLM API is
  unavailable (no key, timeout, network error), `_template_fallback()`
  builds the same structured answer directly from tool results with zero
  LLM involvement. This was tested explicitly (`tests/test_agent.py`) and
  is in fact the path all 44 automated tests run through, since no API key
  is present in the CI/sandbox test environment.

## Known limitations of the AI-built code (disclosed honestly)

- **Provider swap (Anthropic → Gemini):** the app originally called
  Anthropic's API; it was later switched to Google's Gemini API
  (`google-genai`) per updated requirements. Only `app/agent/llm_client.py`
  and the `GEMINI_*` vars in `app/config.py`/`.env.example` changed — the
  public interface (`LLMClient.generate()`, `LLMUnavailableError`,
  `get_llm_client()`) is identical, so `app/agent/core.py` needed zero
  edits. Like the original Anthropic integration, the live Gemini call
  could not be exercised end-to-end in the build/test sandbox (no network
  access to install `google-genai` or reach the API) — the fallback path
  was fully tested instead. Do one live pass with a real `GEMINI_API_KEY`
  before considering the LLM path production-verified.
- The Docker image was not build-tested (no Docker available in the build
  sandbox) — see `README.md` section 3 and `DEBUGGING_REPORT.md`.
- The synthetic sample dataset's anomaly signal required two rounds of
  tuning (noise level, detrending) to be cleanly detectable — this is
  disclosed in full in `DEBUGGING_REPORT.md` rather than glossed over.
