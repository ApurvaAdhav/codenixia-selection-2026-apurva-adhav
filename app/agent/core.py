"""
app/agent/core.py
------------------
The AI Agent. This is the piece that ties together:
    ML Analysis -> AI Agent (tool routing) -> RAG -> LLM -> Structured Insights

Design choice (documented in DECISION_LOG.md): instead of a heavy
multi-step "ReAct"-style agent framework, we use a deliberately simple and
FULLY DETERMINISTIC tool router:
    1. Extract entities from the question (region name? product name?
       asking about anomalies? asking about forecast?) using the actual
       values present in the dataset + keyword matching.
    2. Based on those entities, call the relevant tool(s) from
       agent/tools.py directly (Python function calls - no LLM round-trip
       needed to "decide" which tool to use).
    3. Always also query the RAG knowledge base for relevant policy/KPI/
       incident context.
    4. Hand the ACTUAL tool outputs + RAG context to the LLM and ask it to
       write the narrative - explicitly instructed to only use the given
       numbers, never invent any.
    5. If the LLM is unavailable, fall back to a deterministic template
       that formats the same tool outputs directly (no prose polish, but
       100% correct numbers).

This keeps tool selection debuggable and testable without needing a live
LLM (you can unit test _route_tools() and _template_fallback() with zero
network calls), while still satisfying "AI Agent with tools" + "RAG" +
"LLM" requirements.

Multi-turn memory: ConversationSession keeps a short history of prior
questions/answers/entities so follow-ups like "What about Region West?"
correctly reuse context (e.g. still comparing the same period).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import pandas as pd

from app.agent import tools as agent_tools
from app.agent.llm_client import get_llm_client, LLMUnavailableError
from app.logging_config import setup_logging

logger = setup_logging(__name__)


# ---------------------------------------------------------------------------
# Conversation memory
# ---------------------------------------------------------------------------
@dataclass
class Turn:
    question: str
    answer: str
    entities: dict = field(default_factory=dict)
    tool_results: dict = field(default_factory=dict)


@dataclass
class ConversationSession:
    """Holds multi-turn state for one user session. In this MVP, sessions
    are kept in-memory keyed by session_id (see api/routes.py). For a
    production system you'd back this with Redis/DB, but in-memory is
    correct and simple for a single-process demo/interview."""
    session_id: str
    turns: list[Turn] = field(default_factory=list)
    last_entities: dict = field(default_factory=dict)

    def add_turn(self, turn: Turn) -> None:
        self.turns.append(turn)
        # Merge (not replace) so unmentioned entities persist across turns,
        # e.g. asking about a product after a region was already set.
        self.last_entities = {**self.last_entities, **{k: v for k, v in turn.entities.items() if v}}

    def history_text(self, max_turns: int = 4) -> str:
        recent = self.turns[-max_turns:]
        lines = []
        for t in recent:
            lines.append(f"Q: {t.question}\nA: {t.answer}")
        return "\n\n".join(lines)


_sessions: dict[str, ConversationSession] = {}


def get_session(session_id: str) -> ConversationSession:
    if session_id not in _sessions:
        _sessions[session_id] = ConversationSession(session_id=session_id)
    return _sessions[session_id]


# ---------------------------------------------------------------------------
# Entity extraction / tool routing
# ---------------------------------------------------------------------------
ANOMALY_KEYWORDS = ["anomaly", "anomalies", "unusual", "outlier", "abnormal"]
FORECAST_KEYWORDS = ["forecast", "predict", "next month", "expected", "projection", "next week"]
PRODUCT_KEYWORDS = ["product", "which item", "sku"]
REGION_KEYWORDS = ["region"]


def _extract_entities(question: str, df: pd.DataFrame, prior_entities: dict) -> dict:
    """Look for known Region/Product values (from the actual dataset) inside
    the question text, plus a few intent keyword flags. Falls back to the
    PRIOR turn's entities when the current question doesn't mention one
    (this is what makes 'What about Region West?' -> then 'Which product
    caused it?' work as true follow-ups)."""
    q_lower = question.lower()
    entities: dict = {}

    regions = sorted(df["Region"].dropna().unique().tolist()) if "Region" in df.columns else []
    products = sorted(df["Product"].dropna().unique().tolist()) if "Product" in df.columns else []

    # Use word-boundary regex (not plain substring) so e.g. "Product C" does
    # NOT falsely match inside "...product caused it..." (a real bug we hit
    # during testing: "product c" is a substring of "product caused").
    def _mentions(name: str) -> bool:
        return re.search(rf"\b{re.escape(name.lower())}\b", q_lower) is not None

    found_region = next((r for r in regions if _mentions(r)), None)
    found_product = next((p for p in products if _mentions(p)), None)

    entities["region"] = found_region or (prior_entities.get("region") if _is_followup(question) else None)
    entities["product"] = found_product or (prior_entities.get("product") if _is_followup(question) else None)

    entities["wants_anomalies"] = any(k in q_lower for k in ANOMALY_KEYWORDS)
    entities["wants_forecast"] = any(k in q_lower for k in FORECAST_KEYWORDS)
    entities["wants_product_ranking"] = ("which product" in q_lower) or ("what product" in q_lower)

    return entities


def _is_followup(question: str) -> bool:
    """Heuristic: short questions, or questions starting with 'what about' /
    'and' / lacking a verb, are treated as follow-ups that should inherit
    context from the previous turn."""
    q = question.strip().lower()
    followup_starts = ["what about", "and ", "what if", "how about", "why"]
    return len(q.split()) <= 6 or any(q.startswith(s) for s in followup_starts)


def _route_tools(question: str, entities: dict, df: pd.DataFrame) -> dict:
    """Decide which tools to call and call them. Returns {tool_name: result}.
    Always includes analyze_sales as the baseline context, plus RAG search."""
    results: dict = {}

    # Baseline: always understand overall sales movement.
    results["analyze_sales"] = agent_tools.analyze_sales(df)

    if entities.get("region"):
        results["analyze_region"] = agent_tools.analyze_region(df, region=entities["region"])

    if entities.get("product") or entities.get("wants_product_ranking"):
        results["analyze_products"] = agent_tools.analyze_products(df, product=entities.get("product"))

    # Run anomaly detection whenever explicitly asked, OR whenever the
    # baseline shows a significant drop (>=10%) - this is what lets the
    # agent explain drops even if the user didn't say the word "anomaly".
    significant_drop = results["analyze_sales"].get("revenue_change_pct", 0) <= -10
    if entities.get("wants_anomalies") or significant_drop:
        results["detect_anomalies"] = agent_tools.detect_anomalies(df, top_n=5)

    if entities.get("wants_forecast"):
        results["predict_next_month"] = agent_tools.predict_next_month(
            df, region=entities.get("region"), product=entities.get("product")
        )

    # RAG: always search using the raw question, it's a cheap local call
    # and gives the LLM policy/incident grounding regardless of intent.
    results["search_business_knowledge"] = agent_tools.search_business_knowledge(question)

    logger.info("Routed tools for question=%r -> %s", question, list(results.keys()))
    return results


# ---------------------------------------------------------------------------
# LLM prompt construction
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are an AI Business Intelligence assistant for a sales analytics platform.

STRICT RULES:
1. You MUST only use numbers, percentages, and facts that appear in the
   "TOOL RESULTS" JSON provided to you. NEVER invent, estimate, or guess any
   number that is not explicitly present in the tool results.
2. If the tool results don't contain something the user asked about, say so
   plainly instead of making it up.
3. For the FIRST question in a conversation (no prior turns), structure your
   answer using this format:
   Sales dropped/grew: <X%>
   Main contributors: <region/product breakdown>
   Key insight: <one-sentence synthesis of the single most important takeaway
   from the numbers above - not a repeat of the contributor list>
   Detected anomaly: <anomaly finding or "None detected">
   Recommendation: <one concrete, actionable recommendation>
4. For FOLLOW-UP questions, answer conversationally but stay just as
   numerically precise and grounded in the tool results.
5. When relevant, ground your recommendation in the BUSINESS KNOWLEDGE
   (policies/KPI definitions/incident reports) provided, and mention it
   naturally (e.g. "per the Anomaly Response Policy...").
6. Be concise. Business managers are busy - prefer short, scannable answers.
"""


def _build_user_prompt(question: str, tool_results: dict, history_text: str, is_first_turn: bool) -> str:
    import json
    parts = []
    if history_text:
        parts.append(f"CONVERSATION HISTORY:\n{history_text}\n")
    parts.append(f"CURRENT QUESTION: {question}\n")
    parts.append(f"IS_FIRST_QUESTION: {is_first_turn}\n")
    parts.append(f"TOOL RESULTS (ground truth - use ONLY these numbers):\n{json.dumps(tool_results, indent=2, default=str)}\n")
    parts.append("Write your answer now, following the STRICT RULES from the system prompt.")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Deterministic fallback (used if the LLM is unavailable)
# ---------------------------------------------------------------------------
def _template_fallback(question: str, tool_results: dict, is_first_turn: bool) -> str:
    """Builds a correct, structured answer directly from tool_results with
    no LLM involved. Guarantees the app still works if the API key is
    missing or the LLM call fails/times out."""
    lines = []

    sales = tool_results.get("analyze_sales")
    if sales:
        change = sales.get("revenue_change_pct", 0)
        direction = "dropped" if change < 0 else "grew"
        lines.append(f"Sales {direction}: {abs(change)}%")

        contributors = []
        for r in sales.get("top_region_contributors", [])[:2]:
            contributors.append(f"Region {r['region']} {r['change_pct']:+.0f}%")
        for p in sales.get("top_product_contributors", [])[:2]:
            contributors.append(f"{p['product']} {p['change_pct']:+.0f}%")
        if contributors:
            lines.append("Main contributors: " + ", ".join(contributors))

        # Key insight: the single biggest dollar-impact driver across BOTH
        # dimensions (region and product) - distinct from "Main contributors"
        # above, which just lists the top entries per-dimension. Grounded
        # entirely in fields already present in the tool result.
        ranked = (
            [{"dim": "Region", "name": r["region"], **r} for r in sales.get("top_region_contributors", [])]
            + [{"dim": "Product", "name": p["product"], **p} for p in sales.get("top_product_contributors", [])]
        )
        if ranked:
            biggest = max(ranked, key=lambda r: abs(r["absolute_change"]))
            lines.append(
                f"Key insight: {biggest['dim']} {biggest['name']} is the single largest driver "
                f"of the change ({biggest['change_pct']:+.1f}%, {biggest['absolute_change']:+.0f} revenue impact)."
            )

    region_res = tool_results.get("analyze_region")
    if region_res and "error" not in region_res:
        lines.append(
            f"Region {region_res['region']}: revenue {region_res['revenue_change_pct']:+.1f}% "
            f"({region_res['previous_revenue']} -> {region_res['current_revenue']})"
        )
        top_products = region_res.get("top_product_contributors", [])
        if top_products:
            worst = top_products[0]
            lines.append(
                f"Top product driver in {region_res['region']}: {worst['product']} "
                f"({worst['change_pct']:+.1f}%, {worst['absolute_change']:+.0f} revenue impact)"
            )

    product_res = tool_results.get("analyze_products")
    if product_res and "error" not in product_res:
        if "revenue_change_pct" in product_res:
            lines.append(
                f"Product {product_res['product']}: revenue {product_res['revenue_change_pct']:+.1f}%"
            )
        elif "all_products_ranked_by_change" in product_res:
            worst = product_res["all_products_ranked_by_change"][:3]
            ranked = ", ".join(f"{p['product']} {p['change_pct']:+.0f}%" for p in worst)
            lines.append(f"Products ranked by change: {ranked}")

    anomalies = tool_results.get("detect_anomalies")
    if anomalies:
        a = anomalies[0]
        lines.append(f"Detected anomaly: {a['product']} in {a['region']} {a['direction']} on {a['date']}")
    elif "detect_anomalies" in tool_results:
        lines.append("Detected anomaly: None found")

    forecast = tool_results.get("predict_next_month")
    if forecast and "error" not in forecast:
        lines.append(
            f"Forecast (next {forecast['forecast_horizon_days']} days) for "
            f"{forecast['product']}/{forecast['region']}: {forecast['trend']} "
            f"({forecast['trend_pct']:+.1f}%), predicted revenue {forecast['predicted_total_revenue']}"
        )

    kb = tool_results.get("search_business_knowledge")
    if kb:
        lines.append(f"Related policy/context: {kb[0]['heading']} ({kb[0]['source']})")

    if is_first_turn:
        lines.append("Recommendation: Review the top contributor above and check recent incident reports for a known cause.")

    lines.append("\n[Note: generated via deterministic fallback - LLM was unavailable.]")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def ask(question: str, df: pd.DataFrame, session_id: str = "default") -> dict:
    """
    Main agent entry point. Returns:
      {
        "answer": str,
        "tool_calls": list[str],
        "tool_results": dict,
        "entities": dict,
        "used_llm": bool,
      }
    """
    session = get_session(session_id)
    is_first_turn = len(session.turns) == 0

    entities = _extract_entities(question, df, session.last_entities)
    tool_results = _route_tools(question, entities, df)

    history_text = session.history_text()
    used_llm = False
    llm = get_llm_client()

    try:
        system_prompt = SYSTEM_PROMPT
        user_prompt = _build_user_prompt(question, tool_results, history_text, is_first_turn)
        answer = llm.generate(system_prompt, user_prompt)
        used_llm = True
    except LLMUnavailableError as exc:
        logger.warning("LLM unavailable (%s); using template fallback.", exc)
        answer = _template_fallback(question, tool_results, is_first_turn)

    turn = Turn(question=question, answer=answer, entities=entities, tool_results=tool_results)
    session.add_turn(turn)

    return {
        "answer": answer,
        "tool_calls": list(tool_results.keys()),
        "tool_results": tool_results,
        "entities": entities,
        "used_llm": used_llm,
    }


def reset_session(session_id: str = "default") -> None:
    _sessions.pop(session_id, None)
