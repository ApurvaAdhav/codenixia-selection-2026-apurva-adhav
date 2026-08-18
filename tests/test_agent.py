"""
tests/test_agent.py
---------------------
Tests app/agent/core.py. Since no GEMINI_API_KEY is set in the test
environment, these tests exercise the LLM-unavailable path end-to-end,
which is also exactly the deterministic "never invents numbers" path
required by the challenge - a good thing to test thoroughly since it
requires zero network access and is fully deterministic.
"""
import pandas as pd
import pytest

from app.agent import core as agent_core


@pytest.fixture(autouse=True)
def _reset_sessions():
    """Ensure each test starts with clean conversation memory."""
    agent_core._sessions.clear()
    yield
    agent_core._sessions.clear()


def test_first_question_uses_structured_format(sample_df):
    result = agent_core.ask("Why did sales drop?", sample_df, session_id="t1")
    answer = result["answer"]
    assert "Sales dropped" in answer or "Sales grew" in answer
    assert "analyze_sales" in result["tool_calls"]
    assert result["used_llm"] is False  # no API key in test env


def test_followup_question_inherits_region_context(sample_df):
    agent_core.ask("Why did sales drop?", sample_df, session_id="t2")
    result = agent_core.ask("What about Region West?", sample_df, session_id="t2")
    assert result["entities"]["region"] == "West"
    assert "analyze_region" in result["tool_calls"]
    assert "West" in result["answer"]


def test_followup_which_product_does_not_false_match(sample_df):
    """Regression test for the 'product caused' substring bug: asking
    'Which product caused it?' must NOT be misread as mentioning a product
    literally named in the question text."""
    agent_core.ask("Why did sales drop?", sample_df, session_id="t3")
    agent_core.ask("What about Region West?", sample_df, session_id="t3")
    result = agent_core.ask("Which product caused it?", sample_df, session_id="t3")
    # No product is named in the question itself.
    assert result["entities"]["product"] is None
    assert result["entities"]["wants_product_ranking"] is True


def test_forecast_question_triggers_predict_tool(sample_df):
    result = agent_core.ask("What is expected next month?", sample_df, session_id="t4")
    assert "predict_next_month" in result["tool_calls"]


def test_significant_drop_auto_triggers_anomaly_detection():
    """Even without the word 'anomaly', a >=10% drop should trigger
    detect_anomalies automatically."""
    dates_p1 = pd.date_range("2024-01-01", periods=30, freq="D")
    dates_p2 = pd.date_range("2024-01-31", periods=30, freq="D")
    rows = []
    for d in dates_p1:
        rows.append(dict(Date=d, Region="North", Product="X", Category="C",
                          Customer_Segment="Consumer", Quantity=10, Revenue=1000.0, Cost=500.0, Profit=500.0))
    for d in dates_p2:
        rows.append(dict(Date=d, Region="North", Product="X", Category="C",
                          Customer_Segment="Consumer", Quantity=5, Revenue=500.0, Cost=250.0, Profit=250.0))
    df = pd.DataFrame(rows)
    df["Date"] = pd.to_datetime(df["Date"])

    result = agent_core.ask("How are we doing?", df, session_id="t5")
    assert "detect_anomalies" in result["tool_calls"]


def test_tool_results_are_grounded_not_invented(sample_df):
    """The answer's headline percentage must exactly match the underlying
    analyze_sales tool result - i.e. it comes from real computation, not
    an LLM guess (verifiable here because we're on the fallback path)."""
    result = agent_core.ask("Why did sales drop?", sample_df, session_id="t6")
    pct = result["tool_results"]["analyze_sales"]["revenue_change_pct"]
    assert f"{abs(pct)}%" in result["answer"]


def test_session_reset_clears_context(sample_df):
    agent_core.ask("What about Region West?", sample_df, session_id="t7")
    agent_core.reset_session("t7")
    session = agent_core.get_session("t7")
    assert session.turns == []
    assert session.last_entities == {}
