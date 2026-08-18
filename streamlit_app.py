"""
streamlit_app.py
------------------
The dashboard required by the challenge: "Simple Streamlit dashboard with
KPIs, charts, and AI analysis."

Runs directly in-process (imports app.* modules directly) rather than
calling the FastAPI server over HTTP - simpler for a single-user demo and
one less moving part to explain in an interview. The FastAPI server remains
available separately for programmatic/API access.

NO SAMPLE/DEMO DATA: the dashboard starts empty and requires a real
CSV/Excel upload via the sidebar before any analysis renders. See
app/data/store.py.

Run:
    streamlit run streamlit_app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.agent import core as agent_core
from app.agent.llm_client import get_llm_client
from app.data import store
from app.data.analytics import analyze_sales, get_kpi_summary
from app.data.anomaly import detect_anomalies
from app.data.forecast import predict_next_period
from app.data.processing import REQUIRED_COLUMNS, DataValidationError, load_and_clean
from app.logging_config import setup_logging

logger = setup_logging(__name__)

st.set_page_config(
    page_title="AI Business Intelligence Assistant",
    page_icon="📊",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Enterprise UI polish
# ---------------------------------------------------------------------------
# NOTE: this only touches typography/spacing/card framing in neutral grays.
# It does NOT redefine primaryColor/backgroundColor/accent or any chart
# color - those stay exactly as they were (see .streamlit/config.toml,
# which pins Streamlit's own existing default theme unchanged).
def _inject_custom_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        }

        div.block-container { padding-top: 2rem; padding-bottom: 3rem; }

        h1, h2, h3 { font-weight: 700; letter-spacing: -0.01em; }

        /* KPI metric cards */
        div[data-testid="stMetric"] {
            background: #FFFFFF;
            border: 1px solid #E4E7EC;
            border-radius: 10px;
            padding: 1rem 1.1rem;
            box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04);
        }
        div[data-testid="stMetricLabel"] { font-weight: 600; color: #667085; }
        div[data-testid="stMetricValue"] { font-weight: 700; }

        /* Tabs */
        button[data-baseweb="tab"] { font-weight: 600; }

        /* Sidebar */
        section[data-testid="stSidebar"] { border-right: 1px solid #E4E7EC; }

        /* Chat bubbles */
        div[data-testid="stChatMessage"] { border-radius: 12px; }

        /* Expander (Technical Details) */
        details { border-radius: 8px; }
        </style>
        """,
        unsafe_allow_html=True,
    )


_inject_custom_css()

# ---------------------------------------------------------------------------
# Session state init (no dataset auto-load - starts empty by design)
# ---------------------------------------------------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of {"role": "user"/"assistant", "content": str}
if "session_id" not in st.session_state:
    st.session_state.session_id = "streamlit-session"


# ---------------------------------------------------------------------------
# Sidebar: data upload + status
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("📊 BI Assistant")
    st.caption("AI Business Intelligence & Decision Support")

    st.subheader("Dataset")
    uploaded_file = st.file_uploader("Upload sales CSV/Excel", type=["csv", "xlsx", "xls"])

    if uploaded_file is not None:
        try:
            file_bytes = uploaded_file.read()
            df, report = load_and_clean(file_bytes, uploaded_file.name)
            store.set_active_dataset(df, report, uploaded_file.name)
            agent_core.reset_session(st.session_state.session_id)
            st.session_state.chat_history = []
            st.success(f"Loaded '{uploaded_file.name}': {report.rows_out} usable rows.")
            if report.notes:
                for n in report.notes:
                    st.caption(f"⚠️ {n}")
        except DataValidationError as exc:
            st.error(f"Upload rejected: {exc}")
        except Exception as exc:
            logger.exception("Streamlit upload failed")
            st.error(f"Unexpected error: {exc}")

    if not store.has_dataset():
        st.caption("Required columns: " + ", ".join(REQUIRED_COLUMNS))

    st.divider()
    llm = get_llm_client()
    if llm.enabled:
        st.success("✅ LLM connected (Gemini)")
    else:
        st.warning(
            "⚠️ LLM not configured — using deterministic template fallback.\n"
            "Set GEMINI_API_KEY in .env to enable full narrative answers."
        )

    if store.has_dataset():
        df_preview = store.get_active_dataset()
        st.caption(f"Active dataset: **{store.get_active_source_name()}**")
        st.caption(f"{len(df_preview):,} rows | {df_preview['Date'].min().date()} → {df_preview['Date'].max().date()}")

        st.divider()
        period_days = st.slider("Comparison period (days)", min_value=7, max_value=90, value=30, step=1)

        if st.button("🗑️ Clear chat"):
            st.session_state.chat_history = []
            agent_core.reset_session(st.session_state.session_id)
            st.rerun()
    else:
        period_days = 30  # unused until a dataset is loaded


# ---------------------------------------------------------------------------
# Header (always shown, even before a dataset is loaded)
# ---------------------------------------------------------------------------
st.title("AI Business Intelligence & Decision Support Assistant")
st.caption("CSV/Excel → Pandas cleaning → ML analysis (Isolation Forest + forecast) → AI Agent → RAG → LLM → Structured insights")

# ---------------------------------------------------------------------------
# Empty state: no dataset yet - required upload before any analysis
# ---------------------------------------------------------------------------
if not store.has_dataset():
    st.write("")
    with st.container(border=True):
        st.markdown("### 📤 Upload a dataset to get started")
        st.write(
            "This dashboard requires your own sales data — no sample or demo "
            "data is bundled. Use the **Upload sales CSV/Excel** control in "
            "the sidebar to load a file."
        )
        st.caption("Required columns: " + ", ".join(REQUIRED_COLUMNS))
    st.stop()

df = store.get_active_dataset()

tab_dashboard, tab_chat = st.tabs(["📈 Dashboard", "💬 Ask the Assistant"])

# ---------------------------------------------------------------------------
# Dashboard tab
# ---------------------------------------------------------------------------
with tab_dashboard:
    try:
        kpi = get_kpi_summary(df)
        sales = analyze_sales(df, period_days=period_days)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Revenue (all-time)", f"${kpi['total_revenue']:,.0f}")
        col2.metric("Total Profit (all-time)", f"${kpi['total_profit']:,.0f}", f"{kpi['profit_margin_pct']}% margin")
        col3.metric(
            f"Revenue (last {period_days}d)",
            f"${sales['current_period']['revenue']:,.0f}",
            f"{sales['revenue_change_pct']:+.1f}% vs prior {period_days}d",
        )
        col4.metric(
            f"Profit (last {period_days}d)",
            f"${sales['current_period']['profit']:,.0f}",
            f"{sales['profit_change_pct']:+.1f}% vs prior {period_days}d",
        )

        st.divider()

        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            st.subheader("Daily Revenue Trend")
            daily = df.groupby("Date", as_index=False)["Revenue"].sum()
            fig = px.line(daily, x="Date", y="Revenue")
            fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=320)
            st.plotly_chart(fig, use_container_width=True)

        with chart_col2:
            st.subheader("Revenue by Region")
            by_region = df.groupby("Region", as_index=False)["Revenue"].sum().sort_values("Revenue", ascending=False)
            fig = px.bar(by_region, x="Region", y="Revenue", color="Region")
            fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=320, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        chart_col3, chart_col4 = st.columns(2)

        with chart_col3:
            st.subheader("Revenue by Product")
            by_product = df.groupby("Product", as_index=False)["Revenue"].sum().sort_values("Revenue", ascending=False)
            fig = px.bar(by_product, x="Product", y="Revenue", color="Product")
            fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=320, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        with chart_col4:
            st.subheader(f"Top Movers (last {period_days}d vs prior)")
            movers = sales["top_region_contributors"] + sales["top_product_contributors"]
            movers_df = pd.DataFrame(movers)
            if not movers_df.empty:
                movers_df["label"] = movers_df.get("region", movers_df.get("product"))
                fig = px.bar(
                    movers_df, x="absolute_change", y="label", orientation="h",
                    color="absolute_change", color_continuous_scale=["red", "lightgrey", "green"],
                )
                fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=320, coloraxis_showscale=False)
                st.plotly_chart(fig, use_container_width=True)

        st.divider()
        st.subheader("🔍 Detected Anomalies (Isolation Forest)")
        anomalies = detect_anomalies(df)[:10]
        if anomalies:
            st.dataframe(pd.DataFrame(anomalies), use_container_width=True, hide_index=True)
        else:
            st.info("No significant anomalies detected in the current dataset.")

        st.subheader("📅 Next 30-Day Forecast")
        forecast = predict_next_period(df, periods=30)
        if "error" not in forecast:
            fcol1, fcol2, fcol3 = st.columns(3)
            fcol1.metric("Predicted total revenue (next 30d)", f"${forecast['predicted_total_revenue']:,.0f}")
            fcol2.metric("Trend", forecast["trend"].capitalize(), f"{forecast['trend_pct']:+.1f}%")
            fcol3.metric("Model confidence", forecast["confidence_note"], f"R²={forecast['r_squared']}")
        else:
            st.info(forecast["error"])

    except Exception as exc:
        logger.exception("Dashboard rendering failed")
        st.error(f"Failed to render dashboard: {exc}")


# ---------------------------------------------------------------------------
# Chat tab
# ---------------------------------------------------------------------------
with tab_chat:
    st.subheader("Ask questions about your sales data")
    st.caption(
        "Try: \"Why did sales drop?\" → \"What about Region West?\" → "
        "\"Which product caused it?\" → \"What is expected next month?\""
    )

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    sample_qs = [
        "Why did sales drop?",
        "What about Region West?",
        "Which product caused it?",
        "What is expected next month?",
    ]
    cols = st.columns(len(sample_qs))
    clicked_question = None
    for c, q in zip(cols, sample_qs):
        if c.button(q, use_container_width=True):
            clicked_question = q

    user_question = st.chat_input("Ask about sales, regions, products, anomalies, or forecasts...")
    question = clicked_question or user_question

    if question:
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                try:
                    result = agent_core.ask(question=question, df=df, session_id=st.session_state.session_id)
                    answer = result["answer"]
                    st.markdown(answer)
                    with st.expander("Technical Details"):
                        st.caption(
                            "LLM used: " + ("Yes (Gemini)" if result["used_llm"] else "No — deterministic fallback")
                        )
                        st.caption("Tools called: " + ", ".join(result["tool_calls"]))
                        entities_str = ", ".join(
                            f"{k}={v}" for k, v in result["entities"].items() if v not in (None, False)
                        )
                        if entities_str:
                            st.caption("Extracted entities: " + entities_str)
                        st.markdown("**Raw tool results (RAG + analytics JSON)**")
                        st.json(result["tool_results"])
                except Exception as exc:
                    logger.exception("Chat question failed: %r", question)
                    answer = f"Sorry, something went wrong answering that: {exc}"
                    st.error(answer)

        st.session_state.chat_history.append({"role": "assistant", "content": answer})
