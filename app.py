import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from engine import analyze
from ai_engine import ai_analyze


# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Swing Analyzer V5.2",
    page_icon="🤖",
    layout="wide",
)


# ==================================================
# APP HEADER
# ==================================================

st.title("🤖 Swing Analyzer V5.2")

st.caption(
    "Quantitative technical decision engine for NSE swing opportunities."
)


# ==================================================
# SIDEBAR
# ==================================================

st.sidebar.header("🔍 Analyze")

symbol = st.sidebar.text_input(
    "NSE Symbol",
    "SILVERBEES",
)

capital = st.sidebar.number_input(
    "Available Capital (₹)",
    min_value=1000,
    max_value=10000000,
    value=200000,
    step=5000,
)

risk_pct = st.sidebar.slider(
    "Maximum Risk Per Trade (%)",
    min_value=0.25,
    max_value=2.00,
    value=0.50,
    step=0.25,
)

analyze_button = st.sidebar.button(
    "🤖 Smart Analyze",
    use_container_width=True,
)


# ==================================================
# WAIT FOR ANALYZE BUTTON
# ==================================================

if not analyze_button:
    st.info(
        "Enter an NSE symbol and click 🤖 Smart Analyze."
    )

    st.stop()


# ==================================================
# RUN TECHNICAL ENGINE
# ==================================================

try:
    result = analyze(symbol)

    ai_result = ai_analyze(result)

except Exception as error:
    st.error(
        f"Could not analyze {symbol}: {error}"
    )

    st.stop()


# ==================================================
# EXTRACT DATA
# ==================================================

row = result["row"]

price = result["price"]

score = result["score"]

stage_allocation = result.get(
    "stage_allocation",
    0,
)


# ==================================================
# STOCK HEADER
# ==================================================

st.subheader(
    f"{result['ticker']} — {result['signal']}"
)


# ==================================================
# QUANTITATIVE DECISION
# ==================================================

st.header("🤖 Quantitative Decision")


recommendation = ai_result["recommendation"]

decision = recommendation["decision"]

action = recommendation["action"]


if action == "BUY":
    st.success(
        f"### {decision}"
    )

elif action == "WAIT":
    st.warning(
        f"### {decision}"
    )

else:
    st.error(
        f"### {decision} — DO NOT BUY NOW"
    )


# ==================================================
# DECISION METRICS
# ==================================================

a1, a2, a3, a4 = st.columns(4)


a1.metric(
    "Recommendation Confidence",
    f"{ai_result['confidence']:.1f}%",
)


a2.metric(
    "Favourable Setup Probability",
    f"{ai_result['probability']:.1f}%",
)


a3.metric(
    "Risk Level",
    ai_result["risk_level"],
)


a4.metric(
    "Trade Quality",
    ai_result["stars"],
)


# ==================================================
# MODEL INTERPRETATION
# ==================================================

st.subheader("🧠 Model Interpretation")


for paragraph in ai_result["ai_view"]:
    st.write(
        paragraph
    )


# ==================================================
# REWARD VS RISK
# ==================================================

st.subheader("⚖️ Reward vs Risk")


rr = ai_result["reward_risk"]


r1, r2, r3 = st.columns(3)


r1.metric(
    "ATR Risk",
    f"{rr['risk_percent']:.2f}%",
)


r2.metric(
    "2R Reward",
    f"{rr['reward_2r_percent']:.2f}%",
)


r3.metric(
    "Reward / Risk",
    f"{rr['rr_2r']:.2f} : 1",
)


# ==================================================
# FINAL ACTION
# ==================================================

st.subheader("🎯 Final Action")


if action == "BUY":

    st.success(
        f"BUY APPROVED — Planned allocation: "
        f"{recommendation['allocation']}%"
    )


elif action == "WAIT":

    st.warning(
        "WAIT — The model does not approve "
        "a new position yet."
    )


else:

    st.error(
        "DO NOT BUY — Current conditions "
        "fail the entry model."
    )


st.divider()


# ==================================================
# MARKET SNAPSHOT
# ==================================================

st.header("📊 Market Snapshot")


m1, m2, m3, m4 = st.columns(4)


m1.metric(
    "Current Price",
    f"₹{price:.2f}",
)


m2.metric(
    "EMA20",
    f"₹{row['EMA20']:.2f}",
)


m3.metric(
    "EMA50",
    f"₹{row['EMA50']:.2f}",
)


m4.metric(
    "RSI",
    f"{row['RSI']:.1f}",
)


m5, m6, m7, m8 = st.columns(4)


m5.metric(
    "ADX",
    f"{row['ADX']:.1f}",
)


m6.metric(
    "EMA200",
    f"₹{row['EMA200']:.2f}",
)


m7.metric(
    "20-Day Resistance",
    f"₹{row['RES20']:.2f}",
)


m8.metric(
    "20-Day Support",
    f"₹{row['SUP20']:.2f}",
)


st.divider()


# ==================================================
# POSITION SIZING
# ==================================================

risk_budget = (
    capital
    * risk_pct
    / 100
)


risk_per_share = max(
    price - result["stop"],
    0.01,
)


max_qty_by_risk = int(
    risk_budget
    // risk_per_share
)


max_qty_by_capital = int(
    capital
    // price
)


max_quantity = min(
    max_qty_by_risk,
    max_qty_by_capital,
)


stage_1_qty = int(
    max_quantity * 0.25
)


stage_2_qty = int(
    max_quantity * 0.50
)


stage_3_qty = (
    max_quantity
    - stage_1_qty
    - stage_2_qty
)


# ==================================================
# POSITION PLAN
# ==================================================

st.header("💰 Position Plan")


p1, p2, p3 = st.columns(3)


p1.metric(
    "Maximum Risk Budget",
    f"₹{risk_budget:.2f}",
)


p2.metric(
    "Maximum Quantity",
    max_quantity,
)


p3.metric(
    "Full Position Value",
    f"₹{max_quantity * price:.2f}",
)


# ==================================================
# POSITION STATUS
# ==================================================

approved_allocation = recommendation["allocation"]


position_table = pd.DataFrame(
    {
        "Stage": [
            "Stage 1 — Reversal",
            "Stage 2 — Buy Confirmed",
            "Stage 3 — Breakout",
        ],

        "Allocation": [
            "25%",
            "50%",
            "25%",
        ],

        "Quantity": [
            stage_1_qty,
            stage_2_qty,
            stage_3_qty,
        ],

        "Status": [
            (
                "✅ Approved"
                if approved_allocation >= 25
                else "⏳ Wait"
            ),

            (
                "✅ Approved"
                if approved_allocation >= 75
                else "⏳ Wait"
            ),

            (
                "✅ Approved"
                if approved_allocation >= 100
                else "⏳ Wait"
            ),
        ],
    }
)


st.dataframe(
    position_table,
    use_container_width=True,
    hide_index=True,
)


if approved_allocation == 0:

    st.warning(
        "No new position is currently approved."
    )


elif approved_allocation == 25:

    st.warning(
        "Only a first-stage 25% position is approved. "
        "Do not deploy the full planned position."
    )


elif approved_allocation == 75:

    st.success(
        "The setup has sufficient confirmation "
        "for up to 75% staged allocation."
    )


elif approved_allocation >= 100:

    st.success(
        "The full technical setup is confirmed "
        "under the current model."
    )


st.divider()


# ==================================================
# TRADE REFERENCE
# ==================================================

st.header("🧭 Trade Reference")


t1, t2, t3, t4 = st.columns(4)


t1.metric(
    "Current Reference",
    f"₹{price:.2f}",
)


t2.metric(
    "ATR Stop",
    f"₹{result['stop']:.2f}",
)


t3.metric(
    "2R Target",
    f"₹{result['target_2r']:.2f}",
)


t4.metric(
    "3R Target",
    f"₹{result['target_3r']:.2f}",
)


if action != "BUY":

    st.warning(
        "These levels are research references only. "
        "There is currently NO approved entry."
    )


st.divider()


# ==================================================
# ENGINE REASONS
# ==================================================

st.header(
    "🧠 Why the Engine Gave This Decision"
)


for reason in result["reasons"]:

    icon = (
        "✅"
        if reason["passed"]
        else "❌"
    )

    st.write(
        f"{icon} "
        f"{reason['text']} "
        f"({reason['points']} points)"
    )


st.divider()


# ==================================================
# BUY REQUIREMENTS
# ==================================================

st.header(
    "🚦 What Must Happen Before BUY?"
)


if result["requirements"]:

    for requirement in result["requirements"]:

        st.write(
            f"• {requirement}"
        )


else:

    st.success(
        "The main technical BUY requirements "
        "are currently satisfied."
    )


st.divider()


# ==================================================
# INTERACTIVE PRICE CHART
# ==================================================

st.header(
    "📊 Interactive Price Trend"
)


df = result["data"].tail(300)


figure = go.Figure()


figure.add_trace(
    go.Candlestick(
        x=df.index,
        open=df["Open"],
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        name="Price",
    )
)


figure.add_trace(
    go.Scatter(
        x=df.index,
        y=df["EMA20"],
        name="EMA20",
    )
)


figure.add_trace(
    go.Scatter(
        x=df.index,
        y=df["EMA50"],
        name="EMA50",
    )
)


figure.add_trace(
    go.Scatter(
        x=df.index,
        y=df["EMA200"],
        name="EMA200",
    )
)


figure.update_layout(
    height=650,
    xaxis_rangeslider_visible=False,
)


st.plotly_chart(
    figure,
    use_container_width=True,
)


# ==================================================
# DISCLAIMER
# ==================================================

st.warning(
    "V5.2 is a quantitative rule-based technical research tool. "
    "The recommendation is generated from historical market data "
    "and technical indicators. A BUY signal does not guarantee "
    "profit. Verify live market prices with your broker and follow "
    "your defined risk limit."
)
