import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from engine import analyze


st.set_page_config(
    page_title="Swing Analyzer V5.1",
    page_icon="🤖",
    layout="wide",
)


st.title("🤖 Swing Analyzer V5.1")

st.caption(
    "Technical decision engine with AI-style market interpretation "
    "for NSE swing opportunities."
)


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
    "🤖 AI Analyze",
    use_container_width=True,
)


if not analyze_button:
    st.info(
        "Enter an NSE symbol and click 🤖 AI Analyze."
    )

    st.stop()


try:
    result = analyze(symbol)

except Exception as error:
    st.error(
        f"Could not analyze {symbol}: {error}"
    )

    st.stop()


row = result["row"]

price = result["price"]

score = result["score"]

reversal_score = result["reversal_score"]

stage_allocation = result["stage_allocation"]

ai = result["ai_decision"]


st.subheader(
    f"{result['ticker']} — {result['signal']}"
)


# --------------------------------------------------
# AI DECISION
# --------------------------------------------------

st.header("🤖 AI DECISION")


if (
    "DO NOT BUY" in ai["verdict"]
    or "🛑" in ai["verdict"]
):
    st.error(ai["verdict"])

elif "SMALL BUY" in ai["verdict"]:
    st.warning(ai["verdict"])

else:
    st.success(ai["verdict"])


a1, a2, a3, a4 = st.columns(4)

a1.metric(
    "AI Confidence",
    ai["confidence"],
)

a2.metric(
    "Setup Score",
    f"{score}/100",
)

a3.metric(
    "Reversal Score",
    f"{reversal_score}/100",
)

a4.metric(
    "Trend",
    result["trend"],
)


st.markdown("### AI View")

st.write(ai["summary"])


for observation in ai["observations"]:
    st.write(
        f"• {observation}"
    )


st.markdown("### 🎯 AI Next Step")

st.info(
    ai["next_step"]
)


st.divider()


# --------------------------------------------------
# ENTRY STAGE
# --------------------------------------------------

st.header("🚦 Entry Stage")

s1, s2, s3 = st.columns(3)

s1.metric(
    "Current Stage",
    result["stage"],
)

s2.metric(
    "Recommended Action",
    result["stage_action"],
)

s3.metric(
    "Planned Allocation",
    f"{stage_allocation}%",
)


if stage_allocation == 0:
    st.warning(
        "The engine does not currently approve a new position."
    )

elif stage_allocation == 25:
    st.warning(
        "Only a first-stage 25% entry is approved. "
        "Do not deploy full capital."
    )

elif stage_allocation == 75:
    st.success(
        "The bullish setup is confirmed. "
        "Up to 75% staged allocation is approved by the rules."
    )

else:
    st.success(
        "Breakout stage confirmed by the technical rules."
    )


st.divider()


# --------------------------------------------------
# MARKET SNAPSHOT
# --------------------------------------------------

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


# --------------------------------------------------
# POSITION SIZING
# --------------------------------------------------

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
                if stage_allocation >= 25
                else "⏳ Wait"
            ),
            (
                "✅ Approved"
                if stage_allocation >= 75
                else "⏳ Wait"
            ),
            (
                "✅ Approved"
                if stage_allocation >= 100
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


st.divider()


# --------------------------------------------------
# ACTION PLAN
# --------------------------------------------------

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


if stage_allocation == 0:
    st.warning(
        "These levels are research references only. "
        "There is currently NO approved entry."
    )


st.divider()


# --------------------------------------------------
# WHY
# --------------------------------------------------

st.header("🧠 Why the Engine Gave This Decision")


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


# --------------------------------------------------
# BUY REQUIREMENTS
# --------------------------------------------------

st.header("🚦 What Must Happen Before BUY?")


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


# --------------------------------------------------
# CHART
# --------------------------------------------------

st.header("📊 Interactive Price Trend")


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


st.warning(
    "V5.1 is a rule-based technical research tool. "
    "The AI decision shown here is an automated interpretation "
    "of technical indicators, not a guarantee of profit or "
    "personalized investment advice. Verify live prices with "
    "your broker."
)
