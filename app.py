import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from engine import analyze


st.set_page_config(
    page_title="Swing Analyzer V5",
    page_icon="📈",
    layout="wide",
)


# ---------------------------------------------------------
# PAGE HEADER
# ---------------------------------------------------------

st.title("📈 Swing Analyzer V5")

st.caption(
    "Professional rule-based NSE swing opportunity analyzer"
)


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

st.sidebar.header("🔍 Analyze")

symbol = st.sidebar.text_input(
    "NSE Symbol",
    value="SILVERBEES",
).strip().upper()

capital = st.sidebar.number_input(
    "Available Capital (₹)",
    min_value=1000,
    max_value=10000000,
    value=200000,
    step=5000,
)

risk_percent = st.sidebar.slider(
    "Maximum Risk Per Trade (%)",
    min_value=0.25,
    max_value=2.00,
    value=0.50,
    step=0.25,
)

analyze_button = st.sidebar.button(
    "🚀 Analyze Opportunity",
    use_container_width=True,
)


# ---------------------------------------------------------
# ANALYSIS
# ---------------------------------------------------------

if not symbol:
    st.warning("Enter an NSE symbol.")
    st.stop()


try:
    result = analyze(symbol)

except Exception as error:
    st.error(
        f"Could not analyze {symbol}: {error}"
    )
    st.stop()


ticker = result["ticker"]
df = result["data"]
row = result["row"]
price = result["price"]
score = result["score"]
signal = result["signal"]
trend = result["trend"]
stars = result["stars"]
quality = result["quality"]
reasons = result["reasons"]
requirements = result["requirements"]
stop = result["stop"]
target_2r = result["target_2r"]
target_3r = result["target_3r"]


# ---------------------------------------------------------
# MAIN SIGNAL
# ---------------------------------------------------------

st.subheader(
    f"{ticker} — {signal}"
)

st.caption(
    f"Opportunity rating: {stars} • {quality}"
)


# ---------------------------------------------------------
# TOP METRICS
# ---------------------------------------------------------

metric_1, metric_2, metric_3, metric_4 = st.columns(4)

metric_1.metric(
    "Current Price",
    f"₹{price:.2f}",
)

metric_2.metric(
    "Setup Score",
    f"{score}/100",
)

metric_3.metric(
    "Trend",
    trend,
)

metric_4.metric(
    "RSI",
    f"{row['RSI']:.1f}",
)


# ---------------------------------------------------------
# OPPORTUNITY SCORE
# ---------------------------------------------------------

st.subheader("🎯 Opportunity Score")

st.progress(
    score / 100
)

st.write(
    f"**{stars} — {quality}**"
)


# ---------------------------------------------------------
# ACTION PLAN
# ---------------------------------------------------------

st.subheader("🧭 Action Plan")

action_1, action_2, action_3, action_4 = st.columns(4)

action_1.metric(
    "Entry Reference",
    f"₹{price:.2f}",
)

action_2.metric(
    "ATR Stop",
    f"₹{stop:.2f}",
)

action_3.metric(
    "2R Target",
    f"₹{target_2r:.2f}",
)

action_4.metric(
    "3R Target",
    f"₹{target_3r:.2f}",
)


# ---------------------------------------------------------
# POSITION SIZE
# ---------------------------------------------------------

risk_per_unit = max(
    price - stop,
    0.01,
)

risk_budget = (
    capital
    * risk_percent
    / 100
)

quantity_by_risk = int(
    risk_budget // risk_per_unit
)

quantity_by_capital = int(
    capital // price
)

recommended_quantity = min(
    quantity_by_risk,
    quantity_by_capital,
)

position_value = (
    recommended_quantity
    * price
)

st.subheader("💰 Position Sizing")

position_1, position_2, position_3 = st.columns(3)

position_1.metric(
    "Maximum Risk Budget",
    f"₹{risk_budget:.2f}",
)

position_2.metric(
    "Maximum Quantity",
    str(recommended_quantity),
)

position_3.metric(
    "Approx. Position Value",
    f"₹{position_value:.2f}",
)


# ---------------------------------------------------------
# WHY THIS SIGNAL
# ---------------------------------------------------------

st.subheader("🧠 Why V5 Gave This Signal")

for reason in reasons:

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


# ---------------------------------------------------------
# BUY REQUIREMENTS
# ---------------------------------------------------------

st.subheader("🚦 What Needs to Happen Before BUY?")

if signal in [
    "🟢 BUY",
    "🟢 STRONG BUY",
]:

    st.success(
        "The current setup already meets "
        "the V5 BUY threshold."
    )

else:

    if requirements:

        for requirement in requirements:

            st.write(
                f"• {requirement}"
            )

    else:

        st.info(
            "No additional technical "
            "requirements detected."
        )


# ---------------------------------------------------------
# INTERACTIVE PRICE CHART
# ---------------------------------------------------------

st.subheader("📊 Interactive Price Trend")

chart_data = df.tail(250)

figure = go.Figure()

figure.add_trace(
    go.Candlestick(
        x=chart_data.index,
        open=chart_data["Open"],
        high=chart_data["High"],
        low=chart_data["Low"],
        close=chart_data["Close"],
        name="Price",
    )
)

figure.add_trace(
    go.Scatter(
        x=chart_data.index,
        y=chart_data["EMA20"],
        name="EMA20",
    )
)

figure.add_trace(
    go.Scatter(
        x=chart_data.index,
        y=chart_data["EMA50"],
        name="EMA50",
    )
)

figure.add_trace(
    go.Scatter(
        x=chart_data.index,
        y=chart_data["EMA200"],
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


# ---------------------------------------------------------
# INDICATOR SNAPSHOT
# ---------------------------------------------------------

st.subheader("📋 Indicator Snapshot")

indicator_table = pd.DataFrame(
    {
        "Indicator": [
            "EMA20",
            "EMA50",
            "EMA200",
            "RSI",
            "MACD",
            "MACD Signal",
            "ADX",
            "20-Day Resistance",
            "20-Day Support",
            "ATR",
        ],
        "Value": [
            f"₹{row['EMA20']:.2f}",
            f"₹{row['EMA50']:.2f}",
            f"₹{row['EMA200']:.2f}",
            f"{row['RSI']:.2f}",
            f"{row['MACD']:.4f}",
            f"{row['MACD_SIGNAL']:.4f}",
            f"{row['ADX']:.2f}",
            f"₹{row['RES20']:.2f}",
            f"₹{row['SUP20']:.2f}",
            f"₹{row['ATR']:.2f}",
        ],
    }
)

st.dataframe(
    indicator_table,
    use_container_width=True,
    hide_index=True,
)


# ---------------------------------------------------------
# IMPORTANT NOTICE
# ---------------------------------------------------------

st.warning(
    "V5 is a rule-based technical research tool. "
    "A high setup score does not guarantee profit. "
    "Always verify live prices with your broker and "
    "respect the defined risk limit."
)
