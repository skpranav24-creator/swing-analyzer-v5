import yfinance as yf
import pandas as pd
import numpy as np


def normalize_symbol(symbol):
    ticker = symbol.strip().upper()

    if not ticker.endswith(".NS"):
        ticker += ".NS"

    return ticker


def get_data(symbol, period="5y"):
    ticker = normalize_symbol(symbol)

    df = yf.download(
        ticker,
        period=period,
        interval="1d",
        auto_adjust=False,
        progress=False,
        threads=False,
    )

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    required_columns = ["Open", "High", "Low", "Close", "Volume"]

    if df.empty:
        raise ValueError(f"No market data received for {ticker}")

    missing = [column for column in required_columns if column not in df.columns]

    if missing:
        raise ValueError(
            f"Missing market data columns for {ticker}: {', '.join(missing)}"
        )

    df = df[required_columns].dropna()

    if len(df) < 220:
        raise ValueError(
            f"Not enough historical data for {ticker}. "
            f"Received only {len(df)} rows."
        )

    return ticker, df


def calculate_rsi(series, period=14):
    delta = series.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    average_gain = gain.ewm(
        alpha=1 / period,
        adjust=False,
    ).mean()

    average_loss = loss.ewm(
        alpha=1 / period,
        adjust=False,
    ).mean()

    rs = average_gain / average_loss.replace(0, np.nan)

    return 100 - (100 / (1 + rs))


def calculate_indicators(df):
    data = df.copy()

    data["EMA20"] = data["Close"].ewm(
        span=20,
        adjust=False,
    ).mean()

    data["EMA50"] = data["Close"].ewm(
        span=50,
        adjust=False,
    ).mean()

    data["EMA200"] = data["Close"].ewm(
        span=200,
        adjust=False,
    ).mean()

    data["RSI"] = calculate_rsi(data["Close"])

    ema12 = data["Close"].ewm(
        span=12,
        adjust=False,
    ).mean()

    ema26 = data["Close"].ewm(
        span=26,
        adjust=False,
    ).mean()

    data["MACD"] = ema12 - ema26

    data["MACD_SIGNAL"] = data["MACD"].ewm(
        span=9,
        adjust=False,
    ).mean()

    data["VOL20"] = data["Volume"].rolling(20).mean()

    true_range = pd.concat(
        [
            data["High"] - data["Low"],
            (data["High"] - data["Close"].shift()).abs(),
            (data["Low"] - data["Close"].shift()).abs(),
        ],
        axis=1,
    ).max(axis=1)

    data["ATR"] = true_range.rolling(14).mean()

    up_move = data["High"].diff()
    down_move = -data["Low"].diff()

    plus_dm = pd.Series(
        np.where(
            (up_move > down_move) & (up_move > 0),
            up_move,
            0.0,
        ),
        index=data.index,
    )

    minus_dm = pd.Series(
        np.where(
            (down_move > up_move) & (down_move > 0),
            down_move,
            0.0,
        ),
        index=data.index,
    )

    atr_smoothed = true_range.ewm(
        alpha=1 / 14,
        adjust=False,
    ).mean()

    plus_di = (
        100
        * plus_dm.ewm(alpha=1 / 14, adjust=False).mean()
        / atr_smoothed.replace(0, np.nan)
    )

    minus_di = (
        100
        * minus_dm.ewm(alpha=1 / 14, adjust=False).mean()
        / atr_smoothed.replace(0, np.nan)
    )

    dx = (
        100
        * (plus_di - minus_di).abs()
        / (plus_di + minus_di).replace(0, np.nan)
    )

    data["ADX"] = dx.ewm(
        alpha=1 / 14,
        adjust=False,
    ).mean()

    data["RES20"] = (
        data["High"]
        .shift(1)
        .rolling(20)
        .max()
    )

    data["SUP20"] = (
        data["Low"]
        .shift(1)
        .rolling(20)
        .min()
    )

    return data


def classify_trend(row):
    if (
        row["Close"] > row["EMA20"]
        > row["EMA50"]
        > row["EMA200"]
    ):
        return "🚀 STRONG UPTREND"

    if (
        row["Close"] > row["EMA50"]
        and row["EMA20"] > row["EMA50"]
    ):
        return "📈 UPTREND"

    if (
        row["Close"] < row["EMA20"]
        < row["EMA50"]
        < row["EMA200"]
    ):
        return "📉 STRONG DOWNTREND"

    if row["Close"] < row["EMA50"]:
        return "🔻 DOWNTREND"

    return "↔️ SIDEWAYS"


def score_setup(row):
    score = 0
    reasons = []

    rules = [
        (
            row["Close"] > row["EMA20"],
            15,
            "Price is above EMA20",
            "Price is below EMA20",
        ),
        (
            row["EMA20"] > row["EMA50"],
            15,
            "EMA20 is above EMA50",
            "EMA20 is below EMA50",
        ),
        (
            row["EMA50"] > row["EMA200"],
            10,
            "EMA50 is above EMA200",
            "EMA50 is below EMA200",
        ),
        (
            row["Close"] > row["EMA200"],
            10,
            "Price is above EMA200",
            "Price is below EMA200",
        ),
        (
            50 <= row["RSI"] <= 70,
            15,
            "RSI is in the constructive 50–70 zone",
            f"RSI is {row['RSI']:.1f}, outside the preferred entry zone",
        ),
        (
            row["MACD"] > row["MACD_SIGNAL"],
            10,
            "MACD momentum is bullish",
            "MACD momentum is bearish",
        ),
        (
            row["ADX"] >= 20,
            10,
            "ADX confirms meaningful trend strength",
            "ADX shows weak trend strength",
        ),
        (
            row["Volume"] > row["VOL20"] * 1.2,
            5,
            "Volume confirms the move",
            "No strong volume confirmation",
        ),
        (
            row["Close"] > row["RES20"],
            10,
            "20-day breakout confirmed",
            "No 20-day breakout",
        ),
    ]

    for condition, points, good_reason, bad_reason in rules:
        if bool(condition):
            score += points
            reasons.append(
                {
                    "passed": True,
                    "points": points,
                    "text": good_reason,
                }
            )
        else:
            reasons.append(
                {
                    "passed": False,
                    "points": points,
                    "text": bad_reason,
                }
            )

    return score, reasons


def classify_signal(score):
    if score >= 85:
        return "🟢 STRONG BUY"

    if score >= 70:
        return "🟢 BUY"

    if score >= 55:
        return "🟠 WATCH"

    if score >= 35:
        return "🟡 WAIT"

    return "🔴 AVOID"


def opportunity_rating(score):
    if score >= 90:
        return "★★★★★", "Excellent Swing"

    if score >= 80:
        return "★★★★☆", "Very Good Swing"

    if score >= 70:
        return "★★★★☆", "Good Swing"

    if score >= 55:
        return "★★★☆☆", "Average Setup"

    if score >= 35:
        return "★★☆☆☆", "Weak Setup"

    return "★☆☆☆☆", "Avoid"


def calculate_buy_requirements(row):
    requirements = []

    if row["Close"] <= row["EMA20"]:
        requirements.append(
            f"Price should close above EMA20 near ₹{row['EMA20']:.2f}"
        )

    if row["EMA20"] <= row["EMA50"]:
        requirements.append(
            "EMA20 should move above EMA50"
        )

    if row["RSI"] < 50:
        requirements.append(
            "RSI should recover above 50"
        )

    if row["MACD"] <= row["MACD_SIGNAL"]:
        requirements.append(
            "MACD should turn bullish"
        )

    if row["ADX"] < 20:
        requirements.append(
            "ADX should strengthen above 20"
        )

    if row["Close"] <= row["RES20"]:
        requirements.append(
            f"Price should break 20-day resistance near ₹{row['RES20']:.2f}"
        )

    return requirements


def analyze(symbol):
    ticker, raw = get_data(symbol)

    df = calculate_indicators(raw)

    row = df.iloc[-1]

    score, reasons = score_setup(row)

    signal = classify_signal(score)

    trend = classify_trend(row)

    stars, quality = opportunity_rating(score)

    requirements = calculate_buy_requirements(row)

    price = float(row["Close"])
    atr = float(row["ATR"])

    stop = max(
        0.01,
        price - (1.5 * atr),
    )

    risk = max(
        price - stop,
        0.01,
    )

    target_2r = price + (2 * risk)
    target_3r = price + (3 * risk)

    return {
        "ticker": ticker,
        "data": df,
        "row": row,
        "price": price,
        "score": score,
        "signal": signal,
        "trend": trend,
        "stars": stars,
        "quality": quality,
        "reasons": reasons,
        "requirements": requirements,
        "stop": stop,
        "target_2r": target_2r,
        "target_3r": target_3r,
    }
