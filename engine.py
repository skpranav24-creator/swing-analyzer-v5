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

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing market data columns for {ticker}: "
            f"{', '.join(missing)}"
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
            (
                data["High"]
                - data["Close"].shift()
            ).abs(),
            (
                data["Low"]
                - data["Close"].shift()
            ).abs(),
        ],
        axis=1,
    ).max(axis=1)

    data["ATR"] = true_range.rolling(14).mean()

    up_move = data["High"].diff()
    down_move = -data["Low"].diff()

    plus_dm = pd.Series(
        np.where(
            (up_move > down_move)
            & (up_move > 0),
            up_move,
            0.0,
        ),
        index=data.index,
    )

    minus_dm = pd.Series(
        np.where(
            (down_move > up_move)
            & (down_move > 0),
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
        * plus_dm.ewm(
            alpha=1 / 14,
            adjust=False,
        ).mean()
        / atr_smoothed.replace(0, np.nan)
    )

    minus_di = (
        100
        * minus_dm.ewm(
            alpha=1 / 14,
            adjust=False,
        ).mean()
        / atr_smoothed.replace(0, np.nan)
    )

    dx = (
        100
        * (plus_di - minus_di).abs()
        / (
            plus_di + minus_di
        ).replace(0, np.nan)
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

    data["HIGH20"] = (
        data["High"]
        .rolling(20)
        .max()
    )

    data["LOW20"] = (
        data["Low"]
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
            (
                f"RSI is {row['RSI']:.1f}, outside "
                "the preferred entry zone"
            ),
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
        passed = bool(condition)

        if passed:
            score += points

        reasons.append(
            {
                "passed": passed,
                "points": points,
                "text": (
                    good_reason
                    if passed
                    else bad_reason
                ),
            }
        )

    return score, reasons


def calculate_reversal_score(row):
    score = 0
    reasons = []

    if row["MACD"] > row["MACD_SIGNAL"]:
        score += 20
        reasons.append(
            "MACD has turned bullish"
        )

    if row["RSI"] >= 40:
        score += 15
        reasons.append(
            "RSI has recovered to at least 40"
        )

    if row["RSI"] >= 50:
        score += 15
        reasons.append(
            "RSI has confirmed momentum above 50"
        )

    if row["Close"] > row["EMA20"]:
        score += 20
        reasons.append(
            "Price has reclaimed EMA20"
        )

    if row["EMA20"] > row["EMA50"]:
        score += 15
        reasons.append(
            "EMA20 has crossed above EMA50"
        )

    if row["Close"] > row["RES20"]:
        score += 15
        reasons.append(
            "20-day breakout is confirmed"
        )

    return score, reasons


def classify_stage(row, reversal_score):
    if (
        row["Close"] > row["RES20"]
        and row["RSI"] >= 55
        and row["MACD"] > row["MACD_SIGNAL"]
    ):
        return {
            "stage": "🚀 BREAKOUT CONFIRMED",
            "action": "BUY FINAL 25%",
            "allocation": 100,
            "stage_number": 4,
        }

    if (
        row["Close"] > row["EMA20"]
        and row["EMA20"] > row["EMA50"]
        and row["RSI"] >= 50
        and row["MACD"] > row["MACD_SIGNAL"]
    ):
        return {
            "stage": "🟢 BUY CONFIRMED",
            "action": "BUY / ADD 50%",
            "allocation": 75,
            "stage_number": 3,
        }

    if (
        row["Close"] > row["EMA20"]
        and row["RSI"] >= 45
        and row["MACD"] > row["MACD_SIGNAL"]
        and reversal_score >= 40
    ):
        return {
            "stage": "🟠 REVERSAL CONFIRMED",
            "action": "START WITH 25%",
            "allocation": 25,
            "stage_number": 2,
        }

    if (
        row["MACD"] > row["MACD_SIGNAL"]
        or row["RSI"] >= 40
    ):
        return {
            "stage": "🟡 EARLY REVERSAL WATCH",
            "action": "WAIT — DO NOT BUY YET",
            "allocation": 0,
            "stage_number": 1,
        }

    return {
        "stage": "🔴 FALLING / NO REVERSAL",
        "action": "DO NOT BUY",
        "allocation": 0,
        "stage_number": 0,
    }


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
            f"Price should close above EMA20 "
            f"near ₹{row['EMA20']:.2f}"
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
            f"Price should break 20-day resistance "
            f"near ₹{row['RES20']:.2f}"
        )

    return requirements


def generate_ai_decision(
    row,
    score,
    reversal_score,
    trend,
    stage_data,
):
    price = float(row["Close"])
    ema20 = float(row["EMA20"])
    ema50 = float(row["EMA50"])
    rsi = float(row["RSI"])
    resistance = float(row["RES20"])

    stage_number = stage_data["stage_number"]

    if stage_number == 4:
        verdict = "🚀 BUY / ADD FINAL STAGE"
        confidence = "HIGH"

        summary = (
            "The technical engine sees a confirmed breakout. "
            "Price has broken the 20-day resistance with "
            "supportive momentum conditions."
        )

    elif stage_number == 3:
        verdict = "✅ BUY"
        confidence = "HIGH"

        summary = (
            "The technical engine sees a confirmed bullish "
            "setup. Price and momentum conditions have improved "
            "enough for a staged entry."
        )

    elif stage_number == 2:
        verdict = "🟠 SMALL BUY ONLY"
        confidence = "MEDIUM"

        summary = (
            "A reversal is developing, but the full bullish "
            "setup is not confirmed. Only a small first-stage "
            "position is justified by the current rules."
        )

    elif stage_number == 1:
        verdict = "❌ DO NOT BUY NOW"
        confidence = "HIGH"

        summary = (
            "Some early reversal signs are visible, but there "
            "is not enough confirmation to buy. The price may "
            "still continue falling or move sideways."
        )

    else:
        verdict = "🛑 DO NOT BUY"
        confidence = "HIGH"

        summary = (
            "The technical engine does not detect a reliable "
            "reversal. Buying now would mean trying to catch "
            "a falling price without confirmation."
        )

    observations = []

    if price < ema20:
        observations.append(
            f"Price ₹{price:.2f} is below EMA20 "
            f"₹{ema20:.2f}."
        )
    else:
        observations.append(
            "Price has reclaimed EMA20."
        )

    if ema20 < ema50:
        observations.append(
            "EMA20 remains below EMA50, so the short-term "
            "trend has not fully recovered."
        )
    else:
        observations.append(
            "EMA20 is above EMA50, supporting a bullish trend."
        )

    if rsi < 40:
        observations.append(
            f"RSI is weak at {rsi:.1f}."
        )
    elif rsi < 50:
        observations.append(
            f"RSI is improving at {rsi:.1f}, but remains "
            "below bullish confirmation at 50."
        )
    elif rsi <= 70:
        observations.append(
            f"RSI at {rsi:.1f} supports constructive momentum."
        )
    else:
        observations.append(
            f"RSI at {rsi:.1f} is elevated and requires caution."
        )

    if row["MACD"] > row["MACD_SIGNAL"]:
        observations.append(
            "MACD momentum is improving."
        )
    else:
        observations.append(
            "MACD has not turned bullish."
        )

    next_trigger = []

    if price <= ema20:
        next_trigger.append(
            f"daily close above ₹{ema20:.2f}"
        )

    if rsi < 50:
        next_trigger.append(
            "RSI above 50"
        )

    if price <= resistance:
        next_trigger.append(
            f"breakout above ₹{resistance:.2f}"
        )

    if next_trigger:
        next_step = (
            "Watch for "
            + ", ".join(next_trigger)
            + "."
        )
    else:
        next_step = (
            "The main technical entry conditions are satisfied."
        )

    return {
        "verdict": verdict,
        "confidence": confidence,
        "summary": summary,
        "observations": observations,
        "next_step": next_step,
        "setup_score": score,
        "reversal_score": reversal_score,
        "trend": trend,
    }


def analyze(symbol):
    ticker, raw = get_data(symbol)

    df = calculate_indicators(raw)

    row = df.iloc[-1]

    score, reasons = score_setup(row)

    reversal_score, reversal_reasons = (
        calculate_reversal_score(row)
    )

    signal = classify_signal(score)

    trend = classify_trend(row)

    stars, quality = opportunity_rating(score)

    requirements = calculate_buy_requirements(row)

    stage_data = classify_stage(
        row,
        reversal_score,
    )

    ai_decision = generate_ai_decision(
        row=row,
        score=score,
        reversal_score=reversal_score,
        trend=trend,
        stage_data=stage_data,
    )

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
        "reversal_score": reversal_score,
        "reversal_reasons": reversal_reasons,
        "signal": signal,
        "trend": trend,
        "stars": stars,
        "quality": quality,
        "reasons": reasons,
        "requirements": requirements,
        "stage": stage_data["stage"],
        "stage_action": stage_data["action"],
        "stage_allocation": stage_data["allocation"],
        "stage_number": stage_data["stage_number"],
        "ai_decision": ai_decision,
        "stop": stop,
        "target_2r": target_2r,
        "target_3r": target_3r,
    }
