import math


def clamp(value, minimum=0.0, maximum=100.0):
    return max(minimum, min(maximum, value))


def sigmoid(value):
    return 1 / (1 + math.exp(-value))


def calculate_probability(result):
    row = result["row"]
    score = float(result["score"])

    probability_score = 0.0

    # Base setup quality
    probability_score += (score - 50) / 18

    # Trend structure
    if row["Close"] > row["EMA20"]:
        probability_score += 0.55
    else:
        probability_score -= 0.55

    if row["EMA20"] > row["EMA50"]:
        probability_score += 0.65
    else:
        probability_score -= 0.65

    if row["EMA50"] > row["EMA200"]:
        probability_score += 0.40
    else:
        probability_score -= 0.40

    if row["Close"] > row["EMA200"]:
        probability_score += 0.25
    else:
        probability_score -= 0.25

    # Momentum
    rsi = float(row["RSI"])

    if 50 <= rsi <= 65:
        probability_score += 0.60
    elif 45 <= rsi < 50:
        probability_score += 0.10
    elif rsi < 40:
        probability_score -= 0.45
    elif rsi > 75:
        probability_score -= 0.35

    if row["MACD"] > row["MACD_SIGNAL"]:
        probability_score += 0.40
    else:
        probability_score -= 0.30

    # Trend strength
    adx = float(row["ADX"])

    if adx >= 25:
        probability_score += 0.35
    elif adx >= 20:
        probability_score += 0.15
    else:
        probability_score -= 0.20

    # Volume
    if row["Volume"] > row["VOL20"] * 1.2:
        probability_score += 0.30

    # Breakout
    if row["Close"] > row["RES20"]:
        probability_score += 0.70
    else:
        probability_score -= 0.20

    probability = sigmoid(probability_score) * 100

    return round(clamp(probability, 5, 95), 1)


def calculate_risk_level(result):
    row = result["row"]

    risk_score = 0

    if row["Close"] < row["EMA20"]:
        risk_score += 2

    if row["EMA20"] < row["EMA50"]:
        risk_score += 2

    if row["Close"] < row["EMA200"]:
        risk_score += 2

    if row["RSI"] < 40:
        risk_score += 1

    if row["MACD"] < row["MACD_SIGNAL"]:
        risk_score += 1

    if row["ADX"] < 20:
        risk_score += 1

    atr_percent = (
        float(row["ATR"])
        / float(row["Close"])
        * 100
    )

    if atr_percent >= 4:
        risk_score += 2
    elif atr_percent >= 2.5:
        risk_score += 1

    if risk_score >= 8:
        return "🔴 VERY HIGH", risk_score

    if risk_score >= 5:
        return "🟠 HIGH", risk_score

    if risk_score >= 3:
        return "🟡 MEDIUM", risk_score

    return "🟢 LOW", risk_score


def calculate_reward_risk(result):
    price = float(result["price"])
    stop = float(result["stop"])
    target_2r = float(result["target_2r"])
    target_3r = float(result["target_3r"])

    risk_amount = max(price - stop, 0.01)

    reward_2r = max(target_2r - price, 0)
    reward_3r = max(target_3r - price, 0)

    risk_percent = (
        risk_amount / price
    ) * 100

    reward_2r_percent = (
        reward_2r / price
    ) * 100

    reward_3r_percent = (
        reward_3r / price
    ) * 100

    rr_2r = reward_2r / risk_amount
    rr_3r = reward_3r / risk_amount

    return {
        "risk_percent": round(risk_percent, 2),
        "reward_2r_percent": round(
            reward_2r_percent,
            2,
        ),
        "reward_3r_percent": round(
            reward_3r_percent,
            2,
        ),
        "rr_2r": round(rr_2r, 2),
        "rr_3r": round(rr_3r, 2),
    }


def calculate_trade_quality(probability, risk_score):
    adjusted = probability - (risk_score * 2)

    if adjusted >= 80:
        return "★★★★★", "EXCELLENT"

    if adjusted >= 68:
        return "★★★★☆", "VERY GOOD"

    if adjusted >= 55:
        return "★★★☆☆", "AVERAGE"

    if adjusted >= 40:
        return "★★☆☆☆", "WEAK"

    return "★☆☆☆☆", "POOR"


def calculate_recommendation(
    result,
    probability,
    risk_score,
):
    row = result["row"]

    bullish_structure = (
        row["Close"] > row["EMA20"]
        and row["EMA20"] > row["EMA50"]
    )

    strong_structure = (
        bullish_structure
        and row["EMA50"] > row["EMA200"]
        and row["Close"] > row["EMA200"]
    )

    momentum_confirmed = (
        row["RSI"] >= 50
        and row["MACD"] > row["MACD_SIGNAL"]
    )

    breakout_confirmed = (
        row["Close"] > row["RES20"]
    )

    if (
        probability >= 78
        and strong_structure
        and momentum_confirmed
        and breakout_confirmed
        and risk_score <= 3
    ):
        return {
            "decision": "🟢 STRONG BUY",
            "action": "BUY",
            "allocation": 100,
        }

    if (
        probability >= 65
        and bullish_structure
        and momentum_confirmed
        and risk_score <= 4
    ):
        return {
            "decision": "🟢 BUY",
            "action": "BUY",
            "allocation": 50,
        }

    if (
        probability >= 50
        and row["MACD"] > row["MACD_SIGNAL"]
        and row["RSI"] >= 45
    ):
        return {
            "decision": "🟠 WATCH CLOSELY",
            "action": "WAIT",
            "allocation": 0,
        }

    if probability >= 35:
        return {
            "decision": "🟡 WAIT",
            "action": "WAIT",
            "allocation": 0,
        }

    return {
        "decision": "🔴 AVOID",
        "action": "DO NOT BUY",
        "allocation": 0,
    }


def calculate_confidence(
    probability,
    recommendation,
):
    action = recommendation["action"]

    if action == "BUY":
        confidence = probability

    elif action == "DO NOT BUY":
        confidence = 100 - probability

    else:
        confidence = abs(
            probability - 50
        ) + 50

    return round(
        clamp(confidence, 50, 95),
        1,
    )


def build_ai_view(
    result,
    probability,
    risk_level,
    recommendation,
):
    row = result["row"]

    paragraphs = []

    if row["Close"] < row["EMA20"]:
        paragraphs.append(
            "Price remains below the short-term EMA20, "
            "which indicates that buyers have not yet "
            "regained short-term control."
        )
    else:
        paragraphs.append(
            "Price is trading above EMA20, showing "
            "improvement in short-term market structure."
        )

    if row["EMA20"] < row["EMA50"]:
        paragraphs.append(
            "EMA20 remains below EMA50. The short-term "
            "trend has not fully recovered."
        )
    else:
        paragraphs.append(
            "EMA20 is above EMA50, supporting a bullish "
            "short-term trend structure."
        )

    rsi = float(row["RSI"])

    if rsi < 40:
        paragraphs.append(
            f"RSI is weak at {rsi:.1f}. Momentum remains "
            "under pressure."
        )
    elif rsi < 50:
        paragraphs.append(
            f"RSI has improved to {rsi:.1f}, but momentum "
            "has not yet crossed the preferred bullish zone."
        )
    elif rsi <= 70:
        paragraphs.append(
            f"RSI at {rsi:.1f} supports constructive "
            "bullish momentum."
        )
    else:
        paragraphs.append(
            f"RSI at {rsi:.1f} indicates an extended "
            "market condition."
        )

    if row["MACD"] > row["MACD_SIGNAL"]:
        paragraphs.append(
            "MACD momentum is improving, which is an early "
            "positive signal."
        )
    else:
        paragraphs.append(
            "MACD remains bearish and does not confirm "
            "positive momentum."
        )

    if recommendation["action"] == "BUY":
        conclusion = (
            f"The quantitative model estimates a "
            f"{probability:.1f}% favourable setup probability. "
            f"Risk is currently classified as {risk_level}. "
            "The current structure satisfies the engine's "
            "entry requirements."
        )

    elif recommendation["action"] == "WAIT":
        conclusion = (
            f"The model estimates a {probability:.1f}% "
            "favourable setup probability. Some positive "
            "conditions may be developing, but confirmation "
            "is insufficient for a new entry."
        )

    else:
        conclusion = (
            f"The model estimates only a {probability:.1f}% "
            "favourable setup probability. Current technical "
            "conditions do not justify a new position."
        )

    paragraphs.append(conclusion)

    return paragraphs


def ai_analyze(result):
    probability = calculate_probability(result)

    risk_level, risk_score = calculate_risk_level(
        result
    )

    reward_risk = calculate_reward_risk(result)

    recommendation = calculate_recommendation(
        result,
        probability,
        risk_score,
    )

    confidence = calculate_confidence(
        probability,
        recommendation,
    )

    stars, trade_quality = calculate_trade_quality(
        probability,
        risk_score,
    )

    ai_view = build_ai_view(
        result,
        probability,
        risk_level,
        recommendation,
    )

    return {
        "probability": probability,
        "risk_level": risk_level,
        "risk_score": risk_score,
        "reward_risk": reward_risk,
        "recommendation": recommendation,
        "confidence": confidence,
        "stars": stars,
        "trade_quality": trade_quality,
        "ai_view": ai_view,
    }
