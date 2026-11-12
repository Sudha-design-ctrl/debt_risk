"""
risk_scorer.py
==============
Converts RandomForest probability outputs into a 0–100 risk score
and generates per-feature Explainable AI (XAI) explanations.

Score formula:
    score = P(Medium)*50 + P(High)*100
    → naturally maps to [0, 100], where 0 = fully Low-risk, 100 = fully High-risk
"""

import numpy as np

# Human-readable feature labels
FEATURE_LABELS = {
    "GDP_growth": "GDP Growth (%)",
    "Inflation":  "Inflation (%)",
    "Debt_GDP":   "Debt / GDP (%)",
    "Population": "Population (M)",
}

# Safe/risky thresholds derived from the dataset's classify_risk logic
THRESHOLDS = {
    "GDP_growth": {"safe": 3.0,  "risky": 0.0},    # > 3% is healthy
    "Inflation":  {"safe": 5.0,  "risky": 10.0},   # < 5% is healthy
    "Debt_GDP":   {"safe": 60.0, "risky": 90.0},   # < 60% is safe zone
    "Population": {"safe": None, "risky": None},    # Population is context-only
}


def compute_risk_score(proba: list) -> int:
    """
    Convert class probabilities [P(Low), P(Medium), P(High)] to a 0–100 score.

    Args:
        proba: list of 3 floats summing to 1.0

    Returns:
        Integer risk score in [0, 100]
    """
    p_low, p_medium, p_high = proba
    # Weighted position on a 0→100 axis: Low=0, Medium=50, High=100
    score = p_medium * 50.0 + p_high * 100.0
    return int(round(min(max(score, 0), 100)))


def generate_explanation(inputs: dict, proba: list, model) -> list:
    """
    Generate per-feature XAI explanations.

    Combines:
    - Global feature importance from the Random Forest model
    - Input-specific direction of influence (raises / lowers risk)

    Args:
        inputs:  dict with keys GDP_growth, Inflation, Debt_GDP, Population
        proba:   [P(Low), P(Medium), P(High)]
        model:   fitted sklearn RandomForestClassifier

    Returns:
        List of dicts, sorted by importance descending.
        Each dict: {feature, label, importance, value, influence, direction, bar_color}
    """
    feature_order = ["GDP_growth", "Inflation", "Debt_GDP", "Population"]
    importances   = model.feature_importances_  # shape (4,)

    explanations = []
    for i, feat in enumerate(feature_order):
        val = inputs.get(feat, 0)
        imp = float(importances[i])

        # Determine influence direction based on domain thresholds
        if feat == "GDP_growth":
            if val >= THRESHOLDS[feat]["safe"]:
                influence, direction, bar_color = "positive", "Lowers risk", "#34d399"
            elif val >= THRESHOLDS[feat]["risky"]:
                influence, direction, bar_color = "neutral",  "Mild concern",  "#fbbf24"
            else:
                influence, direction, bar_color = "negative", "Raises risk",   "#ef4444"

        elif feat == "Inflation":
            if val <= THRESHOLDS[feat]["safe"]:
                influence, direction, bar_color = "positive", "Lowers risk", "#34d399"
            elif val <= THRESHOLDS[feat]["risky"]:
                influence, direction, bar_color = "neutral",  "Mild concern",  "#fbbf24"
            else:
                influence, direction, bar_color = "negative", "Raises risk",   "#ef4444"

        elif feat == "Debt_GDP":
            if val <= THRESHOLDS[feat]["safe"]:
                influence, direction, bar_color = "positive", "Lowers risk", "#34d399"
            elif val <= THRESHOLDS[feat]["risky"]:
                influence, direction, bar_color = "neutral",  "Mild concern",  "#fbbf24"
            else:
                influence, direction, bar_color = "negative", "Raises risk",   "#ef4444"

        else:  # Population – informational only
            influence, direction, bar_color = "neutral", "Context only", "#6366f1"

        # Format display value (Population stored as raw number from millions input)
        if feat == "Population":
            display_val = f"{val / 1_000_000:.1f} M" if val >= 1_000_000 else f"{val:.0f}"
        elif feat in ("GDP_growth", "Inflation"):
            display_val = f"{val:.2f}%"
        else:
            display_val = f"{val:.1f}%"

        explanations.append({
            "feature":     feat,
            "label":       FEATURE_LABELS[feat],
            "importance":  round(imp * 100, 1),   # Convert to percentage
            "value":       val,
            "display_val": display_val,
            "influence":   influence,
            "direction":   direction,
            "bar_color":   bar_color,
        })

    # Sort by model importance (descending)
    explanations.sort(key=lambda x: x["importance"], reverse=True)
    return explanations


def get_risk_gauge_data(score: int) -> dict:
    """
    Return display metadata for the risk gauge based on the score.

    Returns:
        dict with keys: color, label, bs_class, description
    """
    if score <= 30:
        return {
            "color":       "#34d399",
            "label":       "Low Risk",
            "bs_class":    "success",
            "description": "Debt trajectory is sustainable given current macro conditions.",
        }
    elif score <= 65:
        return {
            "color":       "#fbbf24",
            "label":       "Moderate Risk",
            "bs_class":    "warning",
            "description": "Moderate vulnerability detected. Careful fiscal management advised.",
        }
    else:
        return {
            "color":       "#ef4444",
            "label":       "High Risk",
            "bs_class":    "danger",
            "description": "Critical debt sustainability risk. Immediate policy action required.",
        }
