"""
recommender.py
==============
Generates context-aware, tiered economic policy recommendations
based on the predicted risk level and specific macroeconomic inputs.

Recommendation tiers:
    - Base actions   : fixed per risk level (Low / Medium / High)
    - Dynamic alerts : triggered by specific indicator thresholds
"""


# ── Base recommendation sets per risk level ──────────────────────────
BASE_RECOMMENDATIONS = {
    "Low": {
        "summary":    "The country maintains a sustainable debt trajectory. Focus on growth and resilience-building.",
        "urgency":    "Monitor",
        "urgency_cls": "success",
        "icon":       "bi-shield-check",
        "actions": [
            "Maintain fiscal discipline and structural primary surpluses.",
            "Invest in infrastructure, education, and human capital to sustain long-run growth.",
            "Build foreign currency reserves to buffer against external shocks.",
            "Implement counter-cyclical fiscal policy to smooth economic cycles.",
            "Diversify revenue sources to reduce commodity or trade dependency.",
            "Strengthen public financial management and debt transparency systems.",
        ],
    },
    "Medium": {
        "summary":    "Moderate debt vulnerability detected. Prudent fiscal management and consolidation are advised.",
        "urgency":    "Caution",
        "urgency_cls": "warning",
        "icon":       "bi-exclamation-triangle",
        "actions": [
            "Develop a credible medium-term fiscal consolidation roadmap.",
            "Reduce primary deficit through targeted expenditure reviews and efficiency gains.",
            "Explore liability management to extend debt maturities and lower refinancing risk.",
            "Broaden the tax base and strengthen revenue administration.",
            "Engage with multilateral lenders (IMF, World Bank) for technical assistance.",
            "Implement social safety nets to protect vulnerable groups during fiscal adjustment.",
            "Develop a Debt Management Strategy with clear debt-to-GDP reduction targets.",
        ],
    },
    "High": {
        "summary":    "Critical debt sustainability risk identified. Immediate structural fiscal reforms are necessary.",
        "urgency":    "Critical",
        "urgency_cls": "danger",
        "icon":       "bi-exclamation-octagon",
        "actions": [
            "Initiate immediate engagement with IMF / World Bank for emergency support.",
            "Pursue debt restructuring or renegotiation with bilateral and multilateral creditors.",
            "Implement emergency revenue mobilisation measures (e.g., windfall taxes, state-asset sales).",
            "Conduct a comprehensive expenditure review — cut non-essential spending immediately.",
            "Adopt a multi-year fiscal adjustment programme with independent oversight.",
            "Strengthen Central Bank independence and adopt formal inflation targeting.",
            "Consider exchange-rate stabilisation measures if currency depreciation is amplifying debt.",
            "Communicate a transparent crisis management plan to restore market confidence.",
        ],
    },
}


def get_recommendations(prediction: str, inputs: dict) -> dict:
    """
    Generate policy recommendations for a given risk prediction and indicator values.

    Args:
        prediction : 'Low', 'Medium', or 'High'
        inputs     : dict with keys GDP_growth, Inflation, Debt_GDP, Population (in millions)

    Returns:
        dict with keys: summary, urgency, urgency_cls, icon, actions, dynamic
    """
    gdp       = inputs.get("GDP_growth", 0.0)
    inflation = inputs.get("Inflation",  0.0)
    debt      = inputs.get("Debt_GDP",   0.0)

    # Clone base recommendation (avoid mutating original)
    result = dict(BASE_RECOMMENDATIONS.get(prediction, BASE_RECOMMENDATIONS["Low"]))
    result["actions"] = list(result["actions"])

    # ── Dynamic, input-specific alerts ──────────────────────────────
    dynamic = []

    if inflation > 15:
        dynamic.append({
            "icon":    "bi-thermometer-high",
            "color":   "danger",
            "message": f"Severe inflation ({inflation:.1f}%): Emergency monetary tightening required — coordinate with Central Bank on aggressive rate hikes and inflation anchoring.",
        })
    elif inflation > 10:
        dynamic.append({
            "icon":    "bi-graph-up-arrow",
            "color":   "warning",
            "message": f"High inflation ({inflation:.1f}%): Adopt tighter monetary policy — consider raising the policy rate to anchor inflation expectations.",
        })

    if gdp < -2:
        dynamic.append({
            "icon":    "bi-arrow-down-circle",
            "color":   "danger",
            "message": f"Severe recession ({gdp:.1f}% GDP growth): Targeted fiscal stimulus programmes required — prioritise job-retention schemes and public investment to restore confidence.",
        })
    elif gdp < 0:
        dynamic.append({
            "icon":    "bi-graph-down",
            "color":   "warning",
            "message": f"Negative GDP growth ({gdp:.1f}%): Implement targeted fiscal stimulus while maintaining medium-term debt sustainability commitments.",
        })

    if debt > 120:
        dynamic.append({
            "icon":    "bi-bank",
            "color":   "danger",
            "message": f"Extreme debt burden ({debt:.1f}% of GDP): Debt restructuring is likely unavoidable — engage creditors and multilateral institutions immediately.",
        })
    elif debt > 90:
        dynamic.append({
            "icon":    "bi-exclamation-circle",
            "color":   "warning",
            "message": f"Debt exceeds 90% of GDP ({debt:.1f}%): Debt-to-GDP reduction must be the primary fiscal objective — aim for a sustained primary surplus.",
        })

    if gdp > 7:
        dynamic.append({
            "icon":    "bi-piggy-bank",
            "color":   "success",
            "message": f"Strong growth ({gdp:.1f}% GDP): Channel surplus fiscal revenue into a Sovereign Wealth Fund to build long-term resilience.",
        })

    if inflation < 1 and gdp < 1:
        dynamic.append({
            "icon":    "bi-cloud-fog2",
            "color":   "info",
            "message": "Deflation / stagnation risk: Consider accommodative monetary policy and supply-side structural reforms to stimulate economic activity.",
        })

    result["dynamic"] = dynamic
    return result
