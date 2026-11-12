"""
forecaster.py
=============
Forecasts future Debt/GDP ratios for a given country using
Polynomial Regression (degree 2) fitted on historical data.
Returns historical + fitted + projected values ready for Chart.js.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score


def _classify_risk(debt: float) -> str:
    """Classify a Debt/GDP value into Low / Medium / High risk."""
    if debt > 90:
        return "High"
    elif debt > 60:
        return "Medium"
    return "Low"


def forecast_country(df: pd.DataFrame, country: str, years_ahead: int = 5) -> dict:
    """
    Forecast Debt/GDP for a specific country over the next `years_ahead` years.

    Algorithm:
      - Filter dataset for the given country
      - Fit Polynomial Regression (degree 2) on Year → Debt_GDP
      - Project future years, clamped to [0, 250]

    Args:
        df          : full dataset DataFrame
        country     : country name (case-insensitive)
        years_ahead : number of years to project (default 5)

    Returns:
        dict with historical, fitted, and forecast arrays; or {'error': msg}
    """
    cdf = df[df["Country"].str.lower() == country.lower()].sort_values("Year").copy()

    if cdf.empty:
        return {"error": f'Country "{country}" not found in the dataset.'}
    if len(cdf) < 3:
        return {"error": f'Not enough data for "{country}" — at least 3 years required.'}

    country_name = cdf["Country"].iloc[0]
    X = cdf["Year"].values.reshape(-1, 1)
    y = cdf["Debt_GDP"].values

    # Cap polynomial degree to avoid overfitting on small series
    degree = min(2, len(cdf) - 1)
    pipe   = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    pipe.fit(X, y)

    # Fitted values on historical range
    y_fitted = np.clip(pipe.predict(X), 0, 250)
    r2       = round(float(r2_score(y, y_fitted)), 3)

    # Future projection
    last_year    = int(cdf["Year"].max())
    future_years = list(range(last_year + 1, last_year + years_ahead + 1))
    y_future     = np.clip(pipe.predict(np.array(future_years).reshape(-1, 1)), 0, 250)

    return {
        "country":          country_name,
        "hist_years":       cdf["Year"].tolist(),
        "hist_debt":        cdf["Debt_GDP"].round(2).tolist(),
        "hist_gdp":         cdf["GDP_growth"].round(2).tolist(),
        "hist_inflation":   cdf["Inflation"].round(2).tolist(),
        "hist_risk":        cdf["Debt_Risk"].tolist(),
        "fitted_debt":      [round(float(v), 2) for v in y_fitted],
        "forecast_years":   future_years,
        "forecast_debt":    [round(float(v), 2) for v in y_future],
        "forecast_risk":    [_classify_risk(v) for v in y_future],
        "last_actual_year": last_year,
        "last_actual_debt": round(float(cdf["Debt_GDP"].iloc[-1]), 2),
        "r2_score":         r2,
    }
