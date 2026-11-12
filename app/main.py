"""
main.py
=======
EcoSight Flask Application — Sovereign Debt Risk Prediction Platform.

Routes:
  GET  /              — Prediction form (index)
  POST /              — Run prediction, return XAI + recommendations
  GET  /dashboard     — Analytics dashboard
  GET  /compare       — Country comparison
  GET  /upload        — CSV batch-prediction upload
  POST /upload        — Process uploaded CSV
  GET  /forecast      — Per-country forecasting page
  POST /download-report — Generate + stream PDF report

API Routes (JSON):
  /api/stats          /api/risk-distribution   /api/debt-trend
  /api/risk-by-year   /api/scatter             /api/top-countries
  /api/countries      /api/country/<name>
  /api/compare        /api/forecast/<country>
"""

import os
import sys
from io import BytesIO
from datetime import datetime

import joblib
import pandas as pd
import numpy as np
from flask import (
    Flask, render_template, request,
    jsonify, send_file, flash, redirect, url_for,
)

# ── Path setup so Flask can find src/ modules ─────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR  = os.path.join(BASE_DIR, "..", "src")
sys.path.insert(0, SRC_DIR)

from risk_scorer   import compute_risk_score, generate_explanation, get_risk_gauge_data
from recommender   import get_recommendations
from forecaster    import forecast_country
from pdf_generator import generate_pdf_report

# ── Flask app setup ───────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = "ecosight-secret-key-2026"

# ── Ensure upload directory exists ────────────────────────────────────
UPLOAD_DIR = os.path.join(BASE_DIR, "static", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ── Load model and dataset once at startup ────────────────────────────
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "debt_model.pkl")
DATA_PATH  = os.path.join(BASE_DIR, "..", "data",   "final_dataset.csv")

model = joblib.load(MODEL_PATH)
df    = pd.read_csv(DATA_PATH)

LABELS   = {0: "Low", 1: "Medium", 2: "High"}
FEATURES = ["GDP_growth", "Inflation", "Debt_GDP", "Population"]


# ═══════════════════════════════════════════════════════════════════════
#  PAGE ROUTES
# ═══════════════════════════════════════════════════════════════════════

@app.route("/", methods=["GET", "POST"])
def index():
    """
    Prediction page.
    GET  → render empty form.
    POST → run RF model, return score, XAI, recommendations.
    """
    result = None

    if request.method == "POST":
        try:
            gdp        = float(request.form["gdp"])
            inflation  = float(request.form["inflation"])
            debt       = float(request.form["debt"])
            population = float(request.form["population"])   # stored in millions

            # Build feature dict (population converted to raw number)
            inputs = {
                "GDP_growth": gdp,
                "Inflation":  inflation,
                "Debt_GDP":   debt,
                "Population": population * 1_000_000,
            }

            data  = pd.DataFrame([inputs])
            pred  = model.predict(data)[0]
            proba = model.predict_proba(data)[0].tolist()

            prediction   = LABELS[pred]
            score        = compute_risk_score(proba)
            gauge        = get_risk_gauge_data(score)
            explanations = generate_explanation(inputs, proba, model)
            recs         = get_recommendations(
                prediction,
                {"GDP_growth": gdp, "Inflation": inflation,
                 "Debt_GDP": debt,  "Population": population},
            )

            result = {
                "prediction":      prediction,
                "score":           score,
                "gauge":           gauge,
                "proba":           {k: round(v * 100, 1)
                                    for k, v in zip(["Low", "Medium", "High"], proba)},
                "explanations":    explanations,
                "recommendations": recs,
                "inputs": {
                    "gdp": gdp, "inflation": inflation,
                    "debt": debt, "population": population,
                },
            }

        except ValueError:
            result = {"error": "Invalid input — please enter numeric values."}
        except Exception as exc:
            result = {"error": f"Prediction failed: {exc}"}

    return render_template("index.html", result=result)


@app.route("/dashboard")
def dashboard():
    """Analytics dashboard."""
    return render_template("dashboard.html")


@app.route("/compare")
def compare():
    """Country comparison page."""
    countries = sorted(df["Country"].unique().tolist())
    return render_template("compare.html", countries=countries)


@app.route("/upload", methods=["GET", "POST"])
def upload():
    """
    CSV batch-prediction upload.
    Accepts a CSV with at least the four model features.
    Returns prediction + risk score for every row.
    """
    results, error, summary = None, None, None

    if request.method == "POST":
        if "file" not in request.files:
            error = "No file part found in the request."
        else:
            f = request.files["file"]
            if not f.filename:
                error = "No file selected."
            elif not f.filename.lower().endswith(".csv"):
                error = "Only CSV files are accepted."
            else:
                try:
                    udf     = pd.read_csv(f)
                    missing = set(FEATURES) - set(udf.columns)

                    if missing:
                        error = (f"Missing required columns: "
                                 f"{', '.join(sorted(missing))}. "
                                 f"Required: {', '.join(FEATURES)}")
                    else:
                        X    = udf[FEATURES].copy()
                        pred = model.predict(X)
                        prob = model.predict_proba(X)

                        udf["Predicted_Risk"] = [LABELS[p] for p in pred]
                        udf["Risk_Score"]     = [compute_risk_score(p) for p in prob]

                        results = udf.to_dict(orient="records")

                        # Summary stats
                        rc = udf["Predicted_Risk"].value_counts().to_dict()
                        summary = {
                            "total":  len(udf),
                            "low":    rc.get("Low",    0),
                            "medium": rc.get("Medium", 0),
                            "high":   rc.get("High",   0),
                            "avg_score": round(udf["Risk_Score"].mean(), 1),
                        }

                except Exception as exc:
                    error = f"Error processing file: {exc}"

    return render_template("upload.html", results=results,
                           error=error, summary=summary)


@app.route("/forecast")
def forecast_page():
    """Per-country debt forecasting page."""
    countries = sorted(df["Country"].unique().tolist())
    return render_template("forecast.html", countries=countries)


# ═══════════════════════════════════════════════════════════════════════
#  PDF DOWNLOAD
# ═══════════════════════════════════════════════════════════════════════

@app.route("/download-report", methods=["POST"])
def download_report():
    """
    Re-runs prediction from JSON payload and returns a PDF report.
    Payload keys: gdp, inflation, debt, population
    """
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({"error": "No JSON payload received."}), 400

        gdp        = float(data["gdp"])
        inflation  = float(data["inflation"])
        debt       = float(data["debt"])
        population = float(data["population"])

        inputs = {
            "GDP_growth": gdp,
            "Inflation":  inflation,
            "Debt_GDP":   debt,
            "Population": population * 1_000_000,
        }

        dfi          = pd.DataFrame([inputs])
        pred         = model.predict(dfi)[0]
        proba        = model.predict_proba(dfi)[0].tolist()
        prediction   = LABELS[pred]
        score        = compute_risk_score(proba)
        explanations = generate_explanation(inputs, proba, model)
        recs         = get_recommendations(
            prediction,
            {"GDP_growth": gdp, "Inflation": inflation,
             "Debt_GDP": debt,  "Population": population},
        )

        pdf_bytes = generate_pdf_report({
            "gdp": gdp, "inflation": inflation,
            "debt": debt, "population": population,
            "prediction":      prediction,
            "score":           score,
            "explanations":    explanations,
            "recommendations": recs,
        })

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return send_file(
            BytesIO(pdf_bytes),
            mimetype="application/pdf",
            as_attachment=True,
            download_name=f"debt_risk_report_{timestamp}.pdf",
        )

    except KeyError as ke:
        return jsonify({"error": f"Missing field: {ke}"}), 400
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


# ═══════════════════════════════════════════════════════════════════════
#  JSON API ROUTES
# ═══════════════════════════════════════════════════════════════════════

@app.route("/api/stats")
def api_stats():
    """KPI statistics for the dashboard."""
    total  = len(df)
    return jsonify({
        "total_countries": int(df["Country"].nunique()),
        "total_records":   total,
        "avg_debt":        round(float(df["Debt_GDP"].mean()), 2),
        "avg_gdp":         round(float(df["GDP_growth"].mean()), 2),
        "avg_inflation":   round(float(df["Inflation"].mean()), 2),
        "high_risk_pct":   round(len(df[df["Debt_Risk"] == "High"]) / total * 100, 1),
    })


@app.route("/api/risk-distribution")
def api_risk_distribution():
    """Donut chart data."""
    c = df["Debt_Risk"].value_counts().to_dict()
    return jsonify({
        "labels": ["Low", "Medium", "High"],
        "values": [c.get("Low", 0), c.get("Medium", 0), c.get("High", 0)],
    })


@app.route("/api/debt-trend")
def api_debt_trend():
    """Annual average macro indicators for line chart."""
    t = (df.groupby("Year")[["Debt_GDP", "GDP_growth", "Inflation"]]
           .mean().round(2).reset_index().sort_values("Year"))
    return jsonify({
        "years":     t["Year"].tolist(),
        "debt":      t["Debt_GDP"].tolist(),
        "gdp":       t["GDP_growth"].tolist(),
        "inflation": t["Inflation"].tolist(),
    })


@app.route("/api/risk-by-year")
def api_risk_by_year():
    """Stacked bar chart data (risk counts per year)."""
    pivot = (df.groupby(["Year", "Debt_Risk"]).size()
               .unstack(fill_value=0).reset_index().sort_values("Year"))
    for col in ["Low", "Medium", "High"]:
        if col not in pivot.columns:
            pivot[col] = 0
    return jsonify({
        "years":  pivot["Year"].tolist(),
        "low":    pivot["Low"].tolist(),
        "medium": pivot["Medium"].tolist(),
        "high":   pivot["High"].tolist(),
    })


@app.route("/api/scatter")
def api_scatter():
    """Scatter plot sample (GDP growth vs Debt/GDP, coloured by risk)."""
    sample    = df.sample(n=min(400, len(df)), random_state=42)
    color_map = {"Low": "#34d399", "Medium": "#fbbf24", "High": "#ef4444"}
    return jsonify([{
        "x":       round(float(r["GDP_growth"]), 2),
        "y":       round(float(r["Debt_GDP"]),   2),
        "risk":    r["Debt_Risk"],
        "country": r["Country"],
        "year":    int(r["Year"]),
        "color":   color_map[r["Debt_Risk"]],
    } for _, r in sample.iterrows()])


@app.route("/api/top-countries")
def api_top_countries():
    """Top 15 high-risk countries ranked by average Debt/GDP."""
    top = (df[df["Debt_Risk"] == "High"]
             .groupby("Country")
             .agg(avg_debt=("Debt_GDP","mean"), avg_gdp=("GDP_growth","mean"),
                  avg_infl=("Inflation","mean"), years_count=("Year","count"))
             .round(2).reset_index()
             .sort_values("avg_debt", ascending=False).head(15))
    return jsonify(top.to_dict(orient="records"))


@app.route("/api/countries")
def api_countries():
    """Full country list for search / dropdowns."""
    return jsonify(sorted(df["Country"].unique().tolist()))


@app.route("/api/country/<name>")
def api_country(name):
    """Time-series data for a single country."""
    sub = df[df["Country"].str.lower() == name.lower()].sort_values("Year")
    if sub.empty:
        return jsonify({"error": "Country not found"}), 404
    return jsonify({
        "country":   sub["Country"].iloc[0],
        "years":     sub["Year"].tolist(),
        "debt":      sub["Debt_GDP"].round(2).tolist(),
        "gdp":       sub["GDP_growth"].round(2).tolist(),
        "inflation": sub["Inflation"].round(2).tolist(),
        "risk":      sub["Debt_Risk"].tolist(),
    })


@app.route("/api/compare")
def api_compare():
    """
    Multi-country comparison data.
    Query param: ?countries=Greece,Japan,Brazil
    """
    names   = [c.strip() for c in request.args.get("countries", "").split(",") if c.strip()]
    results = {}
    for name in names:
        sub = df[df["Country"].str.lower() == name.lower()].sort_values("Year")
        if not sub.empty:
            cname = sub["Country"].iloc[0]
            results[cname] = {
                "years":       sub["Year"].tolist(),
                "debt":        sub["Debt_GDP"].round(2).tolist(),
                "gdp":         sub["GDP_growth"].round(2).tolist(),
                "inflation":   sub["Inflation"].round(2).tolist(),
                "risk":        sub["Debt_Risk"].tolist(),
                "avg_debt":    round(float(sub["Debt_GDP"].mean()), 2),
                "avg_gdp":     round(float(sub["GDP_growth"].mean()), 2),
                "latest_risk": sub["Debt_Risk"].iloc[-1],
            }
    return jsonify(results)


@app.route("/api/forecast/<country>")
def api_forecast(country):
    """Polynomial regression forecast for a single country."""
    years  = int(request.args.get("years", 5))
    result = forecast_country(df, country, years_ahead=years)
    return jsonify(result)


# ═══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    app.run(debug=True)