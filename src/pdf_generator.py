"""
pdf_generator.py
================
Generates a professional A4 PDF report using ReportLab (pure Python).
No system-level binary dependencies required.

Sections:
  1. EcoSight branded header
  2. Risk classification + 0-100 score
  3. Input economic indicators table
  4. Explainable AI feature contributions table
  5. Economic policy recommendations
  6. Legal disclaimer footer
"""

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer,
    Table, TableStyle, HRFlowable,
)
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from io import BytesIO
from datetime import datetime

# ── Brand colours ─────────────────────────────────────────────────────
INDIGO     = colors.HexColor("#6366f1")
LOW_CLR    = colors.HexColor("#34d399")
MED_CLR    = colors.HexColor("#fbbf24")
HIGH_CLR   = colors.HexColor("#ef4444")
DARK_GRAY  = colors.HexColor("#1e293b")
MID_GRAY   = colors.HexColor("#64748b")
LIGHT_GRAY = colors.HexColor("#f1f5f9")
ALT_ROW    = colors.HexColor("#e2e8f0")


def _risk_color(risk: str):
    return {"Low": LOW_CLR, "Medium": MED_CLR, "High": HIGH_CLR}.get(risk, MID_GRAY)


def _styles():
    """Return a dict of named ParagraphStyles."""
    B = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("T", parent=B["Title"], fontSize=21,
                                textColor=INDIGO, fontName="Helvetica-Bold",
                                spaceAfter=4, alignment=TA_CENTER),
        "sub":   ParagraphStyle("S", parent=B["Normal"], fontSize=10,
                                textColor=MID_GRAY, spaceAfter=14, alignment=TA_CENTER),
        "h2":    ParagraphStyle("H2", parent=B["Heading2"], fontSize=12,
                                textColor=INDIGO, fontName="Helvetica-Bold",
                                spaceBefore=14, spaceAfter=6),
        "body":  ParagraphStyle("Bo", parent=B["Normal"], fontSize=9.5,
                                leading=14, alignment=TA_JUSTIFY),
        "bull":  ParagraphStyle("Bu", parent=B["Normal"], fontSize=9.5,
                                leading=15, leftIndent=14),
        "risk":  ParagraphStyle("R",  parent=B["Normal"], fontSize=18,
                                fontName="Helvetica-Bold", alignment=TA_CENTER, spaceAfter=4),
        "score": ParagraphStyle("Sc", parent=B["Normal"], fontSize=12,
                                textColor=MID_GRAY, alignment=TA_CENTER, spaceAfter=14),
        "foot":  ParagraphStyle("F",  parent=B["Normal"], fontSize=7.5,
                                textColor=MID_GRAY, alignment=TA_CENTER),
    }


# ── Table helpers ──────────────────────────────────────────────────────
_TABLE_STYLE = [
    ("FONTSIZE",      (0, 0), (-1, -1), 9.5),
    ("GRID",          (0, 0), (-1, -1), 0.4, colors.lightgrey),
    ("ALIGN",         (0, 0), (-1, -1), "LEFT"),
    ("TOPPADDING",    (0, 0), (-1, -1), 6),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ("LEFTPADDING",   (0, 0), (-1, -1), 8),
    ("ROWBACKGROUNDS",(0, 1), (-1, -1), [LIGHT_GRAY, ALT_ROW]),
]


def _input_table(data: dict) -> Table:
    rows = [
        ["Indicator",        "Value",                          "Unit"],
        ["GDP Growth Rate",  str(data.get("gdp", "—")),        "%"],
        ["Inflation Rate",   str(data.get("inflation", "—")),   "%"],
        ["Debt / GDP Ratio", str(data.get("debt", "—")),       "% of GDP"],
        ["Population",       str(data.get("population", "—")), "Millions"],
    ]
    t = Table(rows, colWidths=[6.5*cm, 4*cm, 5.5*cm])
    t.setStyle(TableStyle(_TABLE_STYLE + [
        ("BACKGROUND",  (0, 0), (-1, 0), INDIGO),
        ("TEXTCOLOR",   (0, 0), (-1, 0), colors.white),
        ("FONTNAME",    (0, 0), (-1, 0), "Helvetica-Bold"),
    ]))
    return t


def _xai_table(explanations: list) -> Table:
    rows = [["Feature", "Importance", "Input Value", "Influence"]]
    for e in explanations:
        rows.append([
            e.get("label", e.get("feature", "")),
            f"{e.get('importance', 0):.1f}%",
            str(e.get("display_val", e.get("value", ""))),
            e.get("direction", ""),
        ])
    t = Table(rows, colWidths=[5.5*cm, 3.5*cm, 3.5*cm, 5.5*cm])
    t.setStyle(TableStyle(_TABLE_STYLE + [
        ("BACKGROUND",  (0, 0), (-1, 0), DARK_GRAY),
        ("TEXTCOLOR",   (0, 0), (-1, 0), colors.white),
        ("FONTNAME",    (0, 0), (-1, 0), "Helvetica-Bold"),
    ]))
    return t


def generate_pdf_report(prediction_data: dict) -> bytes:
    """
    Generate a professional PDF debt risk report.

    Args:
        prediction_data: dict with keys:
          gdp, inflation, debt, population,
          prediction, score, explanations, recommendations

    Returns:
        Raw PDF bytes suitable for Flask send_file().
    """
    buf = BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        rightMargin=2.2*cm, leftMargin=2.2*cm,
        topMargin=2.0*cm,  bottomMargin=2.0*cm,
        title="EcoSight Debt Risk Report", author="EcoSight AI",
    )

    S     = _styles()
    story = []

    # ── Header ──────────────────────────────────────────────────────
    story.append(Paragraph("EcoSight — Sovereign Debt Risk Report", S["title"]))
    story.append(Paragraph(
        f"Generated {datetime.utcnow().strftime('%d %B %Y  ·  %H:%M UTC')}",
        S["sub"],
    ))
    story.append(HRFlowable(width="100%", thickness=1.2, color=INDIGO, spaceAfter=16))

    # ── Risk result ──────────────────────────────────────────────────
    risk  = prediction_data.get("prediction", "N/A")
    score = prediction_data.get("score", 0)
    rc    = _risk_color(risk)

    story.append(Paragraph(
        f'<font color="#{rc.hexval()[1:]}">Risk Classification: {risk.upper()} RISK</font>',
        S["risk"],
    ))
    story.append(Paragraph(f"Debt Risk Score: {score} / 100", S["score"]))
    story.append(HRFlowable(width="100%", thickness=0.5,
                            color=colors.lightgrey, spaceAfter=12))

    # ── Input indicators ─────────────────────────────────────────────
    story.append(Paragraph("1 · Input Economic Indicators", S["h2"]))
    story.append(_input_table(prediction_data))
    story.append(Spacer(1, 12))

    # ── XAI ─────────────────────────────────────────────────────────
    story.append(Paragraph("2 · Explainable AI — Feature Contributions", S["h2"]))
    story.append(Paragraph(
        "Each feature's contribution is ranked by the Random Forest's global importance. "
        "The influence column reflects how this specific input value affects risk.",
        S["body"],
    ))
    story.append(Spacer(1, 7))
    if prediction_data.get("explanations"):
        story.append(_xai_table(prediction_data["explanations"]))
    story.append(Spacer(1, 12))

    # ── Recommendations ──────────────────────────────────────────────
    story.append(Paragraph("3 · Economic Policy Recommendations", S["h2"]))
    recs = prediction_data.get("recommendations", {})
    if recs:
        story.append(Paragraph(
            f"<b>Urgency:</b> {recs.get('urgency','')}  ·  {recs.get('summary','')}",
            S["body"],
        ))
        story.append(Spacer(1, 8))
        story.append(Paragraph("<b>Recommended Actions:</b>", S["body"]))
        for a in recs.get("actions", []):
            story.append(Paragraph(f"• {a}", S["bull"]))
        for d in recs.get("dynamic", []):
            story.append(Paragraph(f"▸ {d.get('message','')}", S["bull"]))

    story.append(Spacer(1, 16))

    # ── Footer ───────────────────────────────────────────────────────
    story.append(HRFlowable(width="100%", thickness=0.5,
                            color=colors.lightgrey, spaceAfter=8))
    story.append(Paragraph(
        "EcoSight · AI-Powered Sovereign Debt Risk Analysis · "
        "Powered by Random Forest + IMF / World Bank Data<br/>"
        "<i>For analytical purposes only. Not financial or investment advice.</i>",
        S["foot"],
    ))

    doc.build(story)
    return buf.getvalue()
