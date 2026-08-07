"""
report_generator.py
====================
Generates a formal, downloadable PDF report for a single customer's churn
prediction. Used by the "Churn Prediction" dashboard page so a relationship
manager or business user can save/print/share a clean, professional summary
instead of just reading numbers off the screen.

The report includes:
    - Customer profile (the inputs used for the prediction)
    - Prediction outcome, probability, risk level, and confidence
    - Business recommendations for that risk level
    - Optional AI-generated explanation of the key drivers (SHAP), when
      Explainable AI is available for the selected model

ReportLab is the only new dependency this module introduces; it produces a
real PDF (not an HTML print-to-PDF hack), which is what most companies
expect from an exported report.
"""

from __future__ import annotations

import io
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

REPORT_TITLE = "Customer Churn Risk Report"


def _risk_color_hex(risk_level: str) -> str:
    """Map a risk level to a report accent color (as a hex string)."""
    return {
        "High": "#D64545",
        "Medium": "#E0A62C",
        "Low": "#2E9E5B",
    }.get(risk_level, "#333333")


def build_prediction_pdf(
    customer: Dict,
    prediction: Dict,
    explanation: Optional[List[str]] = None,
    generated_for: str = "Internal Use",
) -> bytes:
    """Build a formal PDF report for one customer's churn prediction.

    Args:
        customer: Raw input fields used for the prediction (CreditScore,
            Geography, Gender, Age, etc.).
        prediction: The prediction result as a dict, matching
            ``PredictionResult.as_dict()`` from ``src.prediction``.
        explanation: Optional list of plain-English sentences describing the
            key SHAP drivers behind this prediction.
        generated_for: Free-text label shown in the report header (e.g. a
            branch name or team), defaults to "Internal Use".

    Returns:
        The rendered PDF file as raw bytes, ready to hand to a download
        button — no temporary file is written to disk.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        title=REPORT_TITLE,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "ReportTitle", parent=styles["Title"], fontSize=20, textColor=colors.HexColor("#1F2937")
    )
    section_style = ParagraphStyle(
        "SectionHeading", parent=styles["Heading2"], fontSize=13, spaceBefore=14, spaceAfter=6,
        textColor=colors.HexColor("#1F2937"),
    )
    body_style = ParagraphStyle("Body", parent=styles["BodyText"], fontSize=10, leading=14)
    footer_style = ParagraphStyle(
        "Footer", parent=styles["Normal"], fontSize=8, textColor=colors.grey
    )

    elements = []

    # --- Header -----------------------------------------------------------
    elements.append(Paragraph(REPORT_TITLE, title_style))
    elements.append(
        Paragraph(
            f"Generated on {datetime.now().strftime('%d %B %Y, %H:%M')} &middot; {generated_for}",
            footer_style,
        )
    )
    elements.append(Spacer(1, 0.5 * cm))

    # --- Verdict banner -----------------------------------------------------
    risk_level = prediction.get("Risk Level", "Unknown")
    verdict_text = (
        f"<b>Prediction:</b> {prediction.get('Prediction', 'N/A')} &nbsp;&nbsp; "
        f"<b>Risk Level:</b> <font color='{_risk_color_hex(risk_level)}'>{risk_level}</font>"
    )
    elements.append(Paragraph(verdict_text, ParagraphStyle("Verdict", parent=body_style, fontSize=12)))
    elements.append(Spacer(1, 0.3 * cm))

    # --- Key metrics table ---------------------------------------------
    metrics_table_data = [
        ["Churn Probability", f"{prediction.get('Probability (%)', 0)}%"],
        ["Confidence Score", f"{prediction.get('Confidence (%)', 0)}%"],
        ["Model Used", prediction.get("Model Used", "N/A")],
    ]
    metrics_table = Table(metrics_table_data, colWidths=[6 * cm, 6 * cm])
    metrics_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#F3F4F6")),
                ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#1F2937")),
                ("FONTSIZE", (0, 0), (-1, -1), 10),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E5E7EB")),
            ]
        )
    )
    elements.append(metrics_table)

    # --- Customer profile ---------------------------------------------
    elements.append(Paragraph("Customer Profile", section_style))
    profile_rows = [[str(key), str(value)] for key, value in customer.items()]
    profile_table = Table(profile_rows, colWidths=[6 * cm, 6 * cm])
    profile_table.setStyle(
        TableStyle(
            [
                ("FONTSIZE", (0, 0), (-1, -1), 10),
                ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#1F2937")),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.white, colors.HexColor("#F9FAFB")]),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E5E7EB")),
            ]
        )
    )
    elements.append(profile_table)

    # --- Explainability section (optional) ------------------------------
    if explanation:
        elements.append(Paragraph("Key Drivers Behind This Prediction", section_style))
        for sentence in explanation:
            # Strip markdown bold markers — PDF uses <b> tags instead.
            clean_sentence = sentence.replace("**", "")
            elements.append(Paragraph(f"&bull; {clean_sentence}", body_style))

    # --- Recommendations -------------------------------------------------
    recommendations = prediction.get("Recommendation", "")
    if recommendations:
        elements.append(Paragraph("Recommended Next Steps", section_style))
        for rec in recommendations.split(" | "):
            if rec.strip():
                elements.append(Paragraph(f"&bull; {rec.strip()}", body_style))

    # --- Footer disclaimer -------------------------------------------------
    elements.append(Spacer(1, 1 * cm))
    elements.append(
        Paragraph(
            "This report is generated automatically from a trained machine learning model and is "
            "intended to support, not replace, human judgement in customer retention decisions.",
            footer_style,
        )
    )

    doc.build(elements)
    return buffer.getvalue()


def build_business_report_pdf(
    kpis: Dict[str, str],
    insights: List[str],
    recommendations: List[str],
    segment_table: Optional[pd.DataFrame] = None,
    generated_for: str = "Internal Use",
) -> bytes:
    """Build a business-level PDF report (headline KPIs, auto-generated
    insights, retention recommendations, and an optional top-segments
    table) — used by the Reports Center page for a leadership-ready export
    of the Business Insights page.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4, topMargin=2 * cm, bottomMargin=2 * cm,
        leftMargin=2 * cm, rightMargin=2 * cm, title="Business Insights Report",
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("ReportTitle", parent=styles["Title"], fontSize=20, textColor=colors.HexColor("#1F2937"))
    section_style = ParagraphStyle(
        "SectionHeading", parent=styles["Heading2"], fontSize=13, spaceBefore=14, spaceAfter=6,
        textColor=colors.HexColor("#1F2937"),
    )
    body_style = ParagraphStyle("Body", parent=styles["BodyText"], fontSize=10, leading=14)
    footer_style = ParagraphStyle("Footer", parent=styles["Normal"], fontSize=8, textColor=colors.grey)

    elements = [
        Paragraph("Business Insights Report", title_style),
        Paragraph(f"Generated on {datetime.now().strftime('%d %B %Y, %H:%M')} &middot; {generated_for}", footer_style),
        Spacer(1, 0.5 * cm),
    ]

    if kpis:
        elements.append(Paragraph("Headline KPIs", section_style))
        kpi_rows = [[str(k), str(v)] for k, v in kpis.items()]
        kpi_table = Table(kpi_rows, colWidths=[8 * cm, 6 * cm])
        kpi_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#F3F4F6")),
                    ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#1F2937")),
                    ("FONTSIZE", (0, 0), (-1, -1), 10),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                    ("TOPPADDING", (0, 0), (-1, -1), 6),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E5E7EB")),
                ]
            )
        )
        elements.append(kpi_table)

    if insights:
        elements.append(Paragraph("Strategic Insights", section_style))
        for sentence in insights:
            elements.append(Paragraph(f"&bull; {sentence.replace('**', '')}", body_style))

    if segment_table is not None and not segment_table.empty:
        elements.append(Paragraph("Top Customer Segments", section_style))
        table_data = [list(segment_table.columns)] + segment_table.astype(str).values.tolist()
        seg_table = Table(table_data, repeatRows=1)
        seg_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1F2937")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("FONTSIZE", (0, 0), (-1, -1), 8),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E5E7EB")),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F9FAFB")]),
                ]
            )
        )
        elements.append(seg_table)

    if recommendations:
        elements.append(Paragraph("Retention Strategy Recommendations", section_style))
        for rec in recommendations:
            elements.append(Paragraph(f"&bull; {rec}", body_style))

    elements.append(Spacer(1, 1 * cm))
    elements.append(
        Paragraph(
            "This report is generated automatically from the current dataset and trained "
            "models and is intended to support, not replace, human judgement.",
            footer_style,
        )
    )

    doc.build(elements)
    return buffer.getvalue()


def build_excel_report(sheets: Dict[str, pd.DataFrame]) -> bytes:
    """Build a multi-sheet Excel workbook from a dict of {sheet_name: df}.

    Used by the Reports Center page to export the model comparison table,
    segment summary, and high-risk customer list as a single .xlsx download.
    """
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        for sheet_name, df in sheets.items():
            safe_name = sheet_name[:31]  # Excel sheet name length limit
            df.to_excel(writer, sheet_name=safe_name, index=False)
            worksheet = writer.sheets[safe_name]
            for i, col in enumerate(df.columns):
                width = max(12, min(40, int(df[col].astype(str).str.len().mean() + 4) if len(df) else 12))
                worksheet.set_column(i, i, width)
    return buffer.getvalue()
