from __future__ import annotations

from datetime import datetime, timezone
from io import BytesIO
from typing import Any

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

try:
    from .chatbot import build_explanation
except ImportError:
    from chatbot import build_explanation


NAVY = colors.HexColor("#102630")
TEAL = colors.HexColor("#35C9B3")
PALE = colors.HexColor("#EAF7F3")
MUTED = colors.HexColor("#66747B")
LINE = colors.HexColor("#DDE3E2")
RISK_COLORS = {
    "High": colors.HexColor("#D85A3A"),
    "Medium": colors.HexColor("#C88B18"),
    "Low": colors.HexColor("#16806E"),
}


def _footer(canvas, document) -> None:
    canvas.saveState()
    canvas.setStrokeColor(LINE)
    canvas.line(18 * mm, 14 * mm, 192 * mm, 14 * mm)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(MUTED)
    canvas.drawString(18 * mm, 9 * mm, "ChurnSignal - Local customer assessment")
    canvas.drawRightString(192 * mm, 9 * mm, f"Page {document.page}")
    canvas.restoreState()


def create_customer_report(customer: dict[str, Any], result: dict[str, Any]) -> bytes:
    explanation = build_explanation(customer, result)
    output = BytesIO()
    document = SimpleDocTemplate(
        output,
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=20 * mm,
        title="ChurnSignal Customer Assessment",
        author="ChurnSignal",
    )
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "ReportTitle", parent=styles["Title"], fontName="Helvetica-Bold",
        fontSize=24, leading=28, textColor=NAVY, alignment=TA_LEFT, spaceAfter=3 * mm,
    )
    eyebrow = ParagraphStyle(
        "Eyebrow", parent=styles["Normal"], fontName="Helvetica-Bold",
        fontSize=8, leading=10, textColor=colors.HexColor("#187666"),
        spaceAfter=2 * mm,
    )
    heading = ParagraphStyle(
        "Heading", parent=styles["Heading2"], fontName="Helvetica-Bold",
        fontSize=13, leading=16, textColor=NAVY, spaceBefore=5 * mm, spaceAfter=3 * mm,
    )
    body = ParagraphStyle(
        "Body", parent=styles["BodyText"], fontName="Helvetica",
        fontSize=9.5, leading=14, textColor=NAVY,
    )
    small = ParagraphStyle(
        "Small", parent=body, fontSize=8, leading=11, textColor=MUTED,
    )

    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    story = [
        Paragraph("RETENTION INTELLIGENCE", eyebrow),
        Paragraph("Customer Churn Assessment", title_style),
        Paragraph(f"Generated locally on {generated}", small),
        Spacer(1, 6 * mm),
    ]

    probability = result["churn_probability"]
    metric_data = [
        ["CHURN PROBABILITY", "RISK LEVEL", "MODEL DECISION"],
        [f"{probability:.1%}", explanation.risk_level, result["prediction_label"]],
    ]
    metric_table = Table(metric_data, colWidths=[58 * mm, 55 * mm, 61 * mm])
    metric_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), TEAL),
        ("TEXTCOLOR", (0, 1), (-1, 1), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (-1, 1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 7),
        ("FONTSIZE", (0, 1), (-1, 1), 17),
        ("TOPPADDING", (0, 0), (-1, 0), 10),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 4),
        ("TOPPADDING", (0, 1), (-1, 1), 5),
        ("BOTTOMPADDING", (0, 1), (-1, 1), 12),
        ("BOX", (0, 0), (-1, -1), 0.5, NAVY),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#29434C")),
    ]))
    story.append(metric_table)
    story.extend([
        Paragraph("EXECUTIVE DECISION", heading),
        Paragraph(explanation.summary, body),
        Spacer(1, 2 * mm),
    ])

    action_table = Table(
        [[Paragraph("NEXT BEST ACTION", eyebrow)], [Paragraph(explanation.recommended_action, body)]],
        colWidths=[174 * mm],
    )
    action_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), PALE),
        ("BOX", (0, 0), (-1, -1), 0.8, TEAL),
        ("LEFTPADDING", (0, 0), (-1, -1), 12),
        ("RIGHTPADDING", (0, 0), (-1, -1), 12),
        ("TOPPADDING", (0, 0), (-1, 0), 10),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 2),
        ("TOPPADDING", (0, 1), (-1, 1), 2),
        ("BOTTOMPADDING", (0, 1), (-1, 1), 11),
    ]))
    story.append(action_table)

    story.append(Paragraph("MARKETING SIGNALS", heading))
    signal_rows = []
    for index, signal in enumerate(explanation.profile_signals, 1):
        signal_rows.append([str(index), Paragraph(signal, body)])
    signal_table = Table(signal_rows, colWidths=[10 * mm, 164 * mm])
    signal_table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TEXTCOLOR", (0, 0), (0, -1), RISK_COLORS[explanation.risk_level]),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("LINEBELOW", (0, 0), (-1, -2), 0.5, LINE),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(signal_table)

    story.append(Paragraph("CUSTOMER PROFILE", heading))
    labels = {
        "gender": "Gender", "Senior_Citizen": "Senior citizen",
        "Is_Married": "Married", "Dependents": "Dependents", "tenure": "Tenure",
        "Phone_Service": "Phone service", "Dual": "Multiple lines",
        "Internet_Service": "Internet service", "Online_Security": "Online security",
        "Online_Backup": "Online backup", "Device_Protection": "Device protection",
        "Tech_Support": "Technical support", "Streaming_TV": "Streaming TV",
        "Streaming_Movies": "Streaming movies", "Contract": "Contract",
        "Paperless_Billing": "Paperless billing", "Payment_Method": "Payment method",
        "Monthly_Charges": "Monthly charges", "Total_Charges": "Total charges",
    }
    profile_rows = []
    items = list(labels.items())
    for position in range(0, len(items), 2):
        left_key, left_label = items[position]
        row = [Paragraph(left_label, small), Paragraph(str(customer[left_key]), body)]
        if position + 1 < len(items):
            right_key, right_label = items[position + 1]
            row += [Paragraph(right_label, small), Paragraph(str(customer[right_key]), body)]
        else:
            row += ["", ""]
        profile_rows.append(row)
    profile_table = Table(profile_rows, colWidths=[29 * mm, 54 * mm, 32 * mm, 59 * mm])
    profile_table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F8FAF9")),
        ("GRID", (0, 0), (-1, -1), 0.4, LINE),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(KeepTogether(profile_table))
    story.extend([
        Spacer(1, 2 * mm),
        Paragraph(
            "Method note: This report uses the saved Logistic Regression and CatBoost "
            "ensemble. The signals are profile-based marketing guidance, not proof of causation.",
            small,
        ),
    ])

    document.build(story, onFirstPage=_footer, onLaterPages=_footer)
    return output.getvalue()


def create_bulk_report(
    findings: list[dict[str, Any]], summary: dict[str, Any], source_name: str
) -> bytes:
    """Create a portfolio summary followed by a finding for every uploaded row."""
    output = BytesIO()
    document = SimpleDocTemplate(
        output,
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=17 * mm,
        bottomMargin=20 * mm,
        title="ChurnSignal Portfolio Findings",
        author="ChurnSignal",
    )
    styles = getSampleStyleSheet()
    title = ParagraphStyle(
        "BulkTitle", parent=styles["Title"], fontName="Helvetica-Bold",
        fontSize=23, leading=27, textColor=NAVY, alignment=TA_LEFT, spaceAfter=3 * mm,
    )
    eyebrow = ParagraphStyle(
        "BulkEyebrow", parent=styles["Normal"], fontName="Helvetica-Bold",
        fontSize=8, leading=10, textColor=colors.HexColor("#187666"),
        spaceAfter=2 * mm,
    )
    heading = ParagraphStyle(
        "BulkHeading", parent=styles["Heading2"], fontName="Helvetica-Bold",
        fontSize=14, leading=17, textColor=NAVY, spaceBefore=4 * mm, spaceAfter=3 * mm,
    )
    body = ParagraphStyle(
        "BulkBody", parent=styles["BodyText"], fontName="Helvetica",
        fontSize=9, leading=12.5, textColor=NAVY,
    )
    small = ParagraphStyle(
        "BulkSmall", parent=body, fontSize=8, leading=10.5, textColor=MUTED,
    )
    card_title = ParagraphStyle(
        "CardTitle", parent=body, fontName="Helvetica-Bold", fontSize=10.5,
        leading=13, textColor=colors.white,
    )

    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    story = [
        Paragraph("RETENTION PORTFOLIO", eyebrow),
        Paragraph("Customer Churn Findings", title),
        Paragraph(f"Source: {source_name} | Generated locally on {generated}", small),
        Spacer(1, 7 * mm),
    ]

    metrics = [
        ["CUSTOMERS", "SCORED", "PREDICTED CHURN", "CHURN RATE"],
        [
            str(summary["total_rows"]),
            str(summary["scored_rows"]),
            str(summary["predicted_churn"]),
            f'{summary["churn_rate"]:.1%}',
        ],
    ]
    metric_table = Table(metrics, colWidths=[43.5 * mm] * 4)
    metric_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), TEAL),
        ("TEXTCOLOR", (0, 1), (-1, 1), colors.white),
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 7),
        ("FONTSIZE", (0, 1), (-1, 1), 18),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("TOPPADDING", (0, 0), (-1, 0), 10),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 4),
        ("TOPPADDING", (0, 1), (-1, 1), 4),
        ("BOTTOMPADDING", (0, 1), (-1, 1), 12),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#29434C")),
    ]))
    story.append(metric_table)
    story.extend([
        Paragraph("EXECUTIVE INTERPRETATION", heading),
        Paragraph(
            f'{summary["predicted_churn"]} of {summary["scored_rows"]} scored customers '
            f'are above the intervention threshold. Start with High-risk customers, then '
            f'Medium-risk customers. {summary["invalid_rows"]} row(s) require correction.',
            body,
        ),
        Spacer(1, 4 * mm),
        Paragraph(
            "The accompanying workbook is a copy of the uploaded data with Churn and "
            "Churn_Percentage added to each customer row.",
            body,
        ),
        PageBreak(),
        Paragraph("ROW-BY-ROW FINDINGS", eyebrow),
        Paragraph("Customer Findings and Actions", title),
    ])

    for finding in findings:
        if finding["error"]:
            risk_level = "Invalid"
            probability_text = "Not scored"
            decision = "Needs correction"
            signal_text = str(finding["error"])
            action = "Correct the source row and upload the file again."
            accent = colors.HexColor("#9A351C")
        else:
            result = finding["result"]
            explanation = build_explanation(finding["customer"], result)
            risk_level = explanation.risk_level
            probability_text = f'{result["churn_probability"]:.1%}'
            decision = result["prediction_label"]
            signal_text = "; ".join(explanation.profile_signals)
            action = explanation.recommended_action
            accent = RISK_COLORS[risk_level]

        header_text = (
            f'Row {finding["row_number"]} | {finding["customer_id"]} | '
            f'{risk_level} | {probability_text} | {decision}'
        )
        card = Table(
            [
                [Paragraph(header_text, card_title)],
                [Paragraph(f"<b>Finding:</b> {signal_text}", body)],
                [Paragraph(f"<b>Recommended action:</b> {action}", body)],
            ],
            colWidths=[174 * mm],
        )
        card.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (0, 0), accent),
            ("BACKGROUND", (0, 1), (0, -1), colors.HexColor("#F8FAF9")),
            ("BOX", (0, 0), (-1, -1), 0.7, LINE),
            ("LEFTPADDING", (0, 0), (-1, -1), 9),
            ("RIGHTPADDING", (0, 0), (-1, -1), 9),
            ("TOPPADDING", (0, 0), (-1, 0), 7),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 7),
            ("TOPPADDING", (0, 1), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 1), (-1, -1), 5),
        ]))
        story.extend([KeepTogether(card), Spacer(1, 3 * mm)])

    story.extend([
        Spacer(1, 2 * mm),
        Paragraph(
            "Method note: Findings use the saved Logistic Regression and CatBoost ensemble. "
            "Profile signals guide marketing outreach and are not proof of causation.",
            small,
        ),
    ])
    document.build(story, onFirstPage=_footer, onLaterPages=_footer)
    return output.getvalue()
