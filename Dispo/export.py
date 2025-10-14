"""PDF export utilities for availability statistics."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from html import escape
from io import BytesIO
from typing import Any, Dict, Iterable, List

import pandas as pd
import plotly.express as px
import plotly.io as pio

try:  # pragma: no cover - optional dependency handled at runtime
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.platypus import (
        Image,
        PageBreak,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )
except Exception as exc:  # pragma: no cover - fallback for missing library
    raise ImportError(
        "Le module 'reportlab' est requis pour l'export PDF. Installez-le avec 'pip install reportlab'."
    ) from exc


MetricDict = Dict[str, Any]


@dataclass
class SiteReport:
    """Structure de données normalisée pour un site."""

    site: str
    site_label: str
    metrics: MetricDict
    summary_df: pd.DataFrame
    equipment_summary: pd.DataFrame
    raw_blocks: pd.DataFrame


def _format_minutes(total_minutes: int) -> str:
    """Copie locale du formateur utilisé dans l'app."""

    minutes = int(total_minutes or 0)
    days, remainder = divmod(minutes, 1440)
    hours, mins = divmod(remainder, 60)

    parts: List[str] = []
    if days:
        parts.append(f"{days} {'jour' if days == 1 else 'jours'}")
    if hours:
        parts.append(f"{hours} {'heure' if hours == 1 else 'heures'}")
    if mins or not parts:
        parts.append(f"{mins} {'minute' if mins == 1 else 'minutes'}")
    return ", ".join(parts)


def _ensure_timezone(ts: datetime) -> pd.Timestamp:
    """Retourne un timestamp en Europe/Paris."""

    timestamp = pd.Timestamp(ts)
    if timestamp.tz is None:
        return timestamp.tz_localize("Europe/Paris", nonexistent="shift_forward", ambiguous="NaT")
    return timestamp.tz_convert("Europe/Paris")


def _prepare_metrics_cards(metrics: MetricDict) -> List[Dict[str, str]]:
    availability = float(metrics.get("availability_pct", 0.0) or 0.0)
    downtime_minutes = int(metrics.get("downtime_minutes", 0) or 0)
    reference_minutes = int(metrics.get("reference_minutes", 0) or 0)
    coverage_pct = float(metrics.get("coverage_pct", 0.0) or 0.0)
    window_minutes = int(metrics.get("window_minutes", 0) or 0)

    cards = [
        {
            "label": "Disponibilité estimée",
            "value": f"{availability:.2f} %",
            "caption": "",
        },
        {
            "label": "Indisponibilité réelle",
            "value": _format_minutes(downtime_minutes),
            "caption": "",
        },
        {
            "label": "Temps analysé",
            "value": _format_minutes(reference_minutes),
            "caption": f"Couverture {_format_number(coverage_pct)} % / {_format_minutes(window_minutes)}",
        },
    ]
    return cards


def _format_number(value: float) -> str:
    return f"{value:.1f}".replace(".", ",")


def _prepare_summary_rows(summary_df: pd.DataFrame) -> List[Dict[str, str]]:
    if summary_df is None or summary_df.empty:
        return []

    display = summary_df.copy()
    if "Durée_Minutes" in display.columns:
        display["Durée"] = display["Durée_Minutes"].apply(lambda m: _format_minutes(int(m)))
    if "Temps_Analysé_Minutes" in display.columns:
        display["Temps analysé"] = display["Temps_Analysé_Minutes"].apply(lambda m: _format_minutes(int(m)))

    columns = [
        col
        for col in ["Condition", "Durée", "Temps analysé"]
        if col in display.columns
    ]
    if not columns:
        return []

    return [
        {col: escape(str(row.get(col, ""))) for col in columns}
        for _, row in display[columns].iterrows()
    ]


def _prepare_equipment_rows(equipment_df: pd.DataFrame) -> List[Dict[str, str]]:
    if equipment_df is None or equipment_df.empty:
        return []

    columns = [
        "Équipement",
        "Disponibilité Brute (%)",
        "Disponibilité Avec Exclusions (%)",
        "Durée Totale",
        "Temps Disponible",
        "Temps Indisponible",
        "Jours avec des données",
    ]

    rows: List[Dict[str, str]] = []
    for _, row in equipment_df.iterrows():
        data = {}
        for col in columns:
            if col in row.index:
                data[col] = escape(str(row[col]))
        rows.append(data)
    return rows


def _build_monthly_causes(raw_blocks: pd.DataFrame, start_dt: datetime, end_dt: datetime) -> List[Dict[str, Any]]:
    if raw_blocks is None or raw_blocks.empty:
        return []

    df = raw_blocks.copy()
    tz_start = _ensure_timezone(start_dt)
    tz_end = _ensure_timezone(end_dt)

    df["clip_start"] = df["date_debut"].clip(lower=tz_start)
    df["clip_end"] = df["date_fin"].clip(upper=tz_end)
    df = df.loc[df["clip_start"].notna() & df["clip_end"].notna()].copy()
    if df.empty:
        return []

    df["duration"] = (
        (df["clip_end"] - df["clip_start"]).dt.total_seconds() / 60
    ).clip(lower=0).fillna(0)

    df = df.loc[df["duration"] > 0].copy()
    if df.empty:
        return []

    df["month"] = df["clip_start"].dt.to_period("M").dt.to_timestamp()
    df["cause"] = df["cause"].fillna("Non spécifié")

    monthly_records: List[Dict[str, Any]] = []
    palette = px.colors.qualitative.Safe

    for month, group in df.groupby("month"):
        unavail = group.loc[group["est_disponible"] == 0]
        if unavail.empty:
            monthly_records.append(
                {
                    "label": month.strftime("%Y-%m"),
                    "image": None,
                    "rows": [
                        {
                            "cause": "Aucune indisponibilité",
                            "minutes": "0",
                            "percentage": "0,0 %",
                        }
                    ],
                }
            )
            continue

        agg = (
            unavail.groupby("cause")["duration"].sum().reset_index().sort_values("duration", ascending=False)
        )
        total = float(agg["duration"].sum())
        agg["minutes"] = agg["duration"].round(0).astype(int)
        agg["percentage"] = agg["duration"].apply(lambda val: (val / total * 100) if total else 0)

        fig = px.pie(
            agg,
            names="cause",
            values="duration",
            color="cause",
            color_discrete_map={c: palette[i % len(palette)] for i, c in enumerate(agg["cause"])},
            hole=0.45,
        )
        fig.update_layout(
            margin=dict(l=20, r=20, t=30, b=20),
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.2),
            height=360,
        )

        try:
            image_bytes = pio.to_image(fig, format="png", scale=2)
        except ValueError as exc:
            raise ImportError(
                "La génération d'images Plotly nécessite le package 'kaleido'. Installez-le avec 'pip install -U kaleido'."
            ) from exc

        rows = [
            {
                "cause": str(row.cause),
                "minutes": str(int(row.minutes)),
                "percentage": f"{_format_number(float(row.percentage))} %",
            }
            for row in agg.itertuples()
        ]

        monthly_records.append(
            {
                "label": month.strftime("%Y-%m"),
                "image": image_bytes,
                "rows": rows,
            }
        )

    monthly_records.sort(key=lambda item: item["label"])
    return monthly_records


def _build_styles() -> Dict[str, ParagraphStyle]:
    base_styles = getSampleStyleSheet()
    styles: Dict[str, ParagraphStyle] = {
        "title": ParagraphStyle(
            "CustomTitle",
            parent=base_styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=20,
            leading=24,
            spaceAfter=12,
            textColor=colors.HexColor("#0b1f33"),
        ),
        "meta": ParagraphStyle(
            "Meta",
            parent=base_styles["Normal"],
            fontSize=11,
            textColor=colors.HexColor("#1f2933"),
            spaceAfter=16,
        ),
        "site": ParagraphStyle(
            "Site",
            parent=base_styles["Heading2"],
            fontSize=16,
            leading=18,
            textColor=colors.HexColor("#111827"),
            spaceAfter=12,
        ),
        "heading": ParagraphStyle(
            "Heading",
            parent=base_styles["Heading3"],
            fontSize=13,
            leading=16,
            textColor=colors.HexColor("#1f77b4"),
            spaceBefore=12,
            spaceAfter=6,
        ),
        "month": ParagraphStyle(
            "Month",
            parent=base_styles["Normal"],
            fontSize=12,
            leading=14,
            textColor=colors.HexColor("#0b1f33"),
            spaceBefore=8,
            spaceAfter=6,
        ),
        "card_label": ParagraphStyle(
            "CardLabel",
            parent=base_styles["Normal"],
            fontSize=10,
            textColor=colors.HexColor("#1f77b4"),
            leading=12,
        ),
        "card_value": ParagraphStyle(
            "CardValue",
            parent=base_styles["Normal"],
            fontSize=16,
            leading=18,
            textColor=colors.HexColor("#0b1f33"),
        ),
        "card_caption": ParagraphStyle(
            "CardCaption",
            parent=base_styles["Normal"],
            fontSize=9,
            leading=11,
            textColor=colors.HexColor("#6b7280"),
        ),
    }
    return styles


def _build_cards_table(cards: List[Dict[str, str]], styles: Dict[str, ParagraphStyle]) -> List[Any]:
    if not cards:
        return []

    labels = [Paragraph(escape(card["label"]), styles["card_label"]) for card in cards]
    values = [Paragraph(escape(card["value"]), styles["card_value"]) for card in cards]
    captions = [
        Paragraph(escape(card["caption"]), styles["card_caption"]) if card.get("caption") else Spacer(0, 0)
        for card in cards
    ]

    table = Table([labels, values, captions], colWidths=[(A4[0] - 3 * cm) / len(cards)] * len(cards))
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f0f2f6")),
                ("BOX", (0, 0), (-1, -1), 0.25, colors.HexColor("#cbd5f5")),
                ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#f0f2f6")),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                ("TOPPADDING", (0, 0), (-1, 0), 8),
                ("BOTTOMPADDING", (0, 1), (-1, -1), 8),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        )
    )
    table.hAlign = "LEFT"
    return [table, Spacer(1, 12)]


def _build_table(
    headers: List[str],
    rows: List[Dict[str, str]],
    columns: List[str],
    column_widths: List[float],
) -> Table:
    data: List[List[str]] = [headers]
    for row in rows:
        data.append([escape(str(row.get(col, ""))) for col in columns])

    table = Table(data, colWidths=column_widths, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f77b4")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#ffffff"), colors.HexColor("#f9fafb")]),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#d1d5db")),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        )
    )
    table.hAlign = "LEFT"
    return table


def _build_monthly_section(
    monthly: List[Dict[str, Any]],
    styles: Dict[str, ParagraphStyle],
) -> List[Any]:
    if not monthly:
        return []

    elements: List[Any] = []
    elements.append(Paragraph("Causes d'indisponibilité par mois", styles["heading"]))

    for record in monthly:
        elements.append(Paragraph(escape(record["label"]).upper(), styles["month"]))

        table_data: List[List[str]] = [["Cause", "Minutes", "%"]]
        for row in record["rows"]:
            table_data.append([
                escape(row["cause"]),
                escape(row["minutes"]),
                escape(row["percentage"]),
            ])

        causes_table = Table(table_data, colWidths=[8 * cm, 3 * cm, 2.5 * cm], repeatRows=1)
        causes_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f77b4")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#ffffff"), colors.HexColor("#f9fafb")]),
                    ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#d1d5db")),
                    ("LEFTPADDING", (0, 0), (-1, -1), 6),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ]
            )
        )
        causes_table.hAlign = "LEFT"

        if record["image"]:
            image = Image(BytesIO(record["image"]))
            image._restrictSize(8 * cm, 8 * cm)
            image.hAlign = "LEFT"
            layout = Table(
                [[image, causes_table]],
                colWidths=[8 * cm, None],
            )
            layout.setStyle(
                TableStyle(
                    [
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("LEFTPADDING", (0, 0), (-1, -1), 0),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                        ("TOPPADDING", (0, 0), (-1, -1), 0),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
                    ]
                )
            )
            elements.append(layout)
        else:
            elements.append(causes_table)

        elements.append(Spacer(1, 10))

    return elements


def _build_site_flowables(
    report: SiteReport,
    start_dt: datetime,
    end_dt: datetime,
    styles: Dict[str, ParagraphStyle],
) -> List[Any]:
    cards = _prepare_metrics_cards(report.metrics)
    summary_rows = _prepare_summary_rows(report.summary_df)
    equipment_rows = _prepare_equipment_rows(report.equipment_summary)
    monthly = _build_monthly_causes(report.raw_blocks, start_dt, end_dt)

    elements: List[Any] = [Paragraph(f"{escape(report.site_label)} ({escape(report.site)})", styles["site"])]
    elements.extend(_build_cards_table(cards, styles))

    if summary_rows:
        table = _build_table(
            ["Condition", "Durée", "Temps analysé"],
            summary_rows,
            ["Condition", "Durée", "Temps analysé"],
            [7 * cm, 4 * cm, 4 * cm],
        )
        elements.append(Paragraph("Conditions critiques", styles["heading"]))
        elements.append(table)

    if equipment_rows:
        table = _build_table(
            [
                "Équipement",
                "Disponibilité brute (%)",
                "Disponibilité avec exclusions (%)",
                "Durée totale",
                "Temps disponible",
                "Temps indisponible",
                "Jours avec des données",
            ],
            equipment_rows,
            [
                "Équipement",
                "Disponibilité Brute (%)",
                "Disponibilité Avec Exclusions (%)",
                "Durée Totale",
                "Temps Disponible",
                "Temps Indisponible",
                "Jours avec des données",
            ],
            [4.2 * cm, 3.2 * cm, 3.2 * cm, 3 * cm, 3 * cm, 3 * cm, 3.2 * cm],
        )
        elements.append(Paragraph("Indicateurs clés par équipement", styles["heading"]))
        elements.append(table)

    elements.extend(_build_monthly_section(monthly, styles))
    return elements


def generate_statistics_pdf(
    reports: Iterable[SiteReport],
    start_dt: datetime,
    end_dt: datetime,
    title: str = "rapport mensuel de disponibilité",
) -> bytes:
    """Construit un PDF A4 avec les statistiques pour chaque site."""

    reports_list = list(reports)
    if not reports_list:
        raise ValueError("Aucun site à exporter")

    start_label = _ensure_timezone(start_dt).strftime("%d/%m/%Y")
    end_label = _ensure_timezone(end_dt).strftime("%d/%m/%Y")

    styles = _build_styles()
    doc_buffer = BytesIO()
    doc = SimpleDocTemplate(
        doc_buffer,
        pagesize=A4,
        leftMargin=1.5 * cm,
        rightMargin=1.5 * cm,
        topMargin=1.5 * cm,
        bottomMargin=1.5 * cm,
    )

    elements: List[Any] = []
    elements.append(Paragraph(escape(title).upper(), styles["title"]))
    elements.append(
        Paragraph(
            f"Période : {escape(start_label)} → {escape(end_label)}",
            styles["meta"],
        )
    )

    for idx, report in enumerate(reports_list):
        elements.extend(_build_site_flowables(report, start_dt, end_dt, styles))
        if idx < len(reports_list) - 1:
            elements.append(PageBreak())

    doc.build(elements)
    return doc_buffer.getvalue()

