"""PDF export utilities for availability statistics."""

from __future__ import annotations

import base64
from dataclasses import dataclass
from datetime import datetime
from html import escape
from io import BytesIO
from typing import Any, Dict, Iterable, List

import pandas as pd
import plotly.express as px
import plotly.io as pio


try:  # pragma: no cover - optional dependency handled at runtime
    from weasyprint import HTML, CSS
except Exception as exc:  # pragma: no cover - fallback for missing library
    raise ImportError(
        "Le module 'weasyprint' est requis pour l'export PDF. Installez-le avec 'pip install weasyprint'."
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
        agg["percentage"] = agg["duration"].apply(lambda val: _format_number((val / total * 100) if total else 0))

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

        encoded = base64.b64encode(image_bytes).decode("utf-8")
        rows = [
            {
                "cause": escape(str(row.cause)),
                "minutes": escape(str(int(row.minutes))),
                "percentage": escape(f"{row.percentage.replace('.', ',')} %"),
            }
            for row in agg.itertuples()
        ]

        monthly_records.append(
            {
                "label": month.strftime("%Y-%m"),
                "image": encoded,
                "rows": rows,
            }
        )

    monthly_records.sort(key=lambda item: item["label"])
    return monthly_records


def _render_site_block(report: SiteReport, start_dt: datetime, end_dt: datetime) -> str:
    cards = _prepare_metrics_cards(report.metrics)
    summary_rows = _prepare_summary_rows(report.summary_df)
    equipment_rows = _prepare_equipment_rows(report.equipment_summary)
    monthly = _build_monthly_causes(report.raw_blocks, start_dt, end_dt)

    parts = [
        "<section class='site-block'>",
        f"  <h2>{escape(report.site_label)} ({escape(report.site)})</h2>",
        "  <div class='cards'>",
    ]

    for card in cards:
        caption_html = f"<div class='card-caption'>{escape(card['caption'])}</div>" if card["caption"] else ""
        parts.append(
            "    <div class='card'>"
            f"      <div class='card-label'>{escape(card['label'])}</div>"
            f"      <div class='card-value'>{escape(card['value'])}</div>"
            f"      {caption_html}"
            "    </div>"
        )

    parts.append("  </div>")

    if summary_rows:
        parts.append("  <h3>Conditions critiques</h3>")
        parts.append("  <table class='data-table'>")
        parts.append("    <thead><tr><th>Condition</th><th>Durée</th><th>Temps analysé</th></tr></thead>")
        parts.append("    <tbody>")
        for row in summary_rows:
            parts.append(
                "      <tr>"
                f"        <td>{row.get('Condition', '')}</td>"
                f"        <td>{row.get('Durée', '')}</td>"
                f"        <td>{row.get('Temps analysé', '')}</td>"
                "      </tr>"
            )
        parts.append("    </tbody>")
        parts.append("  </table>")

    if equipment_rows:
        parts.append("  <h3>Indicateurs clés par équipement</h3>")
        parts.append("  <table class='data-table'>")
        parts.append(
            "    <thead>"
            "      <tr>"
            "        <th>Équipement</th>"
            "        <th>Disponibilité brute (%)</th>"
            "        <th>Disponibilité avec exclusions (%)</th>"
            "        <th>Durée totale</th>"
            "        <th>Temps disponible</th>"
            "        <th>Temps indisponible</th>"
            "        <th>Jours avec des données</th>"
            "      </tr>"
            "    </thead>"
        )
        parts.append("    <tbody>")
        for row in equipment_rows:
            parts.append(
                "      <tr>"
                f"        <td>{row.get('Équipement', '')}</td>"
                f"        <td>{row.get('Disponibilité Brute (%)', '')}</td>"
                f"        <td>{row.get('Disponibilité Avec Exclusions (%)', '')}</td>"
                f"        <td>{row.get('Durée Totale', '')}</td>"
                f"        <td>{row.get('Temps Disponible', '')}</td>"
                f"        <td>{row.get('Temps Indisponible', '')}</td>"
                f"        <td>{row.get('Jours avec des données', '')}</td>"
                "      </tr>"
            )
        parts.append("    </tbody>")
        parts.append("  </table>")

    if monthly:
        parts.append("  <h3>Causes d'indisponibilité par mois</h3>")
        for record in monthly:
            parts.append("  <div class='month-block'>")
            parts.append(f"    <div class='month-label'>{escape(record['label'])}</div>")
            if record["image"]:
                parts.append(
                    "    <div class='month-chart'>"
                    f"      <img src='data:image/png;base64,{record['image']}' alt='Pie {escape(record['label'])}' />"
                    "    </div>"
                )
            parts.append("    <table class='data-table compact'>")
            parts.append("      <thead><tr><th>Cause</th><th>Minutes</th><th>%</th></tr></thead>")
            parts.append("      <tbody>")
            for row in record["rows"]:
                parts.append(
                    "        <tr>"
                    f"          <td>{row['cause']}</td>"
                    f"          <td>{row['minutes']}</td>"
                    f"          <td>{row['percentage']}</td>"
                    "        </tr>"
                )
            parts.append("      </tbody>")
            parts.append("    </table>")
            parts.append("  </div>")

    parts.append("</section>")
    return "\n".join(parts)


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

    sections = [
        _render_site_block(report, start_dt, end_dt)
        for report in reports_list
    ]

    html = "\n".join(
        [
            "<html>",
            "<head>",
            "  <meta charset='utf-8' />",
            "  <style>",
            "    @page { size: A4; margin: 1.5cm; }",
            "    body { font-family: 'DejaVu Sans', 'Helvetica', Arial, sans-serif; color: #1f2933; font-size: 12px; }",
            "    h1 { text-transform: uppercase; font-size: 20px; letter-spacing: 1px; margin: 0 0 16px 0; }",
            "    h2 { font-size: 16px; color: #111827; margin-bottom: 12px; }",
            "    h3 { font-size: 14px; color: #1f77b4; margin: 18px 0 8px; text-transform: uppercase; letter-spacing: 0.05em; }",
            "    .meta { display: flex; gap: 16px; margin-bottom: 20px; }",
            "    .site-block { page-break-after: always; }",
            "    .site-block:last-of-type { page-break-after: auto; }",
            "    .cards { display: flex; gap: 12px; margin-bottom: 16px; }",
            "    .card { background: #f0f2f6; padding: 12px 14px; border-radius: 10px; flex: 1; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }",
            "    .card-label { font-size: 11px; color: #1f77b4; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 6px; }",
            "    .card-value { font-size: 18px; font-weight: 600; color: #0b1f33; }",
            "    .card-caption { margin-top: 4px; font-size: 10px; color: #6b7280; }",
            "    .data-table { width: 100%; border-collapse: collapse; margin-bottom: 16px; }",
            "    .data-table thead th { background: #1f77b4; color: #ffffff; font-size: 11px; padding: 6px 8px; text-align: left; }",
            "    .data-table tbody td { border-bottom: 1px solid #d1d5db; padding: 6px 8px; font-size: 11px; }",
            "    .data-table.compact tbody td { font-size: 10px; }",
            "    .month-block { display: flex; gap: 16px; margin-bottom: 18px; align-items: flex-start; }",
            "    .month-label { font-size: 12px; font-weight: 600; color: #0b1f33; margin-bottom: 8px; text-transform: uppercase; letter-spacing: 0.08em; }",
            "    .month-chart { flex: 1; min-width: 320px; }",
            "    .month-chart img { width: 100%; border-radius: 8px; border: 1px solid #d1d5db; }",
          "  </style>",
            "</head>",
            "<body>",
            f"  <h1>{escape(title)}</h1>",
            f"  <div class='meta'><div>Période : {escape(start_label)} → {escape(end_label)}</div></div>",
            *sections,
            "</body>",
            "</html>",
        ]
    )

    pdf_io = BytesIO()
    HTML(string=html).write_pdf(target=pdf_io, stylesheets=[CSS(string="")])
    return pdf_io.getvalue()

