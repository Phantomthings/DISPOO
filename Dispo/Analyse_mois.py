from __future__ import annotations
import os
import pandas as pd
from datetime import datetime
from sqlalchemy import create_engine, text

MYSQL_HOST = os.getenv("MYSQL_HOST", "141.94.31.144")
MYSQL_PORT = int(os.getenv("MYSQL_PORT", 3306))
MYSQL_USER = os.getenv("MYSQL_USER", "AdminNidec")
MYSQL_PW   = os.getenv("MYSQL_PASSWORD", "u6Ehe987XBSXxa4")
MYSQL_DB   = os.getenv("MYSQL_DB", "indicator")

ANNOTATION_TYPE_EXCLUSION = "exclusion"
ANNOTATION_TYPE_MISSING_AVAILABLE = "missing_excl_available"
ANNOTATION_TYPE_MISSING_UNAVAILABLE = "missing_excl_unavailable"


def missing_exclusion_case(alias: str) -> str:
    return f"""
        CASE
          WHEN {alias}.est_disponible = -1 THEN
            CASE
              WHEN EXISTS (
                SELECT 1 FROM indicator.dispo_annotations m
                WHERE m.actif = 1
                  AND m.type_annotation = '{ANNOTATION_TYPE_MISSING_UNAVAILABLE}'
                  AND m.site = {alias}.site
                  AND m.equipement_id = {alias}.equipement_id
                  AND NOT (m.date_fin <= {alias}.date_debut OR m.date_debut >= {alias}.date_fin)
              ) THEN 2
              WHEN EXISTS (
                SELECT 1 FROM indicator.dispo_annotations m
                WHERE m.actif = 1
                  AND m.type_annotation = '{ANNOTATION_TYPE_MISSING_AVAILABLE}'
                  AND m.site = {alias}.site
                  AND m.equipement_id = {alias}.equipement_id
                  AND NOT (m.date_fin <= {alias}.date_debut OR m.date_debut >= {alias}.date_fin)
              ) THEN 1
              ELSE 0
            END
          ELSE 0
        END
    """


def mysql_engine():
    return create_engine(
        f"mysql+mysqlconnector://{MYSQL_USER}:{MYSQL_PW}@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DB}"
    )

INSERT_PCT_MOIS = text("""
INSERT INTO indicator.dispo_pct_mois
(site, equipement_id, mois, pct_brut, pct_excl, total_minutes, processed_at)
VALUES (:site, :equip, :mois, :pct_brut, :pct_excl, :total_minutes, UTC_TIMESTAMP())
ON DUPLICATE KEY UPDATE
  pct_brut = VALUES(pct_brut),
  pct_excl = VALUES(pct_excl),
  total_minutes = VALUES(total_minutes),
  processed_at = UTC_TIMESTAMP()
""")

def calculate_availability(df: pd.DataFrame, include_exclusions: bool = False) -> dict:
    if df.empty:
        return {"total_minutes": 0, "available_minutes": 0, "pct_available": 0.0}
    missing_mode = (
        df["missing_exclusion_mode"]
        if "missing_exclusion_mode" in df.columns
        else pd.Series(0, index=df.index)
    )
    missing_as_available = (df["est_disponible"] == -1) & (missing_mode == 1)
    missing_as_unavailable = (df["est_disponible"] == -1) & (missing_mode == 2)

    base_available_mask = (df["est_disponible"] == 1) | missing_as_available

    if include_exclusions:
        available_mask = base_available_mask | (
            (df["est_disponible"] == 0) & (df["is_excluded"] == 1)
        )
        unavailable_mask = (
            ((df["est_disponible"] == 0) & (df["is_excluded"] == 0))
        ) | missing_as_unavailable
    else:
        available_mask = base_available_mask
        unavailable_mask = (df["est_disponible"] == 0) | missing_as_unavailable

    available = int(df.loc[available_mask, "duration_minutes"].sum())
    unavailable = int(df.loc[unavailable_mask, "duration_minutes"].sum())
    effective_total = available + unavailable
    pct_available = (available / effective_total * 100) if effective_total > 0 else 0.0
    return {
        "total_minutes": effective_total,
        "available_minutes": available,
        "pct_available": pct_available,
    }

def update_monthly():
    eng = mysql_engine()
    missing_case_ac = missing_exclusion_case("a")
    missing_case_bt = missing_exclusion_case("b")
    missing_case_bt2 = missing_exclusion_case("c")
    query = f"""
        SELECT site, equipement_id, date_debut, date_fin, est_disponible,
               TIMESTAMPDIFF(MINUTE, date_debut, date_fin) as duration_minutes,
               CASE
                 WHEN est_disponible <> 1 THEN CAST(EXISTS (
                   SELECT 1 FROM indicator.dispo_annotations a
                   WHERE a.actif = 1 AND a.type_annotation = '{ANNOTATION_TYPE_EXCLUSION}'
                     AND a.site = site AND a.equipement_id = equipement_id
                     AND NOT (a.date_fin <= date_debut OR a.date_debut >= date_fin)
                 ) AS UNSIGNED)
                 ELSE 0
               END AS is_excluded,
               {missing_case_ac} AS missing_exclusion_mode
        FROM indicator.dispo_blocs_ac AS a
        UNION ALL
        SELECT site, equipement_id, date_debut, date_fin, est_disponible,
               TIMESTAMPDIFF(MINUTE, date_debut, date_fin) as duration_minutes,
               CASE
                 WHEN est_disponible <> 1 THEN CAST(EXISTS (
                   SELECT 1 FROM indicator.dispo_annotations a
                   WHERE a.actif = 1 AND a.type_annotation = '{ANNOTATION_TYPE_EXCLUSION}'
                     AND a.site = site AND a.equipement_id = equipement_id
                     AND NOT (a.date_fin <= date_debut OR a.date_debut >= date_fin)
                 ) AS UNSIGNED)
                 ELSE 0
               END AS is_excluded,
               {missing_case_bt} AS missing_exclusion_mode
        FROM indicator.dispo_blocs_batt AS b
        UNION ALL
        SELECT site, equipement_id, date_debut, date_fin, est_disponible,
               TIMESTAMPDIFF(MINUTE, date_debut, date_fin) as duration_minutes,
               CASE
                 WHEN est_disponible <> 1 THEN CAST(EXISTS (
                   SELECT 1 FROM indicator.dispo_annotations a
                   WHERE a.actif = 1 AND a.type_annotation = '{ANNOTATION_TYPE_EXCLUSION}'
                     AND a.site = site AND a.equipement_id = equipement_id
                     AND NOT (a.date_fin <= date_debut OR a.date_debut >= date_fin)
                 ) AS UNSIGNED)
                 ELSE 0
               END AS is_excluded,
               {missing_case_bt2} AS missing_exclusion_mode
        FROM indicator.dispo_blocs_batt2 AS c
    """
    df = pd.read_sql(query, eng)

    if df.empty:
        print("⚠️ Pas de données disponibles")
        return

    df["date_debut"] = pd.to_datetime(df["date_debut"], utc=True)
    df["month"] = df["date_debut"].dt.to_period("M").dt.to_timestamp()

    with eng.begin() as conn:
        for (site, equip), group_site in df.groupby(["site", "equipement_id"]):
            for month, group in group_site.groupby("month"):
                stats_raw = calculate_availability(group, include_exclusions=False)
                stats_excl = calculate_availability(group, include_exclusions=True)

                conn.execute(INSERT_PCT_MOIS, {
                    "site": site,
                    "equip": equip,
                    "mois": month.to_pydatetime().date(),
                    "pct_brut": stats_raw["pct_available"],
                    "pct_excl": stats_excl["pct_available"],
                    "total_minutes": stats_raw["total_minutes"],
                })
    print("✅ Table dispo_pct_mois mise à jour avec succès !")

if __name__ == "__main__":
    update_monthly()
