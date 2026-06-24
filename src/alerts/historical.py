"""
Historical alert deduplication.

Prevents alert fatigue by suppressing re-alerts for partners that were
already flagged recently and are presumably under review.

Two levels of deduplication:
  1. Within a single run: keep only highest-confidence flag per partner
  2. Across runs: suppress partners already in the historical alerts CSV
     within the cooldown window

Mirrors the pattern in William LaMonica-Sleczkowski's alert/alerts.py
but adapted for the BusinessPartner entity.
"""
import os
import pandas as pd
from datetime import datetime, timedelta


HISTORICAL_ALERTS_PATH = "outputs/historical_alerts.csv"
COOLDOWN_DAYS = 30   # Don't re-alert same partner within this window


def deduplicate_within_run(df: pd.DataFrame,
                            partner_col: str = "BusinessPartner") -> pd.DataFrame:
    """
    Within a single pipeline run, keep only one alert row per partner.

    Priority:
      1. Flagged by BOTH methods (highest confidence)
      2. Flagged by IQR only
      3. Flagged by Isolation Forest only

    If a partner appears multiple times (shouldn't happen often), keep
    the one with the highest risk_score.
    """
    if df.empty:
        return df

    df = df.copy()

    # Confidence tier
    def confidence(row):
        if row.get("isolation_forest_flag") and row.get("iqr_flag"):
            return 3
        elif row.get("iqr_flag"):
            return 2
        elif row.get("isolation_forest_flag"):
            return 1
        return 0

    df["_confidence"] = df.apply(confidence, axis=1)

    deduped = (
        df.sort_values(["_confidence", "risk_score"], ascending=[False, False])
        .drop_duplicates(subset=[partner_col], keep="first")
        .drop(columns=["_confidence"])
        .reset_index(drop=True)
    )

    n_removed = len(df) - len(deduped)
    if n_removed > 0:
        print(f"Within-run dedup: removed {n_removed} duplicate partner entries")

    return deduped


def load_historical_alerts(path: str = HISTORICAL_ALERTS_PATH) -> pd.DataFrame:
    """
    Load the historical alerts CSV.

    Returns empty DataFrame with correct schema if file doesn't exist yet.
    """
    if not os.path.exists(path):
        return pd.DataFrame(columns=[
            "BusinessPartner", "Segment", "PeerGroup", "run_date",
            "risk_score", "isolation_forest_flag", "iqr_flag", "flag_reason"
        ])

    df = pd.read_csv(path, parse_dates=["run_date"])
    return df


def filter_recently_alerted(new_alerts: pd.DataFrame,
                              historical: pd.DataFrame,
                              partner_col: str = "BusinessPartner",
                              cooldown_days: int = COOLDOWN_DAYS) -> tuple:
    """
    Remove partners from new_alerts that were already flagged recently.

    Returns:
        (fresh_alerts, suppressed_alerts)
        fresh_alerts    -- partners not seen recently, safe to escalate
        suppressed_alerts -- partners suppressed due to cooldown
    """
    if historical.empty or new_alerts.empty:
        return new_alerts, pd.DataFrame()

    cutoff = pd.Timestamp.now() - timedelta(days=cooldown_days)
    recent_historical = historical[historical["run_date"] >= cutoff]

    recently_flagged = set(recent_historical[partner_col].unique())
    new_partners = set(new_alerts[partner_col].unique())

    suppressed_ids = new_partners & recently_flagged
    fresh_ids      = new_partners - recently_flagged

    fresh_alerts     = new_alerts[new_alerts[partner_col].isin(fresh_ids)].copy()
    suppressed_alerts = new_alerts[new_alerts[partner_col].isin(suppressed_ids)].copy()

    if len(suppressed_ids) > 0:
        print(f"Historical dedup: suppressed {len(suppressed_alerts):,} partners "
              f"already flagged within last {cooldown_days} days")

    return fresh_alerts, suppressed_alerts


def append_to_historical(new_alerts: pd.DataFrame,
                          path: str = HISTORICAL_ALERTS_PATH,
                          partner_col: str = "BusinessPartner") -> pd.DataFrame:
    """
    Append new alerts to the historical CSV with today's run date.

    Creates the file if it doesn't exist.
    Returns the full updated historical DataFrame.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    save_cols = [
        partner_col, "Segment", "PeerGroup", "risk_score",
        "isolation_forest_flag", "iqr_flag", "flag_reason",
        "TotalAmount", "ReversalRatio", "MaxSingleAmount",
    ]
    save_cols = [c for c in save_cols if c in new_alerts.columns]

    to_save = new_alerts[save_cols].copy()
    to_save["run_date"] = pd.Timestamp.now().normalize()

    existing = load_historical_alerts(path)
    updated  = pd.concat([existing, to_save], ignore_index=True)
    updated.to_csv(path, index=False)

    print(f"Historical alerts updated: {len(updated):,} total records in {path}")
    return updated


def run_full_deduplication(new_alerts: pd.DataFrame,
                            historical_path: str = HISTORICAL_ALERTS_PATH,
                            cooldown_days: int = COOLDOWN_DAYS) -> dict:
    """
    Full deduplication pipeline. Call this after scoring, before exporting.

    Steps:
      1. Deduplicate within current run
      2. Load historical alerts
      3. Filter out recently alerted partners
      4. Append fresh alerts to historical log

    Returns dict with keys:
      fresh      -- alerts to escalate (new this run)
      suppressed -- alerts suppressed due to cooldown
      historical -- full updated historical log
    """
    # Step 1: within-run dedup
    deduped = deduplicate_within_run(new_alerts)

    # Step 2: load history
    historical = load_historical_alerts(historical_path)

    # Step 3: filter
    fresh, suppressed = filter_recently_alerted(
        deduped, historical, cooldown_days=cooldown_days
    )

    # Step 4: append fresh to history
    updated_historical = append_to_historical(fresh, path=historical_path)

    print(f"\nAlert summary:")
    print(f"  Total flagged this run:  {len(deduped):,}")
    print(f"  Fresh alerts:            {len(fresh):,}  (escalate these)")
    print(f"  Suppressed (cooldown):   {len(suppressed):,}  (already under review)")

    return {
        "fresh":      fresh,
        "suppressed": suppressed,
        "historical": updated_historical,
    }
