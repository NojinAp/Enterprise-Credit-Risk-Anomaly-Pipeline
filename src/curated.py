"""
Gold layer builder - curated business partner anomaly scores.

Full pipeline:
  1. Load silver parquet
  2. Domain feature engineering (features/transaction_features.py)
  3. Cancellation matching -- net out AC/ZC pairs
  4. Granular peer group assignment (Segment x TransactionType)
  5. IQR scoring per peer group
  6. Isolation Forest per peer group
  7. Alert explanations
  8. Alert deduplication (within-run + historical)
  9. Write gold parquet + watchlist CSV
"""
import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from spark_session import get_spark
from pyspark.sql import functions as F
from config import DEFAULT_CONFIG, AUTHORITY_LIMITS, ROS_BANDWIDTH

from features.transaction_features import (
    add_reversal_ratio,
    add_high_value_flags,
    add_ros_score,
    add_round_transaction_flag,
    net_cancellation_pairs,
    add_segment_label,
    add_transaction_type_label,
    aggregate_to_partner_level,
    add_peer_zscores,
)
from detection.iqr_scoring import score_features_by_segment
from detection.isolation_forest import run_isolation_forest_by_segment
from alerts.explanations import build_alert_column
from alerts.historical import run_full_deduplication


def build_curated(config=DEFAULT_CONFIG):
    cfg = config
    spark = get_spark()

    # Load silver layer parquet
    stg = spark.read.parquet(cfg.staging.staging_path)
    print(f"Silver rows loaded: {stg.count():,}")

    stg = stg.withColumn(
        "DaysOverdue",
        F.datediff(F.current_date(), F.col("NetDueDate"))
    )

    df = stg.toPandas()
    spark.stop()
    print("Converted to pandas for feature engineering.")

    # Feature engineering
    df = add_reversal_ratio(df, doc_type_col="CADocumentType")
    df = add_high_value_flags(
        df,
        amount_col="Amount",
        high_threshold=cfg.thresholds.high_value_threshold,
        extreme_threshold=cfg.thresholds.extreme_value_threshold,
    )
    df = add_ros_score(
        df,
        amount_col="Amount",
        authority_limits=AUTHORITY_LIMITS,
        bandwidth=ROS_BANDWIDTH,
    )
    df = add_round_transaction_flag(df, amount_col="Amount")
    df = add_segment_label(df, segment_col="CAAccountDeterminationCode")
    df = add_transaction_type_label(df, main_txn_col="CAMainTransaction")

    # Cancellation matching
    df = net_cancellation_pairs(df, partner_col="BusinessPartner", amount_col="Amount")

    # Aggregate to partner level
    partners = aggregate_to_partner_level(df, use_netted=True)
    print(f"Unique partners: {len(partners):,}")

    # Granular peer group: Segment x TransactionType
    # A partner is compared to others in the same segment AND transaction type
    partners["PeerGroup"] = partners["Segment"] + "_" + partners["TransactionType"]
    peer_counts = partners["PeerGroup"].value_counts()
    print(f"\nPeer groups ({len(peer_counts)} total):")
    print(peer_counts.to_string())
    print()

    # Peer z-scores within peer group 
    partners = add_peer_zscores(partners, segment_col="PeerGroup")

    # IQR scoring per peer group 
    partners = score_features_by_segment(
        partners,
        feature_cols=cfg.features.numeric_features,
        segment_col="PeerGroup",
        iqr_multiplier=cfg.thresholds.iqr_multiplier,
    )
    iqr_flagged = partners["iqr_flag"].sum()
    print(f"IQR flagged: {iqr_flagged:,} partners")

    # Isolation Forest per peer group
    partners = run_isolation_forest_by_segment(
        partners,
        feature_cols=cfg.features.numeric_features,
        segment_col="PeerGroup",
        contamination=cfg.thresholds.isolation_forest_contamination,
        n_estimators=cfg.thresholds.isolation_forest_n_estimators,
        random_state=cfg.thresholds.isolation_forest_random_state,
    )
    if_flagged = partners["isolation_forest_flag"].sum()
    print(f"Isolation Forest flagged: {if_flagged:,} partners")

    both = (partners["isolation_forest_flag"] & partners["iqr_flag"]).sum()
    print(f"Flagged by BOTH methods:  {both:,} partners (highest confidence)\n")

    # Alert explanations
    partners = build_alert_column(
        partners,
        iqr_multiplier=cfg.thresholds.iqr_multiplier,
        suspicious_reversal_ratio=cfg.thresholds.suspicious_reversal_ratio,
        high_balance_threshold=cfg.thresholds.high_balance_threshold,
        high_ros_score=cfg.thresholds.high_ros_score,
    )

    # Write gold parquet
    partners.to_parquet(cfg.output.curated_path + ".parquet", index=False)
    print(f"Gold layer written.")

    # Alert deduplication + historical tracking
    flagged = partners[
        partners["isolation_forest_flag"] | partners["iqr_flag"]
    ].copy()

    alert_results = run_full_deduplication(
        flagged,
        historical_path=cfg.output.historical_alerts_path,
        cooldown_days=cfg.thresholds.alert_cooldown_days,
    )
    fresh = alert_results["fresh"]

    # Write watchlist CSV (fresh alerts only)
    os.makedirs(os.path.dirname(cfg.output.watchlist_path), exist_ok=True)
    watchlist_cols = [c for c in cfg.output.watchlist_columns if c in fresh.columns]
    watchlist = fresh[watchlist_cols].sort_values("risk_score", ascending=False)
    watchlist.to_csv(cfg.output.watchlist_path, index=False)
    print(f"Watchlist written: {len(watchlist):,} fresh alerts to {cfg.output.watchlist_path}")


if __name__ == "__main__":
    build_curated()
