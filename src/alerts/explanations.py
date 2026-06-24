"""
Alert explanation builder.

Generates plain-English reason strings for each flagged partner so
reviewers know exactly what to investigate.
"""
import pandas as pd


def build_flag_reason(row: pd.Series,
                      iqr_multiplier: float = 2.0,
                      suspicious_reversal_ratio: float = 5.0,
                      high_balance_threshold: float = 100_000.0,
                      high_ros_score: float = 0.7) -> str:
    """Build a human-readable explanation for a single flagged partner row."""
    reasons = []
    peer = row.get("PeerGroup", row.get("Segment", "peer group"))

    # IQR-triggered features
    triggered = row.get("iqr_triggered_cols", "")
    if triggered:
        for col in triggered.split(", "):
            col = col.strip()
            if not col:
                continue
            score_col = f"{col}_iqr_score"
            score = row.get(score_col)
            value = row.get(col)

            if value is None or score is None:
                continue

            if col == "ReversalRatio":
                reasons.append(
                    f"ReversalRatio {value:.1f}x is {score:.1f} IQRs above {peer} peers"
                )
            elif col in ("TotalAmount", "MaxSingleAmount", "Balance", "TotalDebit"):
                reasons.append(
                    f"{col} ${value:,.0f} is {score:.1f} IQRs above {peer} peers"
                )
            elif col == "AvgROSScore":
                reasons.append(
                    f"Average transaction amount is suspiciously close to authority limits "
                    f"(ROS score {value:.2f}, {score:.1f} IQRs above {peer} peers)"
                )
            elif col == "RoundTransactionRate":
                reasons.append(
                    f"{value*100:.0f}% of transactions are round numbers "
                    f"({score:.1f} IQRs above {peer} peers)"
                )
            elif col == "NettedPairsCount":
                reasons.append(
                    f"{int(value):,} AC/ZC cancellation pairs detected "
                    f"({score:.1f} IQRs above {peer} peers)"
                )
            else:
                reasons.append(
                    f"{col} ({value:.1f}) is {score:.1f} IQRs above {peer} peers"
                )

    # Domain rule: reversal ratio threshold
    reversal = row.get("ReversalRatio", 0)
    if reversal > suspicious_reversal_ratio and "ReversalRatio" not in triggered:
        reasons.append(
            f"ReversalRatio {reversal:.1f}x exceeds {suspicious_reversal_ratio}x threshold"
        )

    # Domain rule: high ROS score
    ros = row.get("AvgROSScore", 0)
    if ros > high_ros_score and "AvgROSScore" not in triggered:
        reasons.append(
            f"Avg transaction amount near authority limit (ROS score {ros:.2f})"
        )

    # Domain rule: high balance
    balance = row.get("Balance", 0)
    if abs(balance) > high_balance_threshold:
        reasons.append(f"Balance ${balance:,.0f} exceeds ${high_balance_threshold:,.0f} threshold")

    # Isolation Forest only
    if not reasons and row.get("isolation_forest_flag", False):
        reasons.append(
            f"Flagged by Isolation Forest within {peer} peer group "
            f"(risk score {row.get('risk_score', 0):.2f}); "
            "no single metric exceeded IQR threshold -- review full profile"
        )

    return "; ".join(reasons) if reasons else ""


def build_alert_column(df: pd.DataFrame,
                        iqr_multiplier: float = 2.0,
                        suspicious_reversal_ratio: float = 5.0,
                        high_balance_threshold: float = 100_000.0,
                        high_ros_score: float = 0.7) -> pd.DataFrame:
    """Add flag_reason column to flagged partners."""
    df = df.copy()
    is_flagged = (
        df.get("isolation_forest_flag", False) | df.get("iqr_flag", False)
    )
    df["flag_reason"] = ""
    df.loc[is_flagged, "flag_reason"] = df[is_flagged].apply(
        lambda row: build_flag_reason(
            row,
            iqr_multiplier=iqr_multiplier,
            suspicious_reversal_ratio=suspicious_reversal_ratio,
            high_balance_threshold=high_balance_threshold,
            high_ros_score=high_ros_score,
        ),
        axis=1,
    )
    return df
