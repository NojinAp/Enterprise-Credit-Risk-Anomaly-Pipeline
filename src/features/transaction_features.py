"""
Feature engineering functions.

Each function takes a DataFrame and returns a DataFrame with new columns added.
All functions are pure (return copies, never mutate input) and independently
testable.

Domain features borrowed from William LaMonica-Sleczkowski's credit anomaly
project and adapted for the BusinessPartnerListSet SAP entity:
  - ROS (Range of Suspicion) score
  - Round transaction flag
  - Cancellation matching (AC/ZC pair netting)
"""
import pandas as pd
import numpy as np


# Authority limits used for ROS scoring
# Based on empirical analysis: 99th pct = $451, 99.5th pct = $1,083

AUTHORITY_LIMITS = [100.0, 500.0, 1000.0]
ROS_BANDWIDTH = 0.15   # Gaussian kernel bandwidth as fraction of limit


def add_reversal_ratio(df: pd.DataFrame,
                       doc_type_col: str = "CADocumentType",
                       ac_type: str = "AC",
                       zc_type: str = "ZC") -> pd.DataFrame:
    """
    Flag each row as an AC credit or ZC reversal.
    Aggregated per partner to get ReversalRatio = ZCCount / (ACCount + 1).
    """
    df = df.copy()
    # AC = credit document, ZC = reversal of that credit
    df["ACDocCount"] = (df[doc_type_col] == ac_type).astype(int)
    df["ZCDocCount"] = (df[doc_type_col] == zc_type).astype(int)
    return df


def add_high_value_flags(df: pd.DataFrame,
                         amount_col: str = "Amount",
                         high_threshold: float = 10_000.0,
                         extreme_threshold: float = 100_000.0) -> pd.DataFrame:
    """Binary flags for transactions above dollar thresholds."""
    df = df.copy()
    df["IsHighValue"]    = (df[amount_col] > high_threshold).astype(int)
    df["IsExtremeValue"] = (df[amount_col] > extreme_threshold).astype(int)
    return df


def add_ros_score(df: pd.DataFrame,
                  amount_col: str = "Amount",
                  authority_limits: list = None,
                  bandwidth: float = ROS_BANDWIDTH) -> pd.DataFrame:
    """
    Range of Suspicion (ROS) score per transaction.

    Assigns a score based on how close the transaction amount is to known
    authority limits ($100, $500, $1000). Credits just under a limit are
    suspicious; they suggest someone is deliberately staying below the
    threshold that triggers review.

    Uses a Gaussian kernel: score = max over all limits of
        exp( -0.5 * ((amount - limit) / (bandwidth * limit))^2 )

    Score of 1.0 means the amount is exactly at a limit.
    Score near 0 means far from all limits.
    """
    if authority_limits is None:
        authority_limits = AUTHORITY_LIMITS

    df = df.copy()
    amounts = df[amount_col].values
    scores = np.zeros(len(amounts))

    for limit in authority_limits:
        sigma = bandwidth * limit
        kernel = np.exp(-0.5 * ((amounts - limit) / sigma) ** 2)
        scores = np.maximum(scores, kernel)

    df["ROSScore"] = scores
    return df


def add_round_transaction_flag(df: pd.DataFrame,
                                amount_col: str = "Amount",
                                round_thresholds: list = None) -> pd.DataFrame:
    """
    Flag transactions that are suspiciously round numbers.

    Round amounts ($10, $50, $100) suggest the credit was arbitrary rather
    than calculated from actual loss/damage. We flag amounts divisible by
    each threshold.

    Adds columns: IsRound10, IsRound50, IsRound100, IsRoundAny
    """
    if round_thresholds is None:
        round_thresholds = [10, 50, 100]

    df = df.copy()
    amounts = df[amount_col]

    flag_cols = []
    for threshold in round_thresholds:
        col = f"IsRound{threshold}"
        df[col] = ((amounts % threshold == 0) & (amounts > 0)).astype(int)
        flag_cols.append(col)

    df["IsRoundAny"] = df[flag_cols].max(axis=1)
    return df


def net_cancellation_pairs(df: pd.DataFrame,
                            partner_col: str = "BusinessPartner",
                            amount_col: str = "Amount",
                            doc_type_col: str = "CADocumentType",
                            ac_type: str = "AC",
                            zc_type: str = "ZC") -> pd.DataFrame:
    """
    Net out AC/ZC cancellation pairs per business partner.

    When a credit (AC) is reversed by a cancellation (ZC) of the same amount
    for the same partner, both are noise; they cancel out and shouldn't
    inflate the partner's credit totals. This function marks them so they
    can be excluded from aggregate metrics.

    Strategy:
      1. Find exact amount matches between AC and ZC rows per partner
      2. Mark the matched rows as cancelled (IsNetted = 1)
      3. Downstream aggregation should use IsNetted=0 rows only

    Returns DataFrame with IsNetted column added.
    """
    df = df.copy()
    df["IsNetted"] = 0

    ac_rows = df[df[doc_type_col] == ac_type].copy()
    zc_rows = df[df[doc_type_col] == zc_type].copy()

    # Match on partner + amount
    ac_rows["_key"] = ac_rows[partner_col].astype(str) + "_" + ac_rows[amount_col].round(2).astype(str)
    zc_rows["_key"] = zc_rows[partner_col].astype(str) + "_" + zc_rows[amount_col].round(2).astype(str)

    matched_keys = set(ac_rows["_key"]) & set(zc_rows["_key"])

    ac_matched = ac_rows[ac_rows["_key"].isin(matched_keys)].index
    zc_matched = zc_rows[zc_rows["_key"].isin(matched_keys)].index

    df.loc[ac_matched, "IsNetted"] = 1
    df.loc[zc_matched, "IsNetted"] = 1

    n_netted = df["IsNetted"].sum()
    n_partners = df[df["IsNetted"] == 1][partner_col].nunique()

    print(f"Cancellation matching: {n_netted:,} rows netted out across {n_partners:,} partners")

    return df.drop(columns=["_key"], errors="ignore")


def add_segment_label(df: pd.DataFrame,
                      segment_col: str = "CAAccountDeterminationCode",
                      segment_labels: dict = None) -> pd.DataFrame:
    """Map raw SAP segment codes to human-readable vertical names."""
    if segment_labels is None:
        from config import SEGMENT_LABELS
        segment_labels = SEGMENT_LABELS

    df = df.copy()
    df["Segment"] = df[segment_col].map(segment_labels).fillna("Unknown")
    return df


def add_transaction_type_label(df: pd.DataFrame,
                                main_txn_col: str = "CAMainTransaction") -> pd.DataFrame:
    """
    Map CAMainTransaction codes to human-readable peer group labels.

    These codes define what kind of credit it is -- used as a secondary
    segmentation dimension so partners are scored against true peers.
    """
    txn_labels = {
        "R000": "Receivable",
        "0060": "Payment",
        "MG00": "Manual_Goodwill",
        "6000": "WriteOff",
        "0600": "Adjustment",
        "0610": "Reversal_Adjustment",
        "0040": "Interest",
        "CLP0": "Collection",
        "0110": "Dunning",
        "0020": "Installment",
    }
    df = df.copy()
    df["TransactionType"] = df[main_txn_col].map(txn_labels).fillna("Other")
    return df


def aggregate_to_partner_level(df: pd.DataFrame,
                                 use_netted: bool = True) -> pd.DataFrame:
    """
    Roll up transaction-level rows to one row per business partner.

    If use_netted=True (default), excludes rows marked as AC/ZC cancellation
    pairs from the credit totals, giving a cleaner picture of net activity.

    Adds domain features:
      - ReversalRatio
      - AmountVolatility
      - AvgROSScore / MaxROSScore
      - RoundTransactionRate
      - NettedPairsCount
    """
    if use_netted and "IsNetted" in df.columns:
        clean = df[df["IsNetted"] == 0].copy()
        netted_counts = df[df["IsNetted"] == 1].groupby("BusinessPartner").size().rename("NettedPairsCount")
    else:
        clean = df.copy()
        netted_counts = pd.Series(dtype=int, name="NettedPairsCount")

    agg = (
        clean.groupby("BusinessPartner")
        .agg(
            BusinessPartnerName=("BusinessPartnerName", "first"),
            Segment=("Segment", "first"),
            TransactionType=("TransactionType", "first"),

            # Volume
            TotalTransactions=("Amount", "count"),
            TotalAmount=("Amount", "sum"),
            AvgAmount=("Amount", "mean"),
            StdAmount=("Amount", "std"),
            MaxSingleAmount=("Amount", "max"),
            TotalItems=("NumberOfItems", "sum"),

            # Diversity
            UniqueDocTypes=("CADocumentType", "nunique"),
            UniqueMainTransactions=("CAMainTransaction", "nunique"),
            UniqueContracts=("ContractAccount", "nunique"),

            # Reversal counts
            ACDocCount=("ACDocCount", "sum"),
            ZCDocCount=("ZCDocCount", "sum"),

            # High value
            HighValueCount=("IsHighValue", "sum"),
            ExtremeValueCount=("IsExtremeValue", "sum"),

            # Balance & exposure
            Balance=("BalanceAmountInDisplayCurrency", "max"),
            TotalDebit=("DebitAmountInDisplayCrcy", "max"),
            TotalCredit=("CreditAmountInDisplayCrcy", "max"),

            # Overdue
            AvgDaysOverdue=("DaysOverdue", "mean"),
            MaxDaysOverdue=("DaysOverdue", "max"),

            # Domain features
            AvgROSScore=("ROSScore", "mean"),
            MaxROSScore=("ROSScore", "max"),
            RoundTransactionCount=("IsRoundAny", "sum"),
        )
        .reset_index()
        .fillna(0)
    )

    # Derived features
    agg["ReversalRatio"]        = agg["ZCDocCount"] / (agg["ACDocCount"] + 1)
    agg["AmountVolatility"]     = agg["StdAmount"] / (agg["AvgAmount"] + 1)
    agg["RoundTransactionRate"] = agg["RoundTransactionCount"] / (agg["TotalTransactions"] + 1)

    # Join netted counts
    if len(netted_counts) > 0:
        agg = agg.merge(netted_counts, on="BusinessPartner", how="left")
        agg["NettedPairsCount"] = agg["NettedPairsCount"].fillna(0)
    else:
        agg["NettedPairsCount"] = 0

    return agg


def add_peer_zscores(df: pd.DataFrame,
                     cols: list = None,
                     segment_col: str = "PeerGroup") -> pd.DataFrame:
    """
    Add z-score columns for key metrics within each peer group.

    Scoring within peer group means a partner is flagged as unusual
    relative to their true peers -- not the whole dataset.
    """
    if cols is None:
        cols = ["TotalTransactions", "MaxSingleAmount",
                "ReversalRatio", "Balance", "TotalDebit",
                "AvgROSScore", "RoundTransactionRate", "NettedPairsCount"]

    df = df.copy()

    if segment_col not in df.columns:
        if "Segment" in df.columns and "TransactionType" in df.columns:
            df[segment_col] = df["Segment"] + "_" + df["TransactionType"]
        elif "Segment" in df.columns:
            df[segment_col] = df["Segment"]
        else:
            df[segment_col] = "ALL"

    for col in cols:
        if col not in df.columns:
            continue
        group_mean = df.groupby(segment_col)[col].transform("mean")
        group_std  = df.groupby(segment_col)[col].transform("std").replace(0, 1)
        df[f"{col}_zscore"] = (df[col] - group_mean) / group_std

    return df
