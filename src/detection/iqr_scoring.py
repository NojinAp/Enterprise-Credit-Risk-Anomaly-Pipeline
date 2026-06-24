"""
IQR-based outlier scoring.

For each metric, compute how many IQRs above Q3 a value sits. Scores above the threshold are flagged.

Scoring runs per-segment so partners are compared to their own peer group.
"""
import pandas as pd
import numpy as np
from typing import List


def iqr_score(series: pd.Series) -> pd.Series:
    """
    For each value in a Series, return how many IQRs above Q3 it is.

    Positive score = above Q3 by that many IQRs (suspicious).
    Negative or zero score = normal or below average.

    Formula: (value - Q3) / IQR
    """
    q1  = series.quantile(0.25)
    q3  = series.quantile(0.75)
    iqr = q3 - q1

    if iqr == 0:
        return pd.Series(np.zeros(len(series)), index=series.index)

    return (series - q3) / iqr


def score_features_by_segment(df: pd.DataFrame,
                               feature_cols: List[str],
                               segment_col: str = "Segment",
                               iqr_multiplier: float = 2.0) -> pd.DataFrame:
    """
    Run IQR scoring on each feature, within each segment.

    Adds columns:
        {feature}_iqr_score  — raw IQR score
    And a summary column:
        iqr_flag             — True if ANY feature exceeds the multiplier threshold
        iqr_max_score        — highest IQR score across all features for this row
        iqr_triggered_cols   — comma-separated list of features that triggered

    Parameters
    ----------
    df              : partner-level aggregated DataFrame
    feature_cols    : which columns to score
    segment_col     : column to group by before scoring
    iqr_multiplier  : IQRs above Q3 required to flag (default 2.0)
    """
    df = df.copy()
    score_cols = []

    for col in feature_cols:
        if col not in df.columns:
            continue
        score_col = f"{col}_iqr_score"
        df[score_col] = (
            df.groupby(segment_col)[col]
            .transform(iqr_score)
        )
        score_cols.append((col, score_col))

    if not score_cols:
        df["iqr_flag"] = False
        df["iqr_max_score"] = 0.0
        df["iqr_triggered_cols"] = ""
        return df

    raw_score_cols = [s for _, s in score_cols]
    score_matrix   = df[raw_score_cols]

    df["iqr_max_score"] = score_matrix.max(axis=1)
    df["iqr_flag"]      = df["iqr_max_score"] > iqr_multiplier

    def triggered(row):
        cols = [
            orig for (orig, sc) in score_cols
            if row[sc] > iqr_multiplier
        ]
        return ", ".join(cols)

    df["iqr_triggered_cols"] = df.apply(triggered, axis=1)

    return df
