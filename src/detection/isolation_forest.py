"""
Isolation Forest anomaly detection.

Runs per-segment so each vertical is scored against its own peer group
rather than the entire dataset globally. This prevents large corporate
partners from drowning out signals in smaller segments.
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from typing import List


def run_isolation_forest_by_segment(
    df: pd.DataFrame,
    feature_cols: List[str],
    segment_col: str = "Segment",
    contamination: float = 0.05, # Expected fraction of outliers
    n_estimators: int = 200,
    random_state: int = 42,
    min_segment_size: int = 10,
) -> pd.DataFrame:
    """
    Run Isolation Forest independently within each segment.

    Segments smaller than `min_segment_size` are scored globally instead
    to avoid model instability on tiny groups.

    Adds columns:
        isolation_forest_flag   - True if flagged as anomaly (-1 from sklearn)
        isolation_forest_score  - raw decision function score (lower = more anomalous)
        risk_score              - normalized 0-1 score (1 = highest risk)

    Parameters
    ----------
    df                 : partner-level aggregated DataFrame
    feature_cols       : features to pass to the model
    segment_col        : column defining peer groups
    contamination      : expected fraction of outliers
    n_estimators       : number of trees in the forest
    random_state       : for reproducibility
    min_segment_size   : segments below this size are scored globally
    """
    df = df.copy()
    available_cols = [c for c in feature_cols if c in df.columns]

    results = []
    global_model   = None
    global_scaler  = None

    segments = df[segment_col].unique()

    for segment in segments:
        mask  = df[segment_col] == segment
        group = df[mask].copy()

        X = group[available_cols].fillna(0)

        if len(group) >= min_segment_size:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            model = IsolationForest(
                n_estimators=n_estimators,
                contamination=contamination,
                random_state=random_state,
            )
            model.fit(X_scaled)
        else:
            # Fall back to global model for tiny segments
            if global_model is None:
                global_scaler = StandardScaler()
                X_all = df[available_cols].fillna(0)
                X_all_scaled = global_scaler.fit_transform(X_all)
                global_model = IsolationForest(
                    n_estimators=n_estimators,
                    contamination=contamination,
                    random_state=random_state,
                )
                global_model.fit(X_all_scaled)
            scaler = global_scaler
            model  = global_model
            X_scaled = scaler.transform(X)

        group["isolation_forest_flag"]  = model.predict(X_scaled) == -1
        group["isolation_forest_score"] = model.decision_function(X_scaled)
        results.append(group)

    df_out = pd.concat(results).sort_index()

    # Normalize score to 0-1 (1 = most anomalous)
    s_min = df_out["isolation_forest_score"].min()
    s_max = df_out["isolation_forest_score"].max()
    denom = (s_max - s_min) if s_max != s_min else 1
    df_out["risk_score"] = 1 - (df_out["isolation_forest_score"] - s_min) / denom

    return df_out
