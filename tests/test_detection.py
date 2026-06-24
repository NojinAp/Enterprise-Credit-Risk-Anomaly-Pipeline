"""
Tests for detection modules:
  src/detection/iqr_scoring.py
  src/detection/isolation_forest.py
  src/alerts/explanations.py

Run with: pytest tests/ -v
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
import pandas as pd
import numpy as np
from detection.iqr_scoring import iqr_score, score_features_by_segment
from detection.isolation_forest import run_isolation_forest_by_segment
from alerts.explanations import build_flag_reason, build_alert_column

# Fixtures
@pytest.fixture
def scored_partners():
    """
    Five partners, one clear outlier (BP003) with extreme values.
    Used across IQR, IF, and explanation tests.
    """
    return pd.DataFrame({
        "BusinessPartner":  ["BP001", "BP002", "BP003", "BP004", "BP005"],
        "Segment":          ["Standard"] * 5,
        "TotalTransactions": [5,  10,  500,  8,   6],
        "TotalAmount":       [1000, 2000, 500000, 1500, 1200],
        "ReversalRatio":     [0.5, 0.3, 20.0, 0.4, 0.5],
        "MaxSingleAmount":   [200, 300, 100000, 250, 220],
        "Balance":           [100, 200, 250000, 150, 120],
        "TotalDebit":        [500, 600, 300000, 550, 520],
        "AvgDaysOverdue":    [10,  15,  400,   12,  11],
        "MaxDaysOverdue":    [30,  45,  700,   35,  32],
        "HighValueCount":    [0,   0,   10,    0,   0],
        "ExtremeValueCount": [0,   0,   5,     0,   0],
        "UniqueDocTypes":    [2,   3,   8,     2,   2],
        "UniqueMainTransactions": [1, 2, 6,   1,   1],
        "UniqueContracts":   [1,   2,   15,   1,   1],
        "AvgAmount":         [200, 200, 1000, 200, 200],
        "AmountVolatility":  [0.1, 0.1, 3.0,  0.1, 0.1],
    })

# iqr_score
def test_iqr_score_returns_series():
    s = pd.Series([1, 2, 3, 4, 100])
    result = iqr_score(s)
    assert isinstance(result, pd.Series)
    assert len(result) == len(s)


def test_iqr_score_outlier_is_positive():
    s = pd.Series([1, 2, 3, 4, 100])
    result = iqr_score(s)
    # The value 100 is far above Q3; should have highest positive score
    assert result.iloc[-1] == result.max()
    assert result.max() > 2.0


def test_iqr_score_zero_iqr_returns_zeros():
    s = pd.Series([5, 5, 5, 5, 5])
    result = iqr_score(s)
    assert (result == 0).all()


# score_features_by_segment
def test_iqr_scoring_flags_outlier(scored_partners):
    result = score_features_by_segment(
        scored_partners,
        feature_cols=["ReversalRatio", "TotalAmount"],
        segment_col="Segment",
        iqr_multiplier=2.0,
    )
    bp003 = result[result["BusinessPartner"] == "BP003"]
    assert bp003["iqr_flag"].iloc[0] is True or bp003["iqr_flag"].iloc[0] == True


def test_iqr_scoring_does_not_flag_normal(scored_partners):
    result = score_features_by_segment(
        scored_partners,
        feature_cols=["ReversalRatio", "TotalAmount"],
        segment_col="Segment",
        iqr_multiplier=2.0,
    )
    normal = result[result["BusinessPartner"] == "BP001"]
    assert not normal["iqr_flag"].iloc[0]


def test_iqr_scoring_triggered_cols_populated(scored_partners):
    result = score_features_by_segment(
        scored_partners,
        feature_cols=["ReversalRatio", "TotalAmount"],
        segment_col="Segment",
        iqr_multiplier=2.0,
    )
    bp003 = result[result["BusinessPartner"] == "BP003"]
    assert len(bp003["iqr_triggered_cols"].iloc[0]) > 0


def test_iqr_scoring_adds_score_columns(scored_partners):
    result = score_features_by_segment(
        scored_partners,
        feature_cols=["ReversalRatio"],
        segment_col="Segment",
    )
    assert "ReversalRatio_iqr_score" in result.columns
    assert "iqr_flag" in result.columns
    assert "iqr_max_score" in result.columns


# run_isolation_forest_by_segment
def test_isolation_forest_adds_flag_column(scored_partners):
    result = run_isolation_forest_by_segment(
        scored_partners,
        feature_cols=["TotalTransactions", "ReversalRatio", "TotalAmount"],
        segment_col="Segment",
    )
    assert "isolation_forest_flag" in result.columns
    assert "risk_score" in result.columns


def test_isolation_forest_risk_score_range(scored_partners):
    result = run_isolation_forest_by_segment(
        scored_partners,
        feature_cols=["TotalTransactions", "ReversalRatio"],
        segment_col="Segment",
    )
    assert result["risk_score"].between(0, 1).all()


def test_isolation_forest_flags_some_partners(scored_partners):
    result = run_isolation_forest_by_segment(
        scored_partners,
        feature_cols=["TotalTransactions", "ReversalRatio", "TotalAmount"],
        segment_col="Segment",
        contamination=0.2,
    )
    assert result["isolation_forest_flag"].sum() >= 1


# build_alert_column
def test_flag_reason_populated_for_flagged(scored_partners):
    partners = score_features_by_segment(
        scored_partners,
        feature_cols=["ReversalRatio", "TotalAmount"],
        segment_col="Segment",
        iqr_multiplier=2.0,
    )
    partners["isolation_forest_flag"] = False
    partners["risk_score"] = 0.5
    partners = build_alert_column(partners)

    bp003 = partners[partners["BusinessPartner"] == "BP003"]
    assert len(bp003["flag_reason"].iloc[0]) > 0


def test_flag_reason_empty_for_unflagged(scored_partners):
    scored_partners["iqr_flag"] = False
    scored_partners["iqr_triggered_cols"] = ""
    scored_partners["isolation_forest_flag"] = False
    scored_partners["risk_score"] = 0.1
    result = build_alert_column(scored_partners)

    assert (result["flag_reason"] == "").all()


def test_flag_reason_mentions_reversal_ratio():
    row = pd.Series({
        "BusinessPartner": "BP_TEST",
        "Segment": "Standard",
        "ReversalRatio": 12.5,
        "ReversalRatio_iqr_score": 3.5,
        "iqr_triggered_cols": "ReversalRatio",
        "isolation_forest_flag": False,
        "risk_score": 0.8,
        "Balance": 0,
    })
    reason = build_flag_reason(row)
    assert "ReversalRatio" in reason
    assert "12.5" in reason
