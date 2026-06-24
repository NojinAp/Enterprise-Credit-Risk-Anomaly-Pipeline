"""
Tests for src/alerts/historical.py and src/alerts/explanations.py
Run with: pytest tests/ -v
"""
import sys, os, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from alerts.historical import (
    deduplicate_within_run,
    load_historical_alerts,
    filter_recently_alerted,
    append_to_historical,
    run_full_deduplication,
)
from alerts.explanations import build_flag_reason, build_alert_column


@pytest.fixture
def sample_flagged():
    return pd.DataFrame({
        "BusinessPartner":      ["BP001", "BP002", "BP003", "BP004"],
        "Segment":              ["Standard", "Standard", "Corporate", "Government"],
        "PeerGroup":            ["Standard_Receivable"] * 2 + ["Corporate_Payment", "Government_Other"],
        "isolation_forest_flag": [True, False, True, True],
        "iqr_flag":             [True, True, False, True],
        "risk_score":           [0.95, 0.60, 0.75, 0.88],
        "ReversalRatio":        [12.0, 6.0, 1.0, 8.0],
        "TotalAmount":          [50000, 20000, 10000, 30000],
        "MaxSingleAmount":      [5000, 2000, 1000, 3000],
        "flag_reason":          ["reason1", "reason2", "reason3", "reason4"],
        "ReversalRatio_iqr_score": [4.5, 2.5, 0.0, 3.1],
        "iqr_triggered_cols":   ["ReversalRatio", "TotalAmount", "", "ReversalRatio"],
    })


# deduplicate_within_run
def test_dedup_keeps_both_flagged_first(sample_flagged):
    # Add a duplicate of BP001 with lower confidence
    dup = sample_flagged.copy()
    dup.loc[0, "isolation_forest_flag"] = False  # lower confidence version
    combined = pd.concat([sample_flagged, dup]).reset_index(drop=True)
    result = deduplicate_within_run(combined)
    # BP001 should appear once
    assert (result["BusinessPartner"] == "BP001").sum() == 1


def test_dedup_no_duplicates_unchanged(sample_flagged):
    result = deduplicate_within_run(sample_flagged)
    assert len(result) == len(sample_flagged)


def test_dedup_empty_df():
    result = deduplicate_within_run(pd.DataFrame())
    assert result.empty


# filter_recently_alerted
def test_filter_suppresses_recent(sample_flagged):
    historical = pd.DataFrame({
        "BusinessPartner": ["BP001"],
        "run_date": [pd.Timestamp.now() - timedelta(days=5)],
    })
    fresh, suppressed = filter_recently_alerted(sample_flagged, historical, cooldown_days=30)
    assert "BP001" not in fresh["BusinessPartner"].values
    assert "BP001" in suppressed["BusinessPartner"].values


def test_filter_allows_old_alerts(sample_flagged):
    historical = pd.DataFrame({
        "BusinessPartner": ["BP001"],
        "run_date": [pd.Timestamp.now() - timedelta(days=60)],
    })
    fresh, suppressed = filter_recently_alerted(sample_flagged, historical, cooldown_days=30)
    assert "BP001" in fresh["BusinessPartner"].values
    assert len(suppressed) == 0


def test_filter_empty_history(sample_flagged):
    fresh, suppressed = filter_recently_alerted(sample_flagged, pd.DataFrame())
    assert len(fresh) == len(sample_flagged)
    assert len(suppressed) == 0


# append_to_historical
def test_append_creates_file(sample_flagged):
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "historical.csv")
        result = append_to_historical(sample_flagged, path=path)
        assert os.path.exists(path)
        assert len(result) == len(sample_flagged)


def test_append_accumulates(sample_flagged):
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "historical.csv")
        append_to_historical(sample_flagged, path=path)
        append_to_historical(sample_flagged, path=path)
        result = pd.read_csv(path)
        assert len(result) == len(sample_flagged) * 2


# run_full_deduplication
def test_full_dedup_returns_all_keys(sample_flagged):
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "historical.csv")
        result = run_full_deduplication(sample_flagged, historical_path=path)
        assert "fresh" in result
        assert "suppressed" in result
        assert "historical" in result


def test_full_dedup_first_run_all_fresh(sample_flagged):
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "historical.csv")
        result = run_full_deduplication(sample_flagged, historical_path=path)
        assert len(result["fresh"]) == len(sample_flagged)
        assert len(result["suppressed"]) == 0


def test_full_dedup_second_run_all_suppressed(sample_flagged):
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "historical.csv")
        run_full_deduplication(sample_flagged, historical_path=path)
        result = run_full_deduplication(sample_flagged, historical_path=path, cooldown_days=30)
        assert len(result["suppressed"]) == len(sample_flagged)
        assert len(result["fresh"]) == 0


# build_flag_reason
def test_flag_reason_reversal_ratio():
    row = pd.Series({
        "PeerGroup": "Standard_Receivable",
        "ReversalRatio": 12.5,
        "ReversalRatio_iqr_score": 3.5,
        "iqr_triggered_cols": "ReversalRatio",
        "isolation_forest_flag": False,
        "risk_score": 0.8,
        "Balance": 0,
        "AvgROSScore": 0.1,
    })
    reason = build_flag_reason(row)
    assert "ReversalRatio" in reason
    assert "12.5" in reason


def test_flag_reason_ros_score():
    row = pd.Series({
        "PeerGroup": "Standard_Receivable",
        "AvgROSScore": 0.9,
        "AvgROSScore_iqr_score": 3.0,
        "iqr_triggered_cols": "AvgROSScore",
        "isolation_forest_flag": False,
        "risk_score": 0.7,
        "Balance": 0,
        "ReversalRatio": 0.5,
    })
    reason = build_flag_reason(row)
    assert "authority limit" in reason.lower() or "ROS" in reason


def test_flag_reason_if_only():
    row = pd.Series({
        "PeerGroup": "Standard_Payment",
        "iqr_triggered_cols": "",
        "isolation_forest_flag": True,
        "risk_score": 0.85,
        "Balance": 0,
        "ReversalRatio": 0.5,
        "AvgROSScore": 0.1,
    })
    reason = build_flag_reason(row)
    assert "Isolation Forest" in reason


def test_build_alert_column_empty_for_unflagged():
    df = pd.DataFrame({
        "BusinessPartner": ["BP001"],
        "PeerGroup": ["Standard_Receivable"],
        "iqr_flag": [False],
        "iqr_triggered_cols": [""],
        "isolation_forest_flag": [False],
        "risk_score": [0.1],
        "Balance": [0],
        "ReversalRatio": [0.5],
        "AvgROSScore": [0.1],
    })
    result = build_alert_column(df)
    assert result["flag_reason"].iloc[0] == ""
