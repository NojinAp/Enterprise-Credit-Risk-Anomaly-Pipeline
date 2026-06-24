"""
Tests for src/features/transaction_features.py
Run with: pytest tests/ -v
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
import pandas as pd
import numpy as np
from features.transaction_features import (
    add_reversal_ratio,
    add_high_value_flags,
    add_ros_score,
    add_round_transaction_flag,
    net_cancellation_pairs,
    add_segment_label,
    add_transaction_type_label,
    add_peer_zscores,
)


@pytest.fixture
def sample_transactions():
    return pd.DataFrame({
        "BusinessPartner": ["BP001", "BP001", "BP002", "BP002", "BP003"],
        "CADocumentType":  ["AC",    "ZC",    "AC",    "AC",    "ZC"],
        "Amount":          [100.0,   100.0,   500.0,   200.0,   50.0],
        "CAAccountDeterminationCode": ["Z1", "Z1", "Z4", "Z4", "Z2"],
        "CAMainTransaction": ["R000", "R000", "0060", "0060", "MG00"],
        "NumberOfItems": [1, 1, 2, 1, 1],
    })


@pytest.fixture
def sample_partners():
    return pd.DataFrame({
        "BusinessPartner":  ["BP001", "BP002", "BP003", "BP004", "BP005"],
        "PeerGroup":        ["Standard_Receivable"] * 3 + ["Corporate_Payment"] * 2,
        "TotalTransactions": [5,  50,  500,  10,  20],
        "TotalAmount":       [1000, 5000, 500000, 2000, 3000],
        "ReversalRatio":     [0.5, 0.5, 15.0, 0.1, 0.2],
        "MaxSingleAmount":   [200, 300, 100000, 500, 600],
        "Balance":           [100, 200, 50000, 300, 400],
        "TotalDebit":        [500, 600, 200000, 700, 800],
        "AvgDaysOverdue":    [10,  20,  300,   5,   15],
        "MaxDaysOverdue":    [30,  60,  500,   10,  45],
        "HighValueCount":    [0,   0,   5,     0,   0],
        "ExtremeValueCount": [0,   0,   2,     0,   0],
        "UniqueDocTypes":    [2,   3,   4,     2,   2],
        "UniqueMainTransactions": [1, 2, 5,   1,   2],
        "UniqueContracts":   [1,   2,   10,   1,   2],
        "AvgAmount":         [200, 100, 1000, 200, 150],
        "AmountVolatility":  [0.1, 0.2, 2.5,  0.1, 0.1],
        "AvgROSScore":       [0.1, 0.1, 0.9,  0.2, 0.1],
        "MaxROSScore":       [0.2, 0.2, 1.0,  0.3, 0.2],
        "RoundTransactionRate": [0.0, 0.1, 0.8, 0.0, 0.0],
        "NettedPairsCount":  [0,   0,   10,   0,   0],
    })


# add_reversal_ratio
def test_reversal_ratio_adds_columns(sample_transactions):
    result = add_reversal_ratio(sample_transactions)
    assert "ACDocCount" in result.columns
    assert "ZCDocCount" in result.columns


def test_reversal_ratio_counts_correct(sample_transactions):
    result = add_reversal_ratio(sample_transactions)
    bp001 = result[result["BusinessPartner"] == "BP001"]
    assert bp001["ACDocCount"].sum() == 1
    assert bp001["ZCDocCount"].sum() == 1


def test_reversal_ratio_no_mutation(sample_transactions):
    original_cols = sample_transactions.columns.tolist()
    add_reversal_ratio(sample_transactions)
    assert sample_transactions.columns.tolist() == original_cols


# add_high_value_flags
def test_high_value_correct_threshold():
    df = pd.DataFrame({"Amount": [100, 5000, 15000, 200000]})
    result = add_high_value_flags(df, high_threshold=10000, extreme_threshold=100000)
    assert result["IsHighValue"].tolist()    == [0, 0, 1, 1]
    assert result["IsExtremeValue"].tolist() == [0, 0, 0, 1]


def test_high_value_no_flags_below_threshold():
    df = pd.DataFrame({"Amount": [1.0, 50.0, 99.99]})
    result = add_high_value_flags(df, high_threshold=100, extreme_threshold=1000)
    assert result["IsHighValue"].sum() == 0


# add_ros_score
def test_ros_score_added(sample_transactions):
    result = add_ros_score(sample_transactions, authority_limits=[100.0])
    assert "ROSScore" in result.columns


def test_ros_score_exact_limit_is_one():
    df = pd.DataFrame({"Amount": [100.0]})
    result = add_ros_score(df, authority_limits=[100.0], bandwidth=0.15)
    assert abs(result["ROSScore"].iloc[0] - 1.0) < 1e-6


def test_ros_score_far_from_limit_is_low():
    df = pd.DataFrame({"Amount": [1.0]})
    result = add_ros_score(df, authority_limits=[100.0], bandwidth=0.15)
    assert result["ROSScore"].iloc[0] < 0.01


def test_ros_score_between_zero_and_one():
    df = pd.DataFrame({"Amount": [50.0, 100.0, 200.0, 500.0, 1000.0]})
    result = add_ros_score(df, authority_limits=[100.0, 500.0, 1000.0])
    assert result["ROSScore"].between(0, 1).all()


# add_round_transaction_flag
def test_round_flag_detects_round_numbers():
    df = pd.DataFrame({"Amount": [100.0, 50.0, 73.5, 200.0, 0.01]})
    result = add_round_transaction_flag(df, round_thresholds=[50, 100])
    assert result["IsRound100"].tolist() == [1, 0, 0, 1, 0]
    assert result["IsRound50"].tolist()  == [1, 1, 0, 1, 0]


def test_round_flag_any_column():
    df = pd.DataFrame({"Amount": [50.0, 73.5, 100.0]})
    result = add_round_transaction_flag(df, round_thresholds=[50, 100])
    assert result["IsRoundAny"].tolist() == [1, 0, 1]


def test_round_flag_zero_amount_not_flagged():
    df = pd.DataFrame({"Amount": [0.0, 100.0]})
    result = add_round_transaction_flag(df, round_thresholds=[100])
    assert result["IsRound100"].iloc[0] == 0


# net_cancellation_pairs
def test_cancellation_matching_adds_column(sample_transactions):
    result = net_cancellation_pairs(sample_transactions)
    assert "IsNetted" in result.columns


def test_cancellation_matching_nets_exact_pairs():
    df = pd.DataFrame({
        "BusinessPartner": ["BP001", "BP001", "BP002"],
        "CADocumentType":  ["AC",    "ZC",    "AC"],
        "Amount":          [100.0,   100.0,   200.0],
    })
    result = net_cancellation_pairs(df)
    # BP001's AC and ZC of $100 should both be netted
    bp001 = result[result["BusinessPartner"] == "BP001"]
    assert bp001["IsNetted"].sum() == 2


def test_cancellation_matching_leaves_unmatched_alone():
    df = pd.DataFrame({
        "BusinessPartner": ["BP001", "BP001"],
        "CADocumentType":  ["AC",    "ZC"],
        "Amount":          [100.0,   200.0],   # different amounts -- no match
    })
    result = net_cancellation_pairs(df)
    assert result["IsNetted"].sum() == 0


# add_segment_label
def test_segment_label_maps_correctly(sample_transactions):
    result = add_segment_label(sample_transactions)
    assert result[result["CAAccountDeterminationCode"] == "Z1"]["Segment"].iloc[0] == "Standard"


def test_segment_label_unknown_code():
    df = pd.DataFrame({"CAAccountDeterminationCode": ["Z9"]})
    result = add_segment_label(df)
    assert result["Segment"].iloc[0] == "Unknown"


# add_transaction_type_label
def test_transaction_type_maps_correctly(sample_transactions):
    result = add_transaction_type_label(sample_transactions)
    assert "TransactionType" in result.columns
    assert result[result["CAMainTransaction"] == "R000"]["TransactionType"].iloc[0] == "Receivable"


def test_transaction_type_unknown_code():
    df = pd.DataFrame({"CAMainTransaction": ["ZZZZ"]})
    result = add_transaction_type_label(df)
    assert result["TransactionType"].iloc[0] == "Other"


# add_peer_zscores
def test_peer_zscores_within_group(sample_partners):
    result = add_peer_zscores(sample_partners, cols=["TotalTransactions"], segment_col="PeerGroup")
    assert "TotalTransactions_zscore" in result.columns
    bp003_z = result[result["BusinessPartner"] == "BP003"]["TotalTransactions_zscore"].iloc[0]
    assert bp003_z > 1.0


def test_peer_zscores_no_cross_group_contamination(sample_partners):
    result = add_peer_zscores(sample_partners, cols=["TotalTransactions"], segment_col="PeerGroup")
    corp = result[result["PeerGroup"] == "Corporate_Payment"]["TotalTransactions_zscore"]
    assert corp.abs().max() < 2.0
