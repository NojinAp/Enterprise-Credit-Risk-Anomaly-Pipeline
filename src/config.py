"""
Pipeline configuration.

All hardcoded values live here. Change behaviour by editing this file,
not the pipeline code.
"""
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Segment mapping
# ---------------------------------------------------------------------------
SEGMENT_LABELS = {
    "Z1": "Standard",
    "Z2": "Government",
    "Z3": "International",
    "Z4": "Corporate",
    "ZC": "Reversal",
}

# CAMainTransaction -> human-readable type (used as secondary peer group dim)
TRANSACTION_TYPE_LABELS = {
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

# ---------------------------------------------------------------------------
# Document type meanings
# ---------------------------------------------------------------------------
REVERSAL_DOC_TYPE = "ZC"
CREDIT_DOC_TYPE   = "AC"

# ---------------------------------------------------------------------------
# Authority limits for ROS scoring
# Based on empirical analysis of amount percentiles in the dataset:
#   99th pct = $451, 99.5th pct = $1,083
# ---------------------------------------------------------------------------
AUTHORITY_LIMITS = [100.0, 500.0, 1000.0]
ROS_BANDWIDTH    = 0.15   # Gaussian kernel bandwidth as fraction of limit


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------
@dataclass
class ThresholdConfig:
    iqr_multiplier: float = 2.0
    isolation_forest_contamination: float = 0.05
    isolation_forest_n_estimators: int = 200
    isolation_forest_random_state: int = 42
    high_value_threshold: float = 10_000.0
    extreme_value_threshold: float = 100_000.0
    suspicious_reversal_ratio: float = 5.0
    high_balance_threshold: float = 100_000.0
    high_ros_score: float = 0.7     # Flag partners whose AvgROSScore exceeds this
    alert_cooldown_days: int = 30   # Days before re-alerting the same partner


# ---------------------------------------------------------------------------
# Feature columns fed to Isolation Forest
# ---------------------------------------------------------------------------
@dataclass
class FeatureConfig:
    numeric_features: list = field(default_factory=lambda: [
        "TotalTransactions",
        "TotalAmount",
        "AvgAmount",
        "MaxSingleAmount",
        "UniqueDocTypes",
        "UniqueMainTransactions",
        "UniqueContracts",
        "ReversalRatio",
        "AmountVolatility",
        "Balance",
        "TotalDebit",
        "AvgDaysOverdue",
        "MaxDaysOverdue",
        "HighValueCount",
        "ExtremeValueCount",
        # Domain features
        "AvgROSScore",
        "MaxROSScore",
        "RoundTransactionRate",
        "NettedPairsCount",
    ])


# ---------------------------------------------------------------------------
# Staging / preprocessing
# ---------------------------------------------------------------------------
@dataclass
class StagingConfig:
    raw_path: str = "data/raw/business_partners.csv"
    staging_path: str = "data/staging/business_partners"

    drop_columns: list = field(default_factory=lambda: [
        "__metadata", "GeneratedId", "CAWorklistProcessingState",
        "CAWorklistProcessingStateText", "WorklistCreationDate",
        "CAGroupByKey", "Segment", "Division", "BusinessArea",
        "CAAuthorizationGroup", "CASubApplication",
        "CAStatisticalItemCode", "CAContract", "ContractAccountCategory",
        "DisplayCurrency", "CARepetitionItemNumber", "CASubItemNumber",
        "WorklistCreationTime",
    ])


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
@dataclass
class OutputConfig:
    curated_path: str = "data/curated/business_partners"
    watchlist_path: str = "outputs/anomaly_watchlist.csv"
    historical_alerts_path: str = "outputs/historical_alerts.csv"

    watchlist_columns: list = field(default_factory=lambda: [
        "BusinessPartner",
        "BusinessPartnerName",
        "Segment",
        "PeerGroup",
        "TotalTransactions",
        "TotalAmount",
        "AvgAmount",
        "MaxSingleAmount",
        "ReversalRatio",
        "NettedPairsCount",
        "AvgROSScore",
        "MaxROSScore",
        "RoundTransactionRate",
        "Balance",
        "TotalDebit",
        "AvgDaysOverdue",
        "isolation_forest_flag",
        "iqr_flag",
        "risk_score",
        "flag_reason",
    ])


# ---------------------------------------------------------------------------
# Top-level config object
# ---------------------------------------------------------------------------
@dataclass
class PipelineConfig:
    staging: StagingConfig = field(default_factory=StagingConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    thresholds: ThresholdConfig = field(default_factory=ThresholdConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


DEFAULT_CONFIG = PipelineConfig()
