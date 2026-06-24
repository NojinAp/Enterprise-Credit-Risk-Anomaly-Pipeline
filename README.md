# Business Partner Credit Risk - Anomaly Detection Pipeline

An end-to-end data engineering pipeline that pulls live accounts receivable data from SAP S/4HANA via OData API, transforms it through a Medallion Architecture using PySpark, and runs two complementary anomaly detection methods to surface high-risk business partners for financial investigation.

---

## How it works

```
SAP S/4HANA OData API
        │
        ▼
  Bronze Layer          raw/business_partners.csv
        │
        ▼
  Silver Layer          staging/business_partners/      (PySpark → Parquet)
        │  type casting, SAP timestamp parsing,
        │  absolute amounts, null filtering
        ▼
  Gold Layer            curated/business_partners/      (Parquet)
        │  one row per business partner
        │  19 engineered features
        │  peer group assignment (Segment × TransactionType)
        ▼
  Detection Layer
        ├── IQR Scoring     (per-peer-group, human-explainable)
        └── Isolation Forest (per-peer-group, multivariate ML)
        │
        ▼
  Watchlist               outputs/anomaly_watchlist.csv
```

Partners flagged by **both** methods are highest-confidence. Each method catches different patterns - agreement is a strong signal.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Data source | SAP S/4HANA via OData REST API |
| Ingestion | Python, Requests |
| Transformation | PySpark 4.1 |
| Storage | Parquet (Medallion Architecture) |
| Detection | IQR scoring + Isolation Forest (scikit-learn) |
| Orchestration | Apache Airflow (daily at 06:00) |
| Containerisation | Docker + docker-compose |
| Exploration | Jupyter Notebooks |

---

## Project Structure

```
├── dags/
│   └── pipeline_dag.py              # Airflow DAG - ingest → stage → curate
├── notebooks/
│   ├── 01_EDA.ipynb                 # Exploratory data analysis
│   └── 02_features_and_model.ipynb  # Feature distributions, detection, watchlist
├── src/
│   ├── config.py                    # All thresholds and feature lists (single source of truth)
│   ├── sap_odata_connector.py       # Live SAP ingestion
│   ├── spark_session.py             # PySpark session factory
│   ├── staging.py                   # Bronze → Silver
│   ├── curated.py                   # Silver → Gold
│   ├── features/
│   │   └── transaction_features.py  # Feature engineering functions
│   ├── detection/
│   │   ├── iqr_scoring.py           # IQR-based outlier scoring per segment
│   │   └── isolation_forest.py      # Isolation Forest per segment
│   └── alerts/
│       ├── explanations.py          # Human-readable flag reasons
│       └── historical.py            # Alert deduplication and history
├── tests/
│   ├── test_features.py
│   ├── test_detection.py
│   └── test_alerts.py
├── Dockerfile
├── docker-compose.yaml
├── requirements.txt
└── .env.example
```

---

## Feature Engineering

19 features are computed per business partner before detection runs:

| Feature | Description |
|---|---|
| `TotalTransactions` | Count of all transactions |
| `TotalAmount` / `AvgAmount` | Sum and mean of absolute transaction amounts |
| `MaxSingleAmount` | Largest single transaction |
| `AmountVolatility` | Coefficient of variation (std / mean) |
| `UniqueDocTypes` | Diversity of SAP document types used |
| `UniqueMainTransactions` | Diversity of transaction types |
| `UniqueContracts` | Number of distinct contract accounts |
| `ReversalRatio` | ZC reversal docs relative to AC credit docs - primary fraud signal |
| `NettedPairsCount` | Matched AC/ZC cancellation pairs (excluded from credit totals) |
| `Balance` / `TotalDebit` | Outstanding financial exposure |
| `AvgDaysOverdue` / `MaxDaysOverdue` | Overdue profile |
| `HighValueCount` / `ExtremeValueCount` | Transactions above $10k / $100k thresholds |
| `AvgROSScore` / `MaxROSScore` | Proximity of amounts to internal authority limit boundaries |
| `RoundTransactionRate` | Fraction of round-number transactions - classic fraud indicator |

All detection runs **within peer groups** (Segment × TransactionType) so a partner is compared to genuine peers, not the full population.

---

## Detection Methods

### IQR Scoring

For each feature, computes how many IQRs above Q3 a partner sits within their peer group. A partner scoring above the multiplier threshold on any feature is flagged. Fully explainable - the flag reason names the specific features that triggered.

### Isolation Forest

Trains an Isolation Forest on all 19 features within each peer group independently. Catches multivariate anomalies that no single feature would reveal - e.g. a partner with moderate amounts but an unusual combination of high reversal ratio, high volatility, and many round transactions.

### Configuration

All thresholds live in [src/config.py](src/config.py) - no magic numbers in pipeline code:

```python
iqr_multiplier = 2.0
isolation_forest_contamination = 0.05
high_value_threshold = 10_000.0
extreme_value_threshold = 100_000.0
suspicious_reversal_ratio = 5.0
alert_cooldown_days = 30
```

---

## Key Results

Run against 264,107 transactions across 20,605 business partners (SAP staging environment):

- **597 partners flagged by both methods** (highest confidence — independent agreement between IQR and Isolation Forest)
- **4,887 total partners flagged** across all confidence tiers
- **Reversal ratios up to 260x** detected — systematic billing reversals requiring investigation
- **$6.17M outstanding balance** on a partner with only 254 transactions — severe exposure mismatch
- **Single transactions up to $152,171** surfaced for manual review
- **MaxROSScore = 1.0** on majority of top-flagged partners — amounts placed exactly at authority limit boundaries, a classic threshold-gaming signal
- Peer-group scoring (Segment × TransactionType) surfaces partners that appear normal globally but are statistical outliers within their true cohort

---

## Quickstart

> **Note:** The pipeline connects to an internal SAP S/4HANA environment. External users need to provide their own OData endpoint and credentials in `.env`. All transformation and detection logic is fully reusable with any SAP S/4HANA instance.

### Docker (recommended)

```bash
git clone <repo-url>
cd anomaly-detection-upgraded
cp .env.example .env
# Fill in your SAP credentials
docker-compose up
```

Airflow UI: `http://localhost:8080`

### Local

**Prerequisite:** Java 17+ must be installed - required by PySpark.

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux
pip install -r requirements.txt

python src/sap_odata_connector.py   # ingest
python src/staging.py               # bronze → silver
python src/curated.py               # silver → gold + detection

jupyter notebook                    # open notebooks/02_features_and_model.ipynb
```

---

## Environment Variables

Copy `.env.example` to `.env` and fill in:

```
SAP_BASE_URL=https://your-sap-host/sap/opu/odata/sap/...
SAP_SESSION_ID=your_jsessionid_cookie
SAP_VCAP_ID=your_vcap_id_cookie
```
