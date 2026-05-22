# Credit Risk Anomaly Detection Pipeline

An end-to-end data engineering pipeline that ingests live credit data from SAP S/4HANA via OData REST API, transforms it through a Medallion Architecture (Bronze/Silver/Gold) using PySpark, and applies Isolation Forest anomaly detection to identify high-risk business partners.

---

## Architecture

```
SAP S/4HANA (OData REST API)
            ↓
Bronze Layer (Raw CSV)
            ↓
Silver Layer (PySpark → Parquet)
            ↓
Gold Layer (Aggregated → Parquet)
            ↓
Isolation Forest Model
            ↓
Anomaly Watchlist (CSV)

Orchestrated with Apache Airflow, containerized with Docker.
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Data Source | SAP S/4HANA via OData REST API |
| Ingestion | Python, Requests |
| Transformation | PySpark 4.1.1 |
| Storage | Parquet (Medallion Architecture) |
| ML Model | Isolation Forest (scikit-learn) |
| Orchestration | Apache Airflow |
| Containerization | Docker |
| Exploration | Jupyter Notebooks |

---

## Project Structure

```
├── dags/
│   └── pipeline_dag.py          # Airflow DAG — daily schedule at 6am
├── data/
│   └── README.md                # Data layer descriptions
├── notebooks/
│   ├── 01_eda.ipynb             # Exploratory data analysis
│   └── 02_features_and_model.ipynb  # Feature engineering + model
├── src/
│   ├── sap_odata_connector.py   # Live SAP data ingestion
│   ├── spark_session.py         # PySpark session configuration
│   ├── staging.py               # Bronze → Silver transformation
│   └── curated.py               # Silver → Gold aggregation
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

---

## Medallion Architecture

**Bronze**: Raw CSV files pulled directly from SAP S/4HANA OData API. Never modified.

**Silver**: PySpark transformations: correct types, parsed SAP timestamps, absolute amounts, null filtering. Written as Parquet.

**Gold**: Aggregated to one row per business partner with 23 engineered features including reversal ratios, z-scores, overdue metrics, and balance exposure. Written as Parquet.

---

## Anomaly Detection

Isolation Forest trained on 20,605 business partners with 23 features:

- Transaction volume and amount statistics
- Document type diversity
- Reversal ratio (ZC/AC document pairs) — key fraud signal
- Balance and debit/credit exposure
- Days overdue metrics
- Peer benchmarking z-scores

**Results:** 1,031 high-risk partners flagged (5% contamination rate) from 616,248 transactions across Purolator's accounts receivable portfolio.

---

## Key Findings

- **1,031 high-risk partners flagged** (5% contamination rate) from the accounts receivable worklist
- **Extreme reversal patterns** — several partners with reversal ratios exceeding 1,000x, indicating systematic billing reversals requiring investigation
- **Single transactions exceeding $15M** flagged for manual review
- **Partners with negative balances exceeding $39M** identified — indicating significant outstanding exposure for Purolator
- **High-volume outliers** — one partner with 17,743 transactions and a reversal ratio of 17,728x, far outside peer group norms
- **Peer benchmarking** via z-scores surfaced partners that appear normal in absolute terms but are statistical outliers within their cohort

---

## How to Run

> **Note:** This pipeline connects to an internal SAP S/4HANA staging environment. External users will need to substitute their own SAP OData endpoint and credentials in `.env`. The pipeline architecture and transformation logic is fully reusable with any SAP S/4HANA instance.

### With Docker (recommended)

```bash
git clone https://github.com/NojinAp/Enterprise-Credit-Risk-Anomaly-Pipeline
cd Enterprise-Credit-Risk-Anomaly-Pipeline
cp .env.example .env
# Add your SAP credentials to .env
docker-compose up
```

Airflow UI available at `http://localhost:8080`

### Locally

```bash
pip install -r requirements.txt
python src/sap_odata_connector.py
python src/staging.py
python src/curated.py
jupyter notebook
```