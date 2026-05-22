from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from datetime import timedelta
import subprocess
import sys
import os

# ── Default arguments for all tasks ──────────────────────────────────────────
default_args = {
    "owner": "nozhin.azarpanah",
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False,
}

# ── Define task functions ─────────────────────────────────────────────────────
def ingest():
    """Pull live data from SAP S/4HANA via OData API"""
    result = subprocess.run(
        [sys.executable, "src/sap_odata_connector.py"],
        capture_output=True, text=True
    )
    print(result.stdout)
    if result.returncode != 0:
        raise Exception(f"Ingestion failed: {result.stderr}")

def staging():
    """Transform raw CSV to clean parquet — Bronze to Silver"""
    result = subprocess.run(
        [sys.executable, "src/staging.py"],
        capture_output=True, text=True
    )
    print(result.stdout)
    if result.returncode != 0:
        raise Exception(f"Staging failed: {result.stderr}")

def curated():
    """Aggregate per business partner — Silver to Gold"""
    result = subprocess.run(
        [sys.executable, "src/curated.py"],
        capture_output=True, text=True
    )
    print(result.stdout)
    if result.returncode != 0:
        raise Exception(f"Curated failed: {result.stderr}")

def anomaly_detection():
    """Run Isolation Forest and export watchlist"""
    import pandas as pd
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
    import os

    # Load curated gold layer
    df = pd.read_parquet("data/curated/business_partners")

    feature_cols = [
        "TotalTransactions", "TotalAmount", "AvgAmount",
        "MaxSingleAmount", "UniqueDocTypes", "UniqueMainTransactions",
        "UniqueContracts", "ACDocCount", "ZCDocCount",
        "ReversalRatio", "AmountVolatility", "Balance",
        "TotalDebit", "TotalCredit", "AvgDaysOverdue",
        "MaxDaysOverdue", "HighValueCount", "ExtremeValueCount",
        "TotalTransactions_zscore", "MaxSingleAmount_zscore",
        "ReversalRatio_zscore", "Balance_zscore", "TotalDebit_zscore"
    ]

    X = df[feature_cols].fillna(0)
    X_scaled = StandardScaler().fit_transform(X)

    model = IsolationForest(
        n_estimators=200,
        contamination=0.05,
        random_state=42
    )
    model.fit(X_scaled)

    df["anomaly_flag"]  = model.predict(X_scaled)
    df["anomaly_score"] = model.decision_function(X_scaled)
    df["risk_score"] = 1 - (
        (df["anomaly_score"] - df["anomaly_score"].min()) /
        (df["anomaly_score"].max() - df["anomaly_score"].min())
    )

    os.makedirs("outputs", exist_ok=True)
    df.sort_values("risk_score", ascending=False)[[
        "BusinessPartner", "BusinessPartnerName",
        "TotalTransactions", "TotalAmount", "AvgAmount",
        "MaxSingleAmount", "ReversalRatio", "Balance",
        "TotalDebit", "AvgDaysOverdue", "anomaly_flag", "risk_score"
    ]].to_csv("outputs/anomaly_watchlist.csv", index=False)

    flagged = (df["anomaly_flag"] == -1).sum()
    print(f"Anomaly detection complete — {flagged} partners flagged")

# ── Define DAG ────────────────────────────────────────────────────────────────
with DAG(
    dag_id="purolator_credit_anomaly_pipeline",
    description="Daily credit anomaly detection pipeline — SAP S/4HANA to watchlist",
    default_args=default_args,
    schedule_interval="0 6 * * *",  # every day at 6am
    start_date=days_ago(1),
    catchup=False,
    tags=["credit", "anomaly", "SAP", "finance"],
) as dag:

    task_ingest = PythonOperator(
        task_id="ingest_sap_data",
        python_callable=ingest,
    )

    task_staging = PythonOperator(
        task_id="build_staging_layer",
        python_callable=staging,
    )

    task_curated = PythonOperator(
        task_id="build_curated_layer",
        python_callable=curated,
    )

    task_model = PythonOperator(
        task_id="run_anomaly_detection",
        python_callable=anomaly_detection,
    )

    # ── Task dependencies — defines the pipeline order ────────────────────────
    task_ingest >> task_staging >> task_curated >> task_model