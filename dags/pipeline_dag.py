from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from datetime import timedelta
import subprocess
import sys
import os

# Default arguments for all tasks
default_args = {
    "owner": "nozhin.azarpanah",
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False,
}

# Define task functions
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

# Define DAG
with DAG(
    dag_id="enterprise_credit_anomaly_pipeline",
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

    # Task dependencies (pipeline order)
    task_ingest >> task_staging >> task_curated