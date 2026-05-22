FROM python:3.11-slim

# Install Java 17 — required for PySpark
RUN apt-get update && apt-get install -y \
    openjdk-17-jdk \
    curl \
    && rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PATH=$JAVA_HOME/bin:$PATH

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY dags/ ./dags/

ENV AIRFLOW_HOME=/app/airflow
ENV AIRFLOW__CORE__DAGS_FOLDER=/app/dags
ENV AIRFLOW__CORE__LOAD_EXAMPLES=False

RUN airflow db migrate

EXPOSE 8080

CMD ["bash", "-c", "airflow scheduler & airflow webserver --port 8080"]