from spark_session import get_spark
from pyspark.sql import functions as F
from pyspark.sql.window import Window

def build_curated():
    spark = get_spark()

    # ── Load silver layer ─────────────────────────────────────────────────────
    stg_bp = spark.read.parquet("data/staging/business_partners")
    print("Silver rows loaded:", stg_bp.count())

    # ── Aggregate per business partner ────────────────────────────────────────
    curated = stg_bp.groupBy("BusinessPartner").agg(
        F.first("BusinessPartnerName", ignorenulls=True).alias("BusinessPartnerName"),
        
        # Volume signals
        F.count("Amount").alias("TotalTransactions"),
        F.sum("Amount").alias("TotalAmount"),
        F.avg("Amount").alias("AvgAmount"),
        F.stddev("Amount").alias("StdAmount"),
        F.max("Amount").alias("MaxSingleAmount"),
        F.sum("NumberOfItems").alias("TotalItems"),

        # Document type diversity
        F.countDistinct("CADocumentType").alias("UniqueDocTypes"),
        F.countDistinct("CAMainTransaction").alias("UniqueMainTransactions"),
        F.countDistinct("ContractAccount").alias("UniqueContracts"),

        # Reversal pattern — ZC reverses AC, high ratio is suspicious
        F.sum(F.when(F.col("CADocumentType") == "AC", 1).otherwise(0)).alias("ACDocCount"),
        F.sum(F.when(F.col("CADocumentType") == "ZC", 1).otherwise(0)).alias("ZCDocCount"),

        # High value flags
        F.sum(F.when(F.col("Amount") > 10000, 1).otherwise(0)).alias("HighValueCount"),
        F.sum(F.when(F.col("Amount") > 100000, 1).otherwise(0)).alias("ExtremeValueCount"),

        # Balance and exposure
        F.max("Balance").alias("Balance"),
        F.max("TotalDebit").alias("TotalDebit"),
        F.max("TotalCredit").alias("TotalCredit"),

        # Overdue signals
        F.avg(
            F.datediff(F.current_date(), F.col("NetDueDate"))
        ).alias("AvgDaysOverdue"),
        F.max(
            F.datediff(F.current_date(), F.col("NetDueDate"))
        ).alias("MaxDaysOverdue"),

        # Credit vs debit ratio
        F.avg(
            F.col("TotalDebit") / (F.abs(F.col("TotalCredit")) + 1)
        ).alias("AvgDebitCreditRatio"),
    ).fillna(0)

    # ── Derived features ──────────────────────────────────────────────────────
    curated = curated \
        .withColumn(
            "ReversalRatio",
            F.col("ZCDocCount") / (F.col("ACDocCount") + 1)
        ) \
        .withColumn(
            "AmountVolatility",
            F.col("StdAmount") / (F.col("AvgAmount") + 1)
        )

    # ── Peer benchmarking — z-score for key metrics ───────────────────────────
    window_all = Window.rowsBetween(
        Window.unboundedPreceding, Window.unboundedFollowing
    )
    for col in ["TotalTransactions", "MaxSingleAmount", "ReversalRatio", "Balance", "TotalDebit"]:
        mean_col = F.mean(col).over(window_all)
        std_col  = F.stddev(col).over(window_all)
        curated  = curated.withColumn(
            col + "_zscore",
            (F.col(col) - mean_col) / (std_col + 1)
        )

    # ── Write gold layer ──────────────────────────────────────────────────────
    curated.write.mode("overwrite").parquet("data/curated/business_partners")

    print("Curated complete — parquet written to data/curated/")
    spark.stop()


if __name__ == "__main__":
    build_curated()