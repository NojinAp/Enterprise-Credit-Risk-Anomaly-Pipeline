from spark_session import get_spark
from pyspark.sql import functions as F

def build_staging():
    spark = get_spark()

    # Load raw CSV
    df_bp = spark.read.csv(
        "data/raw/business_partners.csv",
        header=True,
        inferSchema=False
    )

    print("Raw rows loaded:", df_bp.count())

    # Select and cast useful columns
    stg_bp = df_bp.select(
        F.col("BusinessPartner").cast("string"),
        F.col("BusinessPartnerName").cast("string"),
        F.col("FirstName").cast("string"),
        F.col("LastName").cast("string"),
        F.col("PostalCode").cast("string"),
        F.col("CompanyCode").cast("string"),
        F.col("ContractAccount").cast("string"),
        F.col("CADocumentNumber").cast("string"),
        F.col("CADocumentType").cast("string"),
        F.col("CAMainTransaction").cast("string"),
        F.col("CASubTransaction").cast("string"),
        F.col("CAAccountDeterminationCode").cast("string"),
        F.abs(F.col("AmountInDisplayCurrency").cast("double")).alias("Amount"),
        F.col("BalanceAmountInDisplayCurrency").cast("double").alias("Balance"),
        F.col("DebitAmountInDisplayCrcy").cast("double").alias("TotalDebit"),
        F.col("CreditAmountInDisplayCrcy").cast("double").alias("TotalCredit"),
        F.col("NumberOfItems").cast("integer"),
        # Parse SAP Unix timestamp /Date(1745884800000)/
        F.to_timestamp(
            F.regexp_extract(
                F.col("CANetDueDate"), r"\d+", 0
            ).cast("long") / 1000
        ).alias("NetDueDate")
    ).filter(
        F.col("Amount").isNotNull()
    ).filter(
        F.col("BusinessPartner").isNotNull()
    )

    # Write silver layer
    stg_bp.write.mode("overwrite").parquet("data/staging/business_partners")

    print("Staging complete; parquet written to data/staging/")
    spark.stop()


if __name__ == "__main__":
    build_staging()