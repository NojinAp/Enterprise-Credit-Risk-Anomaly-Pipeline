from pyspark.sql import SparkSession

def get_spark():
    spark = SparkSession.builder \
        .appName("Enterprise-Anomaly-Detection") \
        .config("spark.driver.memory", "4g") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    return spark