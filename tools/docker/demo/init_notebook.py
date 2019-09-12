from pyspark.sql import SparkSession
import os
spark = SparkSession.builder \
    .master("local[*]") \
    .appName("MMLSpark Docker App") \
    .config("spark.jars.packages", "com.microsoft.ml.spark:mmlspark_2.11:" + os.environ["MMLSPARK_VERSION"] +
            ",org.apache.hadoop:hadoop-azure:2.7.0") \
    .config("spark.jars.repositories", "https://mmlspark.azureedge.net/maven") \
    .getOrCreate()
sc = spark.sparkContext
sc.hadoopConfiguration.set("fs.azure", "org.apache.hadoop.fs.azure.NativeAzureFileSystem")

