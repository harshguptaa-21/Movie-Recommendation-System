
from pyspark.sql import SparkSession
from pyspark.sql.functions import count, avg, col
from pathlib import Path

# Start Spark on our MacBook
spark = (
    SparkSession.builder
    .appName("MovieRecommendationFeatures")
    .master("local[4]")
    .config("spark.driver.memory", "4g")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

# Read TRAINING data only
ratings = spark.read.csv(
    "data/processed/train.csv",
    header=True,
    inferSchema=True
)

print("Total training ratings:", ratings.count())

# 1. Movie popularity and average rating
movie_features = ratings.groupBy("movieId").agg(
    count("*").alias("popularity"),
    avg("rating").alias("average_rating")
)

# 2. Number of ratings per user
user_features = ratings.groupBy("userId").agg(
    count("*").alias("ratings_count")
)

print("\nTOP 10 MOST POPULAR MOVIES:")
movie_features.orderBy(
    col("popularity").desc()
).show(10)

print("\nTOP 10 MOST ACTIVE USERS:")
user_features.orderBy(
    col("ratings_count").desc()
).show(10)

# Save features
Path("data/features").mkdir(parents=True, exist_ok=True)

movie_features.write.mode("overwrite").parquet(
    "data/features/movie_features"
)

user_features.write.mode("overwrite").parquet(
    "data/features/user_features"
)

print("\nSpark feature engineering completed!")

spark.stop()

