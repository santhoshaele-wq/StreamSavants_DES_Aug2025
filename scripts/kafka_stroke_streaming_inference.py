from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_csv, when
from pyspark.sql.types import StructType, StructField, IntegerType, StringType, DoubleType
from pyspark.ml.feature import StringIndexerModel, OneHotEncoderModel, VectorAssembler, StandardScalerModel
from pyspark.ml.classification import RandomForestClassificationModel

# --- CONFIGURATION ---
TOPIC_NAME = "StrokePrediction"
CHECKPOINT_PATH = "hdfs://localhost:9000/user/stroke/checkpoint"
OUTPUT_PATH = "hdfs://localhost:9000/user/stroke/stroke_predictions"

spark = SparkSession.builder \
    .appName("StrokePrediction") \
    .config("spark.sql.streaming.checkpointLocation", CHECKPOINT_PATH) \
    .getOrCreate()

# --- SCHEMA MATCHES TRAINING PIPELINE ---
schema = StructType([
    StructField("id", IntegerType()),
    StructField("gender", StringType()),
    StructField("age", DoubleType()),
    StructField("hypertension", IntegerType()),
    StructField("heart_disease", IntegerType()),
    StructField("ever_married", StringType()),
    StructField("work_type", StringType()),
    StructField("Residence_type", StringType()),
    StructField("avg_glucose_level", DoubleType()),
    StructField("bmi", DoubleType()),
    StructField("smoking_status", StringType()),
    StructField("stroke", IntegerType())
])

# --- INGEST STREAMING DATA FROM KAFKA ---
df = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("subscribe", TOPIC_NAME) \
    .option("startingOffsets", "earliest") \
    .option("failOnDataLoss", "false") \
    .load()

parsed = df.selectExpr("CAST(value AS STRING)") \
    .select(from_csv(col("value"), schema.simpleString()).alias("data")) \
    .select("data.*")

# --- FILL DEFAULTS FOR ALL FEATURES ---
fill_dict = {
    "age": 0.0,
    "hypertension": 0,
    "heart_disease": 0,
    "avg_glucose_level": 0.0,
    "bmi": 25.0,
    "gender": "Unknown",
    "ever_married": "Unknown",
    "work_type": "Unknown",
    "Residence_type": "Unknown",
    "smoking_status": "No Info"
}
parsed = parsed.fillna(fill_dict)

# --- FEATURE ENGINEERING: AGE BUCKET ---
parsed = parsed.withColumn(
    "age_bucket",
    when(parsed["age"] < 40, "young")
    .when(parsed["age"] < 60, "middle")
    .otherwise("senior")
)

# --- APPLY TRAINED TRANSFORMERS ---
for col_name in ["gender", "ever_married", "work_type", "Residence_type", "smoking_status", "age_bucket"]:
    si_model = StringIndexerModel.load(f"/user/stroke/indexer_{col_name}")
    si_model.setHandleInvalid("keep")
    parsed = si_model.transform(parsed)

encoder = OneHotEncoderModel.load("/user/stroke/ohe_model")
encoder.setInputCols([c + "_index" for c in ["gender", "ever_married", "work_type", "Residence_type", "smoking_status", "age_bucket"]])
encoder.setOutputCols([c + "_encoded" for c in ["gender", "ever_married", "work_type", "Residence_type", "smoking_status", "age_bucket"]])
parsed = encoder.transform(parsed)

# --- FEATURE ASSEMBLING ---
assembler = VectorAssembler(
    inputCols=[
        "age", "hypertension", "heart_disease", "avg_glucose_level", "bmi"
    ] + [c + "_encoded" for c in ["gender", "ever_married", "work_type", "Residence_type", "smoking_status", "age_bucket"]],
    outputCol="features"
)
parsed = assembler.transform(parsed)

# --- FEATURE SCALING (IDENTICAL TO TRAINING) ---
scaler_model = StandardScalerModel.load("/user/stroke/scaler_model")
parsed = scaler_model.transform(parsed)

# --- INFERENCE WITH TRAINED RANDOM FOREST MODEL ---
model = RandomForestClassificationModel.load("/user/stroke/rf_model")
predictions = model.transform(parsed)

# --- WRITE PREDICTIONS TO HDFS FOR ANALYSIS ---
query = predictions.writeStream \
    .format("parquet") \
    .option("path", OUTPUT_PATH) \
    .option("checkpointLocation", CHECKPOINT_PATH) \
    .outputMode("append") \
    .start()

query.awaitTermination()