from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, IntegerType, StringType, DoubleType
from pyspark.sql.functions import when
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler
from pyspark.ml.classification import RandomForestClassifier

# --- Configuration ---
input_path = "hdfs://localhost:9000/user/healthcare-dataset-stroke-data.csv"
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

spark = SparkSession.builder.appName("StrokeRFTraining").getOrCreate()

# --- Load Data ---
static_data_df = spark.read.csv(input_path, header=True, schema=schema)

# --- Impute Missing Values ---
static_data_df = static_data_df.fillna({"bmi": 25.0, "smoking_status": "No Info"})
for col_name in ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]:
    static_data_df = static_data_df.fillna({col_name: "Unknown"})

# --- Feature Engineering: Age Bucket ---
static_data_df = static_data_df.withColumn(
    "age_bucket",
    when(static_data_df["age"] < 40, "young")
    .when(static_data_df["age"] < 60, "middle")
    .otherwise("senior")
)

# --- String Indexing and Encoding ---
indexed_data_df = static_data_df
for col_name in ["gender", "ever_married", "work_type", "Residence_type", "smoking_status", "age_bucket"]:
    idx = StringIndexer(inputCol=col_name, outputCol=col_name+"_index", handleInvalid="keep").fit(indexed_data_df)
    idx.write().overwrite().save(f"/user/stroke/indexer_{col_name}")  # Save for inference
    indexed_data_df = idx.transform(indexed_data_df)

encoder = OneHotEncoder(
    inputCols=[f"{c}_index" for c in ["gender", "ever_married", "work_type", "Residence_type", "smoking_status", "age_bucket"]],
    outputCols=[f"{c}_encoded" for c in ["gender", "ever_married", "work_type", "Residence_type", "smoking_status", "age_bucket"]],
    handleInvalid="keep"
).fit(indexed_data_df)
encoder.write().overwrite().save("/user/stroke/ohe_model")
encoded_df = encoder.transform(indexed_data_df)

# --- Assemble Features ---
assembler = VectorAssembler(
    inputCols=["age", "hypertension", "heart_disease", "avg_glucose_level", "bmi"]
    + [f"{c}_encoded" for c in ["gender", "ever_married", "work_type", "Residence_type", "smoking_status", "age_bucket"]],
    outputCol="features"
)
prepped = assembler.transform(encoded_df)

# --- Feature Scaling ---
scaler = StandardScaler(inputCol="features", outputCol="features_scaled", withMean=True, withStd=True)
scaler_model = scaler.fit(prepped)
prepped = scaler_model.transform(prepped)
scaler_model.write().overwrite().save("/user/stroke/scaler_model")

# --- Class Imbalance Handling (weights) ---
num_pos = prepped.filter(prepped["stroke"] == 1).count()
num_neg = prepped.filter(prepped["stroke"] == 0).count()
weight_pos = num_neg / num_pos
prepped = prepped.withColumn("classWeightCol", when(prepped["stroke"] == 1, weight_pos).otherwise(1.0))

# --- Train/Test Split (optional) ---
train_data, test_data = prepped.randomSplit([0.8, 0.2], seed=42)

# --- Random Forest Training with Weights ---
rf = RandomForestClassifier(featuresCol="features_scaled", labelCol="stroke", weightCol="classWeightCol", seed=42)
model = rf.fit(train_data)
model.write().overwrite().save("/user/stroke/rf_model")

# --- Evaluate on Test Data ---
predictions = model.transform(test_data)
predictions.select("stroke", "prediction", "probability").show(5)

print("Training complete! Model, scaler, encoders, and indexers are saved to /user/stroke/")
spark.stop()