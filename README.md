# Stroke Prediction - Real-Time Streaming Pipeline

## Overview

A scalable stroke prediction system using Apache Spark, Kafka, and HDFS for real-time data ingestion, processing, and model inference.

## Features

- Real-time data ingestion via Apache Kafka
- Distributed data processing using Spark Structured Streaming
- Persistent storage via Hadoop HDFS
- Batch model training with Spark MLlib (Random Forest)
- Streaming inference with Parquet output
- Exploratory data analysis and visualization

## Tech Stack

- Apache Spark (PySpark)
- Apache Kafka
- Hadoop HDFS
- Python 3.10+
- Jupyter Notebook

## Project Structure

```
stroke-prediction/
├── config/
│   └── kafka_config.json
├── data/
│   ├── healthcare-dataset-stroke-data.csv
│   └── synthetic_stroke_data.csv
├── notebooks/
│   └── exploratory_analysis.ipynb
├── scripts/
│   ├── ingest_to_kafka.py
│   ├── kafka_stroke_streaming_inference.py
│   ├── train_rf_model.py
│   ├── upload_to_hdfs.py
│   └── requirements.txt
├── README.md
└── .gitignore
```

## Setup

### Prerequisites

- Java (for Hadoop/Spark)
- Hadoop with HDFS configured
- Kafka and Zookeeper running
- Spark with Kafka integration JARs
- Python 3.10+

### Installation

```bash
pip install -r scripts/requirements.txt
```

## Usage

### Start Services

```bash
# Start HDFS
start-dfs.sh

# Start Kafka & Zookeeper
zkServer.sh start
kafka-server-start.sh config/server.properties
```

### Train Model

```bash
spark-submit scripts/train_rf_model.py
```

### Run Streaming Inference

```bash
spark-submit --jars <kafka-jars> scripts/kafka_stroke_streaming_inference.py
```

### Upload Data to HDFS

```bash
python scripts/upload_to_hdfs.py
```

## Data Schema

| Column | Type | Description |
|--------|------|-------------|
| id | int | Patient ID |
| gender | string | Gender |
| age | float | Age |
| hypertension | int | Binary flag |
| heart_disease | int | Binary flag |
| ever_married | string | Yes/No |
| work_type | string | Work type |
| Residence_type | string | Urban/Rural |
| avg_glucose_level | float | Glucose level |
| bmi | float | BMI |
| smoking_status | string | Smoking status |
| stroke | int | Target (0/1) |

## Notes

- Update Kafka bootstrap servers and HDFS paths in config files as needed
- Results are stored as Parquet in `/user/stroke/stroke_predictions` (HDFS)
- Ensure proper permissions for Hadoop directories
