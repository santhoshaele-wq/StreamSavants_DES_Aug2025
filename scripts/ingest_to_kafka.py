from confluent_kafka import Producer
import csv
import subprocess

hdfs_executable = "C:/hadoop/bin/hdfs.cmd"

# Read CSV from HDFS
hdfs_path = "/user/stroke/synthetic_stroke_data_2.csv"
result = subprocess.run([hdfs_executable, 'dfs', '-cat', hdfs_path], capture_output=True, text=True)
lines = result.stdout.strip().split('\n')

# Kafka producer config
conf = {'bootstrap.servers': 'localhost:9092'}
producer = Producer(conf)

for line in lines[1:]:  # Skip header
    producer.produce('StrokePrediction', value=line)
producer.flush()
print("Pushed line:", line)

