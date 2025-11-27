import os

# Local path to your dataset
local_path = "C:/stroke-prediction/data/synthetic_stroke_data_2.csv"

# HDFS destination directory
hdfs_dir = "/user/stroke/"

# Hadoop command to upload the dataset
command = f"hdfs dfs -put {local_path} {hdfs_dir}"

# Execute the command
os.system(command)