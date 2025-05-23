import os
from PIL import Image
import cv2  # OpenCV for video processing
from enum import Enum

import pyarrow as pa
import pyarrow.parquet as pq

import pandas as pd
import numpy as np

from utils import DatasetKeys

N = 10  # Number of samples

### Create a directory with dummy images
if not os.path.exists("data/imgs"):
    os.makedirs("data/imgs")
for i in range(N):
    img = np.random.randint(0, 255, (64, 64, 3)).astype(np.uint8)
    Image.fromarray(img).save(f"data/imgs/img_{i}.png")

### Create dummy video files
video_filenames = []
for i in range(N):
    video_path = f"data/videos/video_{i}.mp4"
    video_filenames.append(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, 20.0, (640, 480))
    for j in range(100):  # Create 100 frames
        frame = np.random.randint(0, 255, (480, 640, 3)).astype(np.uint8)
        out.write(frame)
    out.release()

### Create robot actions and observations
act_data = []
for i in range(N):
    sampling_rate = 100
    act_length = np.random.randint(20, 200)
    timestamp = np.arange(act_length) / sampling_rate
    acts = np.random.rand(act_length, 3)

    act_data.append(np.concatenate((timestamp.reshape(-1, 1), acts), axis=1).tolist())

obs_data = []
for i in range(N):
    sampling_rate = 50
    obs_length = np.random.randint(10, 100)
    timestamp = np.arange(obs_length) / sampling_rate
    obs = np.random.rand(obs_length, 7)

    obs_data.append(np.concatenate((timestamp.reshape(-1, 1), obs), axis=1).tolist())

### Store as Parquet file
df = pd.DataFrame(
    {
        DatasetKeys.TEXT.value: ["This is a test sentence."] * N,
        DatasetKeys.ACTIONS.value: act_data,
        DatasetKeys.OBSERVATIONS.value: obs_data,
        DatasetKeys.IMAGE.value: [f"data/imgs/img_{i}.png" for i in range(N)],
        DatasetKeys.VIDEO.value: video_filenames,  # Add video filenames to the dataset
    }
)

table = pa.Table.from_pandas(df)
data_path = "data/data.pq"
pq.write_table(table, data_path)

loaded_table = pq.read_table(data_path)

print(f"Table created and stored as Parquet file: {data_path}")
