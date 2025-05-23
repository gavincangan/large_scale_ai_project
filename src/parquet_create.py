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

### Create a directory for dummy video files
if not os.path.exists("data/videos"):
    os.makedirs("data/videos")
video_filenames = []
for i in range(N):
    video_path = f"data/videos/video_{i}.mp4"
    video_filenames.append(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, 20.0, (64, 64))
    for j in range(30):  # Create 30 frames per video
        frame = np.random.randint(0, 255, (64, 64, 3)).astype(np.uint8)
        out.write(frame)
    out.release()

### Create robot actions and observations
act_data = []
for i in range(N):
    sampling_rate = 100
    # Ensure at least 1 action, and always 3 action dims
    act_length = max(1, np.random.randint(20, 200))
    acts = np.random.rand(act_length, 3)
    assert acts.shape[1] == 3 and acts.shape[0] >= 1, f"Actions shape error: {acts.shape}"
    timestamp = np.arange(act_length) / sampling_rate
    acts_concat = np.concatenate((timestamp.reshape(-1, 1), acts), axis=1)
    assert acts_concat.shape[1] == 4 and acts_concat.shape[0] == act_length, f"Actions concat shape error: {acts_concat.shape}"
    # Defensive: check for NaN or inf
    if np.isnan(acts_concat).any() or np.isinf(acts_concat).any():
        raise ValueError(f"NaN or inf detected in actions for sample {i}: {acts_concat}")
    # Defensive: check for empty list after tolist()
    acts_list = acts_concat.tolist()
    if not acts_list or not all(isinstance(row, list) and len(row) == 4 for row in acts_list):
        raise ValueError(f"Malformed actions list for sample {i}: {acts_list}")
    act_data.append(acts_list)

obs_data = []
for i in range(N):
    sampling_rate = 50
    # Ensure at least 1 observation, and always 7 obs dims
    obs_length = max(1, np.random.randint(10, 100))
    obs = np.random.rand(obs_length, 7)
    assert obs.shape[1] == 7 and obs.shape[0] >= 1, f"Observations shape error: {obs.shape}"
    timestamp = np.arange(obs_length) / sampling_rate
    obs_concat = np.concatenate((timestamp.reshape(-1, 1), obs), axis=1)
    assert obs_concat.shape[1] == 8 and obs_concat.shape[0] == obs_length, f"Observations concat shape error: {obs_concat.shape}"
    # Defensive: check for NaN or inf
    if np.isnan(obs_concat).any() or np.isinf(obs_concat).any():
        raise ValueError(f"NaN or inf detected in observations for sample {i}: {obs_concat}")
    # Defensive: check for empty list after tolist()
    obs_list = obs_concat.tolist()
    if not obs_list or not all(isinstance(row, list) and len(row) == 8 for row in obs_list):
        raise ValueError(f"Malformed observations list for sample {i}: {obs_list}")
    obs_data.append(obs_list)

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
