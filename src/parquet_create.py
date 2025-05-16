
import os
from PIL import Image
from enum import Enum

import pyarrow as pa
import pyarrow.parquet as pq

import pandas as pd
import numpy as np

from utils import DatasetKeys



N = 10 # Number of samples

### Create a directory with dummy images
if not os.path.exists("data/imgs"):
    os.makedirs("data/imgs")
for i in range(N):
    img = np.random.randint(0, 255, (64, 64, 3)).astype(np.uint8)
    Image.fromarray(img).save(f"data/imgs/img_{i}.png")


### Create robot trajectory and sensor data
traj_data = []
for i in range(N):
    sampling_rate = 150
    traj_length = np.random.randint(50, 200)
    timestamp = np.arange(traj_length) / sampling_rate
    traj = np.random.rand(traj_length, 3)

    traj_data.append(np.concatenate((timestamp.reshape(-1,1), traj), axis=1).tolist())

sensor_data = []
for i in range(N):
    sampling_rate = 50
    traj_length = np.random.randint(10, 100)
    timestamp = np.arange(traj_length) / sampling_rate
    sensor = np.random.rand(traj_length, 7)

    sensor_data.append(np.concatenate((timestamp.reshape(-1,1), sensor), axis=1).tolist())


### Store as Parquet file
df = pd.DataFrame({
    DatasetKeys.TEXT.value: ["This is a test sentence."] * N,
    # First entry of each list is the timestamp.
    DatasetKeys.ROBOT_TRAJ.value: traj_data,
    DatasetKeys.SENSOR_DATA.value: sensor_data,

    # Loading images externally
    DatasetKeys.IMAGE.value: [f"data/imgs/img_{i}.png" for i in range(N)],
})

table = pa.Table.from_pandas(df)
pq.write_table(table, "data/data.pq")

loaded_table = pq.read_table("data/data.pq")



