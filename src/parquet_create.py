
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


### Create robot actions and observations
act_data = []
for i in range(N):
    sampling_rate = 100
    act_length = np.random.randint(20, 200)
    timestamp = np.arange(act_length) / sampling_rate
    acts = np.random.rand(act_length, 3)

    act_data.append(np.concatenate((timestamp.reshape(-1,1), acts), axis=1).tolist())

obs_data = []
for i in range(N):
    sampling_rate = 50
    obs_length = np.random.randint(10, 100)
    timestamp = np.arange(obs_length) / sampling_rate
    obs = np.random.rand(obs_length, 7)

    obs_data.append(np.concatenate((timestamp.reshape(-1,1), obs), axis=1).tolist())


### Store as Parquet file
df = pd.DataFrame({
    DatasetKeys.TEXT.value: ["This is a test sentence."] * N,
    # First entry of each list is the timestamp.
    DatasetKeys.ACTIONS.value: act_data,
    DatasetKeys.OBSERVATIONS.value: obs_data,

    # Loading images externally
    DatasetKeys.IMAGE.value: [f"data/imgs/img_{i}.png" for i in range(N)],
})

table = pa.Table.from_pandas(df)
data_path = "data/data.pq"
pq.write_table(table, data_path)

loaded_table = pq.read_table(data_path)

print(f"Table created and store as Parquet file: {data_path}")


