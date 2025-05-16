
import os
import h5py

import numpy as np
import pandas as pd

from enum import Enum
from PIL import Image
import pyarrow as pa
import pyarrow.parquet as pq

from utils import DatasetKeys


txt_data = []
act_data = []
obs_data = []

filenames = ["20250325_170840.h5"]
for filename in filenames:
    with h5py.File(filename, "r") as f:
        # Time in nanoseconds should be stored in seconds
        actions = []
        for t in f['hand']['right']['policy_output']:
            a = f['hand']['right']['policy_output'][t]
            actions.append(np.concatenate([[int(t)/1e9], a]))
        actions = np.array(actions)

        observations = []
        for t in f['right']['franka']['end_effector_pose']:
            o = np.array(f['right']['franka']['end_effector_pose'][t])
            observations.append(np.concatenate([[int(t)/1e9], o]))
        observations = np.array(observations)

        # Temporary description
        text = list(f['task_description'])[0]

        # for t in f['oakd_front_view']['color']:
        #     img = np.array(f['oakd_front_view']['color'][t])
        # Not storing individual frames for now.

    txt_data.append(text)
    act_data.append(actions.tolist())
    obs_data.append(observations.tolist())


### Store as Parquet file
df = pd.DataFrame({
    DatasetKeys.TEXT.value: txt_data,
    # First entry of each list is the timestamp.
    DatasetKeys.ACTIONS.value: act_data,
    DatasetKeys.OBSERVATIONS.value: obs_data,

    # Loading images externally
    # DatasetKeys.IMAGE.value: [f"data/imgs/img_{i}.png" for i in range(N)],
})

table = pa.Table.from_pandas(df)
data_path = "data/data_fromh5.pq"
pq.write_table(table, data_path)

loaded_table = pq.read_table(data_path)

print(f"Table created and store as Parquet file: {data_path}")




