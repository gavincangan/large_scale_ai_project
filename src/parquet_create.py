import pyarrow as pa
import pyarrow.parquet as pq

import pandas as pd
import numpy as np
import os

from enum import Enum


from utils import DatasetKeys


N = 10
df = pd.DataFrame({
    DatasetKeys.TEXT.value: ["This is a test sentence."] * N,
    # DatasetKeys.IMAGE.value: [np.random.rand(3, 64, 64).astype(np.float32)] * N,
    DatasetKeys.ROBOT_TRAJ.value: [np.random.rand(100, 3).astype(np.float32).tolist()] * N,
    DatasetKeys.SENSOR_DATA.value: [np.random.rand(100, 3).astype(np.float32).tolist()] * N,
})

table = pa.Table.from_pandas(df)
pq.write_table(table, "test.pq")

new_table = pq.read_table("test.pq")


breakpoint()

