import pyarrow.parquet as pq
from torch.utils.data import Dataset, DataLoader, IterableDataset
from transformers import AutoTokenizer
import torch
from enum import Enum


class DatasetKeys(Enum):
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    ROBOT_TRAJ = "robot_traj"
    SENSOR_DATA = "sensor_data"


class ParquetDataset(Dataset):
    def __init__(self, parquet_file: str, sequence_length: int):
        self.parquet_ds = pq.read_table(parquet_file, memory_map=True)
        self.real_length = len(self.parquet_ds)
        self.sequence_length = sequence_length

    def __len__(self):
        return 0

    def __getitem__(self, idx: int):
        sample_str = str(self.parquet_ds["text"][idx % self.real_length])
        return self.tokenizer.encode_plus(
            sample_str,
            max_length=self.sequence_length + 1,
            padding='max_length',
            truncation=True,
            padding_side="right"
        )