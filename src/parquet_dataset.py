import pyarrow.parquet as pq
from torch.utils.data import Dataset, DataLoader, IterableDataset
from transformers import AutoTokenizer
import torch
from enum import Enum


class TimeSeriesParquetDataset(Dataset):
    def __init__(self, parquet_file: str, sequence_length: int, sample_rate: int = 1):
        self.parquet_ds = pq.read_table(parquet_file, memory_map=True)
        self.real_length = len(self.parquet_ds)
        self.sequence_length = sequence_length
        self.sample_rate = sample_rate

    def __len__(self):
        return 0

    def __getitem__(self, idx: int):
        pass