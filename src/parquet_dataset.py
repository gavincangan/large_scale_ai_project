import numpy as np
import pyarrow.parquet as pq
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, IterableDataset
from transformers import AutoTokenizer
import torch
import bisect
from utils import DatasetKeys


class TimeSeriesParquetDataset(Dataset):
    def __init__(
        self,
        parquet_file: Path,
        acts_length_sec: float = 1.0,
        obs_length_sec: float = 1.0,
        device=None,
        window_mode: str = 'start',  # 'start' or 'random'
    ):
        super().__init__()

        self.parquet_file = parquet_file
        self._ds = pq.read_table(parquet_file, memory_map=True)

        self.acts_length_sec = acts_length_sec
        self.obs_length_sec = obs_length_sec
        self.window_mode = window_mode

        # Each column is a list of episodes
        self.text_col = self._ds[DatasetKeys.TEXT.value].to_pylist()
        self.actions_col = self._ds[DatasetKeys.ACTIONS.value].to_pylist()
        self.observations_col = self._ds[DatasetKeys.OBSERVATIONS.value].to_pylist()
        self.image_col = self._ds[DatasetKeys.IMAGE.value].to_pylist() if DatasetKeys.IMAGE.value in self._ds.column_names else None

        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.text_col = self.tokenizer(self.text_col, padding=True, truncation=True, return_tensors="pt")

        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")

    def __len__(self):
        return len(self.actions_col)

    def __getitem__(self, idx: int):
        # Get episode data
        actions = np.array(self.actions_col[idx], dtype=np.float32)
        observations = np.array(self.observations_col[idx], dtype=np.float32)
        text_data = self.text_col['input_ids'][idx]

        # --- Actions window ---
        actions_timestamps = actions[:, 0]
        actions_diffs = np.diff(actions_timestamps)
        actions_diffs = actions_diffs[actions_diffs > 0]
        if len(actions_diffs) > 0:
            actions_freq = np.median(1.0 / actions_diffs)
        else:
            actions_freq = 1.0
        actions_seq_length = int(self.acts_length_sec * actions_freq)
        if self.window_mode == 'random' and len(actions) > actions_seq_length:
            start = np.random.randint(0, len(actions) - actions_seq_length + 1)
        else:
            start = 0
        end = start + actions_seq_length
        actions_window = actions[start:end]
        if len(actions_window) < actions_seq_length:
            actions_window = np.pad(actions_window, ((0, actions_seq_length - len(actions_window)), (0, 0)), mode='constant', constant_values=0)

        # --- Observations window ---
        observations_timestamps = observations[:, 0]
        observations_diffs = np.diff(observations_timestamps)
        observations_diffs = observations_diffs[observations_diffs > 0]
        if len(observations_diffs) > 0:
            observations_freq = np.median(1.0 / observations_diffs)
        else:
            observations_freq = 1.0
        observations_seq_length = int(self.obs_length_sec * observations_freq)
        if self.window_mode == 'random' and len(observations) > observations_seq_length:
            start_obs = np.random.randint(0, len(observations) - observations_seq_length + 1)
        else:
            start_obs = 0
        end_obs = start_obs + observations_seq_length
        observations_window = observations[start_obs:end_obs]
        if len(observations_window) < observations_seq_length:
            observations_window = np.pad(observations_window, ((0, observations_seq_length - len(observations_window)), (0, 0)), mode='constant', constant_values=0)

        # Convert to tensors
        actions_window = torch.tensor(actions_window, dtype=torch.float32)
        observations_window = torch.tensor(observations_window, dtype=torch.float32)
        # text_data is already a tensor

        return {
            DatasetKeys.TEXT.value: text_data,
            DatasetKeys.ACTIONS.value: actions_window,
            DatasetKeys.OBSERVATIONS.value: observations_window,
            DatasetKeys.IMAGE.value: self.image_col[idx] if self.image_col is not None else None,
        }


if __name__ == "__main__":
    parquet_file = Path("data/data_fromh5.pq")
    print(f"Loading dataset from {parquet_file.absolute()}")

    acts_length_sec = 1.0
    obs_length_sec = 1.0

    dataset = TimeSeriesParquetDataset(parquet_file, acts_length_sec, obs_length_sec)

    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    for idx, batch in enumerate(dataloader):
        print(f"Batch {idx}:")
        print(f"\t{batch[DatasetKeys.TEXT.value]}")
        print(f"\t{batch[DatasetKeys.ACTIONS.value].shape}")
        print(f"\t{batch[DatasetKeys.OBSERVATIONS.value].shape}")
