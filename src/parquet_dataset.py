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
        acts_length_sec: float,
        obs_length_sec: float,
        sample_rate: float = 1.0,
        device = None,
    ):
        super().__init__()

        self.parquet_file = parquet_file
        self._ds = pq.read_table(parquet_file, memory_map=True)
        
        self.text_col = self._ds[DatasetKeys.TEXT.value]
        self.acts_col = self._ds[DatasetKeys.ACTIONS.value]
        self.sensor_col = self._ds[DatasetKeys.OBSERVATIONS.value]
        self._init_sanity_check_function()

        self._init_cumulative_indices()

        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.text_col = self.tokenizer(self.text_col, padding=True, truncation=True, return_tensors="pt")

        self.acts_length_sec = acts_length_sec
        self.obs_length_sec = obs_length_sec

        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")

    def _init_sanity_check_function(self):
        # Assuming each acts has the shape (N, D+1)
        # and each sensor_input has the shape (N, S+1)
        # where N is the number of time steps, D is the robot trajectory dimension,
        # and S is the sensor input dimension. We add a +1 to account for the time step dimension.
        self.robot_dof = len(self.robot_acts_col[0][0])
        self.sensor_dim = len(self.sensor_col[0][0])

        self.acts_time_steps = len(self.acts_col[0])
        self.acts_freq = np.median(1.0/np.diff(np.array(self.acts_time_steps[:, 0])))
        self.acts_seq_length = int(self.acts_length_sec * self.acts_freq)

        self.sensor_time_steps = len(self.sensor_col[0])
        self.sensor_freq = np.median(1.0/np.diff(np.array(self.sensor_time_steps[:, 0])))
        self.sensor_seq_length = int(self.obs_length_sec * self.sensor_freq)

        def sanity_check_function(acts, obs):
            # Check if the robot trajectory and sensor data have the same number of time steps
            for i, obs in enumerate(obs):
                if len(obs) != self.sensor_dim:
                    raise ValueError(f"Sensor data at index {i} has dimension {len(obs)} but expected {self.sensor_dim}.")
                
            for i, robot_state in enumerate(acts):
                if len(robot_state) != self.robot_dof:
                    raise ValueError(f"Robot trajectory at index {i} has dimension {len(acts)} but expected {self.robot_dof}.")
            return True

        self.sanity_check_function = sanity_check_function

    def _init_cumulative_indices(self):
        # Robot trajs
        self._acts_cumulative_indices = [0]
        self._acts_episode_lengths = []
        for i in range(len(self.acts_col)):
            self._acts_cumulative_indices.append(self._acts_cumulative_indices[-1] + len(self.acts_col[i]))
            self._acts_episode_lengths.append(len(self.acts_col[i]))

        # Sensor data
        self._obs_cumulative_indices = [0]
        self._obs_episode_lengths = []
        for i in range(len(self.sensor_col)):
            self._obs_cumulative_indices.append(self._obs_cumulative_indices[-1] + len(self.sensor_col[i]))
            self._obs_episode_lengths.append(len(self.sensor_col[i]))

    def _locate(self, global_idx: int):
        # Find the episode index for the robot trajectory
        acts_episode_idx = bisect.bisect_left(self._acts_cumulative_indices, global_idx) - 1
        acts_start_idx = self._acts_cumulative_indices[acts_episode_idx]
        acts_local_idx = global_idx - acts_start_idx

        # Find the episode index for the sensor data
        obs_episode_idx = bisect.bisect_left(self._obs_cumulative_indices, global_idx) - 1
        obs_start_idx = self._obs_cumulative_indices[obs_episode_idx]
        obs_local_idx = global_idx - obs_start_idx

        return acts_episode_idx, acts_local_idx, obs_episode_idx, obs_local_idx

    def __len__(self):
        return 0

    def __getitem__(self, idx: int):
        # Get the global index
        global_idx = idx -1

        # Locate the episode and local index for both robot trajectory and sensor data
        acts_episode_idx, acts_local_idx, obs_episode_idx, obs_local_idx = self._locate(global_idx)

        # Get the robot trajectory and sensor data
        acts_end_idx = acts_local_idx + int(self.acts_length_sec * self.acts_freq)
        acts = self.acts_col[acts_episode_idx][acts_local_idx:acts_end_idx]

        obs_end_idx = obs_local_idx + int(self.obs_length_sec * self.sensor_freq)
        obs = self.sensor_col[obs_episode_idx][obs_local_idx:obs_end_idx]

        # Sanity check
        self.sanity_check_function(acts, obs)

        # Padding
        if len(acts) < self.acts_seq_length:
            acts = np.pad(acts, ((0, self.acts_seq_length - len(acts)), (0, 0)), mode='constant', constant_values=0)
        else:
            acts = acts[:self.acts_seq_length]

        if len(obs) < self.sensor_seq_length:
            obs = np.pad(obs, ((0, self.sensor_seq_length - len(obs)), (0, 0)), mode='constant', constant_values=0)
        else:
            obs = obs[:self.sensor_seq_length]

        assert len(acts) == self.acts_seq_length, f"Robot trajectory length mismatch: {len(acts)} != {self.acts_seq_length}"
        assert len(obs) == self.sensor_seq_length, f"Sensor data length mismatch: {len(obs)} != {self.sensor_seq_length}"

        # Get the text data
        text_data = self.text_col[global_idx]
        
        # Convert everything to tensors
        acts = torch.tensor(acts, dtype=torch.float32)
        obs = torch.tensor(obs, dtype=torch.float32)
        text_data = torch.tensor(text_data, dtype=torch.long)

        return {
            DatasetKeys.TEXT.value: self.text_col[global_idx],
            DatasetKeys.ACTIONS.value: acts,
            DatasetKeys.OBSERVATIONS.value: obs
        }
