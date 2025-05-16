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
        robot_traj_length_sec: float,
        sensor_data_length_sec: float,
        sample_rate: float = 1.0,
        device = None,
    ):
        super().__init__()

        self.parquet_file = parquet_file
        self._ds = pq.read_table(parquet_file, memory_map=True)
        
        self.text_col = self._ds[DatasetKeys.TEXT.value]
        self.robot_trajs_col = self._ds[DatasetKeys.ROBOT_TRAJ.value]
        self.sensor_col = self._ds[DatasetKeys.SENSOR_DATA.value]
        self._init_sanity_check_function()

        self._init_cumulative_indices()

        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.text_col = self.tokenizer(self.text_col, padding=True, truncation=True, return_tensors="pt")

        self.robot_traj_length_sec = robot_traj_length_sec
        self.sensor_data_length_sec = sensor_data_length_sec

        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")

    def _init_sanity_check_function(self):
        # Assuming each robot_traj has the shape (N, D+1)
        # and each sensor_input has the shape (N, S+1)
        # where N is the number of time steps, D is the robot trajectory dimension,
        # and S is the sensor input dimension. We add a +1 to account for the time step dimension.
        self.robot_dof = len(self.robot_actions_col[0][0])
        self.sensor_dim = len(self.sensor_col[0][0])

        self.robot_traj_time_steps = len(self.robot_trajs_col[0])
        self.robot_traj_freq = np.median(1.0/np.diff(np.array(self.robot_traj_time_steps[:, 0])))
        self.robot_traj_seq_length = int(self.robot_traj_length_sec * self.robot_traj_freq)

        self.sensor_time_steps = len(self.sensor_col[0])
        self.sensor_freq = np.median(1.0/np.diff(np.array(self.sensor_time_steps[:, 0])))
        self.sensor_seq_length = int(self.sensor_data_length_sec * self.sensor_freq)

        def sanity_check_function(robot_traj, sensor_data):
            # Check if the robot trajectory and sensor data have the same number of time steps
            for i, sensor_data in enumerate(sensor_data):
                if len(sensor_data) != self.sensor_dim:
                    raise ValueError(f"Sensor data at index {i} has dimension {len(sensor_data)} but expected {self.sensor_dim}.")
                
            for i, robot_state in enumerate(robot_traj):
                if len(robot_state) != self.robot_dof:
                    raise ValueError(f"Robot trajectory at index {i} has dimension {len(robot_traj)} but expected {self.robot_dof}.")
            return True

        self.sanity_check_function = sanity_check_function

    def _init_cumulative_indices(self):
        # Robot trajs
        self._robot_traj_cumulative_indices = [0]
        self._robot_traj_episode_lengths = []
        for i in range(len(self.robot_trajs_col)):
            self._robot_traj_cumulative_indices.append(self._robot_traj_cumulative_indices[-1] + len(self.robot_trajs_col[i]))
            self._robot_traj_episode_lengths.append(len(self.robot_trajs_col[i]))

        # Sensor data
        self._sensor_data_cumulative_indices = [0]
        self._sensor_data_episode_lengths = []
        for i in range(len(self.sensor_col)):
            self._sensor_data_cumulative_indices.append(self._sensor_data_cumulative_indices[-1] + len(self.sensor_col[i]))
            self._sensor_data_episode_lengths.append(len(self.sensor_col[i]))

    def _locate(self, global_idx: int):
        # Find the episode index for the robot trajectory
        robot_traj_episode_idx = bisect.bisect_left(self._robot_traj_cumulative_indices, global_idx) - 1
        robot_traj_start_idx = self._robot_traj_cumulative_indices[robot_traj_episode_idx]
        robot_traj_local_idx = global_idx - robot_traj_start_idx

        # Find the episode index for the sensor data
        sensor_data_episode_idx = bisect.bisect_left(self._sensor_data_cumulative_indices, global_idx) - 1
        sensor_data_start_idx = self._sensor_data_cumulative_indices[sensor_data_episode_idx]
        sensor_data_local_idx = global_idx - sensor_data_start_idx

        return robot_traj_episode_idx, robot_traj_local_idx, sensor_data_episode_idx, sensor_data_local_idx

    def __len__(self):
        return 0

    def __getitem__(self, idx: int):
        # Get the global index
        global_idx = idx -1

        # Locate the episode and local index for both robot trajectory and sensor data
        robot_traj_episode_idx, robot_traj_local_idx, sensor_data_episode_idx, sensor_data_local_idx = self._locate(global_idx)

        # Get the robot trajectory and sensor data
        robot_traj_end_idx = robot_traj_local_idx + int(self.robot_traj_length_sec * self.robot_traj_freq)
        robot_traj = self.robot_trajs_col[robot_traj_episode_idx][robot_traj_local_idx:robot_traj_end_idx]

        sensor_data_end_idx = sensor_data_local_idx + int(self.sensor_data_length_sec * self.sensor_freq)
        sensor_data = self.sensor_col[sensor_data_episode_idx][sensor_data_local_idx:sensor_data_end_idx]

        # Sanity check
        self.sanity_check_function(robot_traj, sensor_data)

        # Padding
        if len(robot_traj) < self.robot_traj_seq_length:
            robot_traj = np.pad(robot_traj, ((0, self.robot_traj_seq_length - len(robot_traj)), (0, 0)), mode='constant', constant_values=0)
        else:
            robot_traj = robot_traj[:self.robot_traj_seq_length]

        if len(sensor_data) < self.sensor_seq_length:
            sensor_data = np.pad(sensor_data, ((0, self.sensor_seq_length - len(sensor_data)), (0, 0)), mode='constant', constant_values=0)
        else:
            sensor_data = sensor_data[:self.sensor_seq_length]

        assert len(robot_traj) == self.robot_traj_seq_length, f"Robot trajectory length mismatch: {len(robot_traj)} != {self.robot_traj_seq_length}"
        assert len(sensor_data) == self.sensor_seq_length, f"Sensor data length mismatch: {len(sensor_data)} != {self.sensor_seq_length}"

        # Get the text data
        text_data = self.text_col[global_idx]
        
        # Convert everything to tensors
        robot_traj = torch.tensor(robot_traj, dtype=torch.float32)
        sensor_data = torch.tensor(sensor_data, dtype=torch.float32)
        text_data = torch.tensor(text_data, dtype=torch.long)

        return {
            DatasetKeys.TEXT.value: self.text_col[global_idx],
            DatasetKeys.ROBOT_TRAJ.value: robot_traj,
            DatasetKeys.SENSOR_DATA.value: sensor_data
        }
