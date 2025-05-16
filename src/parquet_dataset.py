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
        sampling_rate: int = 1,
        acts_length_sec: float = 1.0,
        obs_length_sec: float = 1.0,
        device = None,
    ):
        super().__init__()

        self.parquet_file = parquet_file
        self._ds = pq.read_table(parquet_file, memory_map=True)

        self.sampling_rate = sampling_rate
        self.acts_length_sec = acts_length_sec
        self.obs_length_sec = obs_length_sec

        self.text_col = self._ds[DatasetKeys.TEXT.value]
        self.actions_col = self._ds[DatasetKeys.ACTIONS.value]
        self.observations_col = self._ds[DatasetKeys.OBSERVATIONS.value]
        self._init_sanity_check_function()

        self._init_cumulative_indices()

        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

        self.text_col = self.tokenizer(self.text_col.to_pylist(), padding=True, truncation=True, return_tensors="pt")

        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")

    def _init_sanity_check_function(self):
        # Assuming each actions has the shape (N, D+1)
        # and each observations has the shape (N, S+1)
        # where N is the number of time steps, D is the robot trajectory dimension,
        # and S is the sensor input dimension. We add a +1 to account for the time step dimension.
        self.robot_dof = len(self.actions_col[0][0])
        self.observation_dim = len(self.observations_col[0][0])

        actions_first_np = np.array(self.actions_col[0].as_py(), dtype=np.float32) # from the first episode
        actions_time_steps = actions_first_np[:, 0]  # time steps are in the first column
        self.actions_freq = np.median(1.0/np.diff(actions_time_steps))
        self.actions_seq_length = int(self.acts_length_sec * self.actions_freq)

        observations_first_np = np.array(self.observations_col[0].as_py(), dtype=np.float32) # from the first episode
        observations_time_steps = observations_first_np[:, 0] # time steps are in the first column
        self.observations_freq = np.median(1.0/np.diff(np.array(observations_time_steps)))
        self.observations_seq_length = int(self.obs_length_sec * self.observations_freq)

        def sanity_check_function(actions, observations):
            # Check if the robot trajectory and sensor data have the same number of time steps
            for i, obs in enumerate(observations):
                if len(obs) != self.observation_dim:
                    raise ValueError(f"Observation data at index {i} has dimension {len(obs)} but expected {self.observation_dim}.")
                
            for i, robot_state in enumerate(actions):
                if len(robot_state) != self.robot_dof:
                    raise ValueError(f"Action at index {i} has dimension {len(actions)} but expected {self.robot_dof}.")
            return True

        self.sanity_check_function = sanity_check_function

    def _init_cumulative_indices(self):
        # Actions
        self._actions_episode_lengths = np.array([len(actions) for actions in self.actions_col])
        self._actions_cumulative_indices = np.cumsum(self._actions_episode_lengths)
        self._actions_cumulative_indices = np.insert(self._actions_cumulative_indices, 0, 0)

        # Observations
        self._observations_episode_lengths = np.array([len(observations) for observations in self.observations_col])
        self._observations_cumulative_indices = np.cumsum(self._observations_episode_lengths)
        self._observations_cumulative_indices = np.insert(self._observations_cumulative_indices, 0, 0)

    def _locate_action(self, global_idx: int):
        # Find the episode index for the actions
        actions_episode_idx = bisect.bisect_left(self._actions_cumulative_indices, global_idx) - 1
        
        actions_episode_idx = min(actions_episode_idx, len(self._actions_episode_lengths) - 1)
        if actions_episode_idx < 0:
            actions_episode_idx = 0
        
        actions_start_idx = self._actions_cumulative_indices[actions_episode_idx]
        actions_local_idx = global_idx - actions_start_idx
        if actions_local_idx < 0:
            actions_local_idx = 0

        return actions_episode_idx, actions_local_idx

    def _locate_observation(self, global_idx: int):
        # Find the episode index for the observations
        observations_episode_idx = bisect.bisect_left(self._observations_cumulative_indices, global_idx) - 1
        
        observations_episode_idx = min(observations_episode_idx, len(self._observations_episode_lengths) - 1)
        if observations_episode_idx < 0:
            observations_episode_idx = 0
        
        observations_start_idx = self._observations_cumulative_indices[observations_episode_idx]
        observations_local_idx = global_idx - observations_start_idx

        return observations_episode_idx, observations_local_idx

    def __len__(self):
        return min(self._actions_cumulative_indices[-1], self._observations_cumulative_indices[-1]) - 1

    def __getitem__(self, idx: int):
        # Get the global index
        global_time = idx * self.sampling_rate

        # Locate the episode and local index for both actions and observations
        actions_global_idx = self.actions_freq * global_time
        actions_episode_idx, actions_local_idx = self._locate_action(actions_global_idx)

        observations_global_idx = self.observations_freq * global_time
        observation_episode_idx, observations_local_idx = self._locate_observation(observations_global_idx)

        # Sanity check
        if actions_episode_idx != observation_episode_idx:
            raise ValueError(f"Actions and observations are not from the same episode: {actions_episode_idx} != {observation_episode_idx}")
        episode_idx = actions_episode_idx

        # Get the actions and observations
        actions_end_idx = actions_local_idx + int(self.acts_length_sec * self.actions_freq)
        action_episode_np = np.array(self.actions_col[episode_idx].as_py(), dtype=np.float32)
        actions = action_episode_np[actions_local_idx:actions_end_idx]

        observations_end_idx = observations_local_idx + int(self.obs_length_sec * self.observations_freq)
        observation_episode_np = np.array(self.observations_col[episode_idx].as_py(), dtype=np.float32)
        observations = observation_episode_np[observations_local_idx:observations_end_idx]

        # Sanity check
        self.sanity_check_function(actions, observations)

        # Padding
        if len(actions) < self.actions_seq_length:
            actions = np.pad(actions, ((0, self.actions_seq_length - len(actions)), (0, 0)), mode='constant', constant_values=0)
        else:
            actions = actions[:self.actions_seq_length]

        if len(observations) < self.observations_seq_length:
            observations = np.pad(observations, ((0, self.observations_seq_length - len(observations)), (0, 0)), mode='constant', constant_values=0)
        else:
            observations = observations[:self.observations_seq_length]

        assert len(actions) == self.actions_seq_length, f"Actions length mismatch: {len(actions)} != {self.actions_seq_length}"
        assert len(observations) == self.observations_seq_length, f"Observations length mismatch: {len(observations)} != {self.observations_seq_length}"

        # Get the text data
        text_data = self.text_col[episode_idx]
        
        # Convert everything to tensors
        actions = torch.tensor(actions, dtype=torch.float32)
        observations = torch.tensor(observations, dtype=torch.float32)
        text_data = torch.tensor(text_data, dtype=torch.long)

        return {
            DatasetKeys.TEXT.value: self.text_col[global_idx],
            DatasetKeys.ACTIONS.value: actions,
            DatasetKeys.OBSERVATIONS.value: observations
        }


if __name__ == "__main__":
    parquet_file = Path("data/data.pq")
    print(f"Loading dataset from {parquet_file.absolute()}")

    acts_length_sec = 1.0
    obs_length_sec = 1.0

    dataset = TimeSeriesParquetDataset(parquet_file, acts_length_sec, obs_length_sec)

    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    for batch in dataloader:
        print(batch[DatasetKeys.TEXT.value])
        print(batch[DatasetKeys.ACTIONS.value].shape)
        print(batch[DatasetKeys.OBSERVATIONS.value].shape)
        break
