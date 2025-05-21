import numpy as np
import pyarrow.parquet as pq
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, IterableDataset
from transformers import AutoTokenizer
import torch
import bisect
from utils import DatasetKeys
import pickle
import os


class TimeSeriesParquetDataset(Dataset):
    def __init__(
        self,
        parquet_file: Path,
        acts_length_sec: float,
        obs_length_sec: float,
        sample_rate: float = 1.0,
        device=None,
        checkpoint_path: str = None,  # Path to save/load checkpoint
    ):
        super().__init__()

        self.parquet_file = parquet_file
        self.acts_length_sec = acts_length_sec
        self.obs_length_sec = obs_length_sec
        self._ds = pq.read_table(parquet_file, memory_map=True)

        self.text_col = self._ds[DatasetKeys.TEXT.value].to_pylist()
        self.acts_col = self._ds[DatasetKeys.ACTIONS.value]
        self.obs_col = self._ds[DatasetKeys.OBSERVATIONS.value]
        self._init_sanity_check_function()

        self._init_cumulative_indices()

        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.text_col = self.tokenizer(
            self.text_col, padding=True, truncation=True, return_tensors="pt"
        )

        self.checkpoint_path = checkpoint_path
        self._dataloader_index = 0
        self._rng_state = None

        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")

        if checkpoint_path is not None and os.path.exists(checkpoint_path):
            self.load_checkpoint(checkpoint_path)

    def _init_sanity_check_function(self):
        # Assuming each acts has the shape (N, D+1)
        # and each obs_input has the shape (N, S+1)
        # where N is the number of time steps, D is the robot trajectory dimension,
        # and S is the obs input dimension. We add a +1 to account for the time step dimension.
        self.robot_dof = len(self.acts_col[0][0])
        self.obs_dim = len(self.obs_col[0][0])

        # Convert pyarrow scalars to numpy float arrays for timestamp columns
        self.acts_time_steps = np.array([float(x[0].as_py() if hasattr(x[0], 'as_py') else x[0]) for x in self.acts_col[0]])
        acts_diffs = np.diff(self.acts_time_steps)
        acts_diffs = acts_diffs[acts_diffs > 0]
        if len(acts_diffs) > 0:
            self.acts_freq = np.median(1.0 / acts_diffs)
        else:
            self.acts_freq = 1.0
        self.acts_seq_length = int(self.acts_length_sec * self.acts_freq)

        self.obs_time_steps = np.array([float(x[0].as_py() if hasattr(x[0], 'as_py') else x[0]) for x in self.obs_col[0]])
        obs_diffs = np.diff(self.obs_time_steps)
        obs_diffs = obs_diffs[obs_diffs > 0]
        if len(obs_diffs) > 0:
            self.obs_freq = np.median(1.0 / obs_diffs)
        else:
            self.obs_freq = 1.0
        self.obs_seq_length = int(self.obs_length_sec * self.obs_freq)

        def sanity_check_function(acts, obs):
            # Check if the robot trajectory and obs data have the same number of time steps
            for i, obs in enumerate(obs):
                if len(obs) != self.obs_dim:
                    raise ValueError(
                        f"Observation data at index {i} has dimension {len(obs)} but expected {self.obs_dim}."
                    )

            for i, robot_state in enumerate(acts):
                if len(robot_state) != self.robot_dof:
                    raise ValueError(
                        f"Robot trajectory at index {i} has dimension {len(acts)} but expected {self.robot_dof}."
                    )
            return True

        self.sanity_check_function = sanity_check_function

    def _init_cumulative_indices(self):
        # Robot trajs
        self._acts_cumulative_indices = [0]
        self._acts_episode_lengths = []
        for i in range(len(self.acts_col)):
            self._acts_cumulative_indices.append(
                self._acts_cumulative_indices[-1] + len(self.acts_col[i])
            )
            self._acts_episode_lengths.append(len(self.acts_col[i]))

        # Observation data
        self._obs_cumulative_indices = [0]
        self._obs_episode_lengths = []
        for i in range(len(self.obs_col)):
            self._obs_cumulative_indices.append(
                self._obs_cumulative_indices[-1] + len(self.obs_col[i])
            )
            self._obs_episode_lengths.append(len(self.obs_col[i]))

    def _locate(self, global_idx: int):
        # Find the episode index for the robot trajectory
        acts_episode_idx = (
            bisect.bisect_left(self._acts_cumulative_indices, global_idx) - 1
        )
        acts_start_idx = self._acts_cumulative_indices[acts_episode_idx]
        acts_local_idx = global_idx - acts_start_idx

        # Find the episode index for the observation data
        obs_episode_idx = (
            bisect.bisect_left(self._obs_cumulative_indices, global_idx) - 1
        )
        obs_start_idx = self._obs_cumulative_indices[obs_episode_idx]
        obs_local_idx = global_idx - obs_start_idx

        return acts_episode_idx, acts_local_idx, obs_episode_idx, obs_local_idx

    def __len__(self):
        # Return the total number of possible windows (episodes) in the dataset
        return len(self.acts_col)

    def __getitem__(self, idx: int):
        # Get the global index
        global_idx = idx - 1

        # Locate the episode and local index for both robot trajectory and observation data
        acts_episode_idx, acts_local_idx, obs_episode_idx, obs_local_idx = self._locate(global_idx)

        # Convert pyarrow ListScalar to python list for slicing (ALWAYS convert, not just if hasattr)
        acts_episode = self.acts_col[acts_episode_idx]
        try:
            acts_episode = acts_episode.to_pylist()
        except AttributeError:
            acts_episode = list(acts_episode)
        acts_end_idx = acts_local_idx + int(self.acts_length_sec * self.acts_freq)
        acts = acts_episode[acts_local_idx:acts_end_idx]

        obs_episode = self.obs_col[obs_episode_idx]
        try:
            obs_episode = obs_episode.to_pylist()
        except AttributeError:
            obs_episode = list(obs_episode)
        obs_end_idx = obs_local_idx + int(self.obs_length_sec * self.obs_freq)
        obs = obs_episode[obs_local_idx:obs_end_idx]

        # Sanity check
        self.sanity_check_function(acts, obs)

        # Padding
        acts = np.array(acts, dtype=np.float32)
        obs = np.array(obs, dtype=np.float32)
        if acts.ndim == 1:
            acts = acts[None, :]
        if obs.ndim == 1:
            obs = obs[None, :]
        if len(acts) < self.acts_seq_length:
            acts = np.pad(
                acts,
                ((0, self.acts_seq_length - len(acts)), (0, 0)),
                mode="constant",
                constant_values=0,
            )
        else:
            acts = acts[: self.acts_seq_length]

        if len(obs) < self.obs_seq_length:
            obs = np.pad(
                obs,
                ((0, self.obs_seq_length - len(obs)), (0, 0)),
                mode="constant",
                constant_values=0,
            )
        else:
            obs = obs[: self.obs_seq_length]

        assert (
            len(acts) == self.acts_seq_length
        ), f"Robot trajectory length mismatch: {len(acts)} != {self.acts_seq_length}"
        assert (
            len(obs) == self.obs_seq_length
        ), f"Observation data length mismatch: {len(obs)} != {self.obs_seq_length}"

        # Get the text data
        text_data = {k: v[global_idx] for k, v in self.text_col.items()}

        # Convert everything to tensors
        acts = torch.tensor(acts, dtype=torch.float32)
        obs = torch.tensor(obs, dtype=torch.float32)
        # text_data is already a tensor or dict of tensors

        return {
            DatasetKeys.TEXT.value: text_data,
            DatasetKeys.ACTIONS.value: acts,
            DatasetKeys.OBSERVATIONS.value: obs,
        }

    def save_checkpoint(self, checkpoint_path=None, dataloader_index=None):
        """Save dataloader index and RNG state to checkpoint_path."""
        if checkpoint_path is None:
            checkpoint_path = self.checkpoint_path
        if dataloader_index is None:
            dataloader_index = self._dataloader_index
        state = {
            "dataloader_index": dataloader_index,
            "rng_state": np.random.get_state(),
            "torch_rng_state": torch.get_rng_state().cpu().numpy().tolist(),
        }
        with open(checkpoint_path, "wb") as f:
            pickle.dump(state, f)
        print(f"Checkpoint saved to {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path=None):
        """Restore dataloader index and RNG state from checkpoint_path."""
        if checkpoint_path is None:
            checkpoint_path = self.checkpoint_path
        print(f"Loading checkpoint from {checkpoint_path}")
        with open(checkpoint_path, "rb") as f:
            state = pickle.load(f)
        self._dataloader_index = state["dataloader_index"]
        np.random.set_state(state["rng_state"])
        torch.set_rng_state(torch.tensor(state["torch_rng_state"], dtype=torch.uint8))

    def set_dataloader_index(self, idx):
        self._dataloader_index = idx

    def get_dataloader_index(self):
        return self._dataloader_index


if __name__ == "__main__":
    # Example usage for checkpointing
    parquet_file = Path("data/data_fromh5.pq")
    acts_length_sec = 0.1
    obs_length_sec = 0.1

    checkpoint_path = "data/dataloader_ckpt.pkl"

    dataset = TimeSeriesParquetDataset(
        parquet_file, acts_length_sec, obs_length_sec, checkpoint_path=checkpoint_path
    )
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    start_idx = dataset.get_dataloader_index() if dataset.get_dataloader_index() else 0
    for idx, batch in enumerate(dataloader, start=start_idx):
        dataset.set_dataloader_index(idx)
        # do something here with the batch
        print(f"Batch {idx}: {batch}")
        # save checkpoint every N batches
        dataset.save_checkpoint()
