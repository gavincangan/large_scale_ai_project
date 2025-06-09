# Multimodal Parquet Dataset

This project demonstrates a simple multimodal dataset used in the Large-scale AI Training class.  The dataset stores text, robot trajectories, observations, images and small video clips inside a single Parquet table.  `TimeSeriesParquetDataset` in `src/parquet_dataset.py` can load windows of this data and returns PyTorch tensors ready for training.

## Setup

1. Create a Python environment (Python 3.8+).
2. Install the required packages:
   ```bash
   pip install numpy pandas pyarrow h5py pillow opencv-python torch transformers
   ```
3. (Optional) Generate a dummy dataset for experimentation:
   ```bash
   python src/parquet_create.py
   ```
   This produces `data/data.pq` and a folder of random videos under `data/videos/`.
4. (Optional) Convert your own HDF5 logs to the same format:
   ```bash
   python src/parquet_fromh5.py
   ```
   The script expects filenames listed inside `src/parquet_fromh5.py` and writes `data/data_fromh5.pq`.

## Usage

`src/parquet_dataset.py` provides `TimeSeriesParquetDataset`.  The simplest way to try it is to execute the file directly:

```bash
python src/parquet_dataset.py
```

This loads `data/data.pq`, creates a `DataLoader`, and iterates through a few batches while saving the loader state to `data/dataloader_ckpt.pkl`.  The dataset automatically resumes from this checkpoint if it exists.

When constructing the dataset manually you can specify:

```python
from pathlib import Path
from src.parquet_dataset import TimeSeriesParquetDataset

parquet_file = Path("data/data.pq")
# length of each returned window in seconds
acts_length_sec = 0.1
obs_length_sec = 0.1

# offsets of video frames relative to the first action timestamp
video_frame_offsets = [-1, 0, 1]

dataset = TimeSeriesParquetDataset(
    parquet_file,
    acts_length_sec,
    obs_length_sec,
    video_frame_offsets=video_frame_offsets,
    checkpoint_path="data/dataloader_ckpt.pkl",
)
```

A dictionary containing tokenised text, robot actions, observations and a tensor of video frames is returned for each index.

## Features

- **Multimodal support**: text, images, robot trajectories, observations and sampled video frames are stored in a single Parquet file.
- **Windowed access**: returns fixed length windows of actions and observations with zero padding when sequences end.
- **Automatic device selection**: tensors are moved to CPU, CUDA or MPS based on availability.
- **Text tokenisation** using `transformers` BERT tokenizer.
- **Checkpointing**: dataloader index and RNG state are saved/restored so training jobs can resume.
- **Dummy data generation** scripts and HDF5 conversion utilities.

## Limitations and TODOs

- The `sample_rate` argument is kept for compatibility but not yet used.
- Interpolation or resampling between different sensor frequencies is not implemented.
- No distributed dataloader example is provided.
- Video loading relies on OpenCV and reads entire frames from disk; more efficient decoders could be used.
- Error handling is basic and the current dataset format is minimal.

## Team

Barnabas Gavin Cangan
Mike Yan Michelis
