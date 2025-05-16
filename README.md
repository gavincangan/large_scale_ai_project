# Final Project for the Large-scale AI Training Class

## Problem:
Extend Parquet DataLoader from Assignment #2 to be a multimodal data loader with following features:
- Support for video, robot trajectories
- Recover dataloader from checkpoint
- Padding
- Sanity check input output
- Distributed loader

- Data format: folder (database) with files of all types of data. Parquet format can be used to load parts of the data (columns) that we currently need on the rank. All of these are sequences that should align. All videos different length. Allow different sampling rates for sensor data, video, robot traj, ... Dataset has fixed sampling rate, then interpolate all datastreams for desired dataset framerate.
    - Parquet for videos: strings that point to paths for mp4 videos, and metadata on video fps.
    - Parquet for image: string to path, single image for whole trajectory.
    - Parquet for text: Directly file, single text for whole video.
    - Parquet for robot traj: Directly in file, list of values including times of data as index 0. Shape [data_length, N+1].
    - Parquet for sensor data: Directly in file, list of values including times of data. Shape [data_length, N+1].

- ParquetDataset __getitem__: 
    - Loads everything, video loads from file entire length, video metadata contains length of each video sequence, index is first mapped to correct video.
    - Also does some form of interpolation
    - Define offset for sequence loading, index finds certain video and specific sequence inside of video depending on offset. Offset default 1.
    - Allow end of sequence to sample in a dataset, then pad after it has ended with pad tokens.
    - Sampling rate fixed for each modality, user defines number of seconds of data to load for each modality, and depending on the modality sampling rate, that much sequence length is loaded.

- Checkpointing: 
    - Keep track of current dataloader index and rng state, load where left off previous batch.

For each field in the dataset, we need to know
- if it's time series or just one time time thing that will be shared across the trajectory (is there a time component to it)


## Success Metrics
- Able to load a robot dataset with text descriptions, videos and robot trajectories
- Can save dataloader state to a checkpoint and restart a training job
- Padding for variable length data (just zero padding?)
- First prototype, ignore videos, just have all the other in-file parquet loaders.


## Task Distributions
- Barny:
    - ParquetDataset that loads from dummy source.
    - Compute correct sequence length. Includes getting length of all sequences and seeing which getitem index goes into which file.

- Mike:
    - Dummy dataset with external image.
    - Create conversion from h5 to parquet.


## Team
Barnabas Gavin Cangan
Mike Yan Michelis