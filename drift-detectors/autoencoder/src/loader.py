import torch
import random
import multiprocessing
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
import glob
import mne

from . import utils

logger = utils.get_logger(level='DEBUG')

def get_boas_data(base_path, output_path):
    """
    Retrieve and combine EEG and event data for subjects from the specified Bitbrain dataset folder.

    :param base_path: path to the base directory containing subject folders
    :param output_path: path to the directory where combined data will be saved
    """
    for subject_folder in glob.glob(os.path.join(base_path, 'sub-*')):
        subject_id = os.path.basename(subject_folder)
        eeg_folder = os.path.join(subject_folder, 'eeg')

        output_file = os.path.join(output_path, f'{subject_id}.csv')

        if os.path.exists(output_file):
            continue

        if not os.path.exists(eeg_folder):
            print(f"No EEG folder found for {subject_id}. Skipping.")
            continue

        eeg_file_pattern = os.path.join(eeg_folder, f'{subject_id}_task-Sleep_acq-headband_eeg.edf')
        events_file_pattern = os.path.join(eeg_folder, f'{subject_id}_task-Sleep_acq-psg_events.tsv')

        try:
            raw = mne.io.read_raw_edf(eeg_file_pattern, preload=True)
            x_data = raw.to_data_frame()
        except Exception as e:
            print(f"Error loading EEG data for {subject_id}: {e}")
            continue

        try:
            y_data = pd.read_csv(events_file_pattern, delimiter='\t')
        except Exception as e:
            print(f"Error loading events data for {subject_id}: {e}")
            continue

        combined_data = pd.concat([x_data, y_data], axis=1)

        combined_data.to_csv(output_file, index=False)
        print(f"Saved combined data for {subject_id} to {output_file}")

class TSDataset(Dataset):
    def __init__(self, df, seq_len, X, t, y, per_epoch=True):
        """
        Initializes a time series dataset.

        :param df: dataframe
        :param seq_len: length of the input sequence
        :param X: list of feature columns
        :param t: list of time columns
        :param y: list of target columns
        :param per_epoch: whether to create sequences with overlapping epochs or not
        """
        self.seq_len = seq_len
        self.X = pd.concat([df[X], df[t]], axis=1)
        self.y = df[y]
        self.per_epoch = per_epoch

        logger.info(f"Initializing dataset with: samples={self.num_samples}, samples/seq={seq_len}, seqs={self.num_seqs}, epochs={self.num_epochs} ")

    def __len__(self):
        """
        :return: length of the dataset
        """
        return self.num_seqs

    def __getitem__(self, idx):
        """
        Retrieves a sample from the dataset at the specified index.

        :param idx: index of the sample
        :return: tuple of features and target tensors
        """
        if self.per_epoch:
            start_idx = idx * self.seq_len
        else:
            start_idx = idx

        end_idx = start_idx + self.seq_len

        X = self.X.iloc[start_idx:end_idx].values
        y = self.y.iloc[start_idx:end_idx].values

        X, y = torch.FloatTensor(X), torch.LongTensor(y)

        return X, y
    
    @property
    def num_samples(self):
        return self.X.shape[0]
    
    @property
    def num_epochs(self):
        return self.num_samples // 7680

    @property
    def max_seq_id(self):
        """
        :return: maximum index for a sequence
        """
        return self.num_samples - self.seq_len
    
    @property
    def num_seqs(self):
        """
        :return: number of sequences that can be created from the dataset
        """
        if self.per_epoch:
            return self.num_samples // self.seq_len
        else:
            return self.max_seq_id + 1

def split_data(dir, train_size=57, val_size=1, test_size=1):
    """
    Split the csv files into training, validation, and test sets.

    :param dir: directory containing the csv files
    :param train_size: number of files for training
    :param val_size: number of files for validation
    :param test_size: number of files for testing
    :return: tuple of lists containing csv file paths for train, val, and test sets
    """
    paths = [utils.get_path(dir, filename=file) for file in os.listdir(dir)]
    logger.info(f"Found {len(paths)} files in directory: {dir} ready for splitting.")

    train_paths = paths[:train_size]
    val_paths = paths[train_size:train_size + val_size]
    test_paths = paths[train_size + val_size:train_size + val_size + test_size]

    logger.info(f"Splitting complete!")

    return (train_paths, val_paths, test_paths)

def load_file(path):
    """
    Load data from a .csv file.

    :param path: path to the .csv file
    :return: tuple (X, y)
    """
    df = pd.read_csv(path)

    X = df[['HB_1', 'HB_2']].values
    y = df['majority'].values

    return X, y

def get_fs(path):
    """
    Get the sampling frequency (fs) from a randomly selected csv file in the specified directory.

    :param path: path to the directory containing csv files
    :return: sampling frequency (fs) in Hz
    """
    csv_files = [f for f in os.listdir(path) if f.endswith('.csv')]
    selected_file = os.path.join(path, random.choice(csv_files))

    data = pd.read_csv(selected_file)
    time = data['time']

    time_diffs = time.diff().dropna()
    avg_time_diff = time_diffs.mean()

    fs = 1 / avg_time_diff
    print(f"Sampling frequency: {fs:.2f} Hz")

    return fs

def combine_data(paths, samples=7680, seq_len=240):
    """
    Combine data from multiple npz files into a dataframe.

    :param paths: list of file paths to npz files
    :param samples: int
    :param seq_len: int
    :return: dataframe
    """
    dataframes = []

    logger.info(f"Combining data from {len(paths)} files.")

    for path in paths:
        X, y = load_file(path)

        df = pd.DataFrame(X, columns=['HB_1', 'HB_2'])
        df['Majority'] = y
        df['Time'] = (np.arange(len(df)) % samples) + 1
        df['ID'] = (df['Time'] - 1) // seq_len + 1

        dataframes.append(df)

    df = pd.concat(dataframes, ignore_index=True)
    logger.info(f"Combined dataframe shape: {df.shape}")

    rows_before_majority_drop = df.shape[0]
    df = df[df['Majority'] != -1]
    logger.info(f"Removed {rows_before_majority_drop - df.shape[0]} rows with majority value -1.")
    
    rows_before_nan_drop = df.shape[0]
    df = df.dropna()
    logger.info(f"Removed {rows_before_nan_drop - df.shape[0]} rows with NaN values.")

    assert not df.isna().any().any(), "NaN values found in the dataframe!"

    df = utils.robust_normalize(df, exclude=['Majority', 'Time', 'ID'])

    return df

def get_dataframes(paths, rate=240, samples=7680, seq_len=240, exist=False):
    """
    Create or load dataframes for training, validation, and testing.

    :param paths: list of training, validation and test file paths
    :param exist: whether dataframe csvs already exist
    :return: tuple of dataframes
    """
    dataframes = []
    names = ['train', 'val', 'test']

    logger.info("Creating dataframes for training, validation, and testing.")

    for paths, name in zip(paths, names):
        proc_path = utils.get_path('data', 'proc', filename=f"{name}.csv")

        if exist:
            df = pd.read_csv(proc_path)
            logger.info(f"Loaded existing dataframe from {proc_path}.")
        else:
            df = combine_data(paths, samples, seq_len)
            df.to_csv(proc_path, index=False)
            logger.info(f"Saved new dataframe to {proc_path}.")

        dataframes.append(df)

    logger.info("Dataframes for training, validation, and testing are ready!")

    return tuple(dataframes)

def extract_weights(df, label_col):
    """
    Calculate class weights from the training dataframe to handle class imbalance.

    :param df: dataframe containing the training data
    :param label_col: the name of the column containing class labels
    :return: dictionary
    """
    logger.info("Calculating class weights from the training dataframe.")

    occs = df[label_col].value_counts().to_dict()
    inverse_occs = {key: 1e-10 for key in range(5)}

    for key, value in occs.items():
        inverse_occs[int(key)] = 1 / (value + 1e-10)

    weights = {key: value / sum(inverse_occs.values()) for key, value in inverse_occs.items()}
    weights = dict(sorted(weights.items()))

    path = utils.get_path('data', filename='weights.json')
    utils.save_json(data=weights, filename=path)

    return weights

def create_datasets(dataframes, seq_len=7680):
    """
    Create datasets for the specified dataframes, e.g. training, validation and testing, or a subset of those.

    :param dataframes: tuple of dataframes
    :param seq_len: length of the input sequence
    :return: tuple of datasets
    """

    datasets = []
    X, t, y = ["HB_1", "HB_2"], ["Time", "ID"], ["Majority"]

    logger.info("Creating datasets from dataframes.") 

    for df in dataframes:
        dataset = TSDataset(df, seq_len, X, t, y)
        datasets.append(dataset)

    logger.info(f"Datasets created successfully!")

    return tuple(datasets)

def create_dataloaders(datasets, batch_size=1, shuffle=[True, False, False], num_workers=None, drop_last=False):
    """
    Create dataloaders for the specified datasets, e.g. training, validation and testing, or a subset of those.

    The batch size should be 1, 2, 4, 8, 16, or 32, as these values ensure that all sequences are distributed 
    across batches (batch_size must divide evenly: num_seqs = 7680 * epochs / 240 = 32 * epochs). This prevents 
    the number of sequences in the last batch from being smaller than the batch size.

    The number of batches is determined by num_seqs / batch_size. Additionally, aggr = 32 should ideally divide 
    batches evenly: (7680 * epochs) / (240 * batch_size * 32). Therefore, the batch size must be 1 if we want the 
    transformer model to receive input with a fixed seq_len=aggr. Otherwise, any of the batch sizes listed above 
    should be fine. The model works well with input X being (batch_size, smaller_seq_len, num_feats).

    :param datasets: tuple of datasets
    :param batch_size: batch size for the dataloaders
    :param num_workers: number of subprocesses to use for data loading
    :param shuffle: whether to shuffle the data
    :return: tuple of dataloaders
    """
    dataloaders = []
    cpu_cores = multiprocessing.cpu_count()

    if num_workers is None:
        num_workers = cpu_cores

    logger.info(f"System has {cpu_cores} CPU cores. Using {num_workers}/{cpu_cores} workers for data loading.")
    
    for dataset, shuffle in zip(datasets, shuffle):
        full_batches = dataset.num_seqs // batch_size

        dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            drop_last=drop_last
        )
        dataloaders.append(dataloader)

        logger.info(f"Total batches={len(dataloader)} & full batches={full_batches}, with each full batch containing {batch_size} sequences.")
    
    logger.info("DataLoaders created successfully.")

    return tuple(dataloaders)

def separate(src, c, t):
    """
    Separates channels and time features from the source tensor.

    :param src: tensor (batch_size, seq_len, num_feats)
    :param c: range of channels features
    :param t: range of time features
    :return: tuple of (batch_size, seq_len, num_channels_feats) and (batch_size, seq_len, num_time_feats)
    """
    channels = src[:, :, c]
    time = src[:, :, t]

    return channels, time

def aggregate_seqs(data):
    """
    Aggregates the tensor by reducing the sequence length to a single time step.

    :param data: tensor (batch_size, seq_len, num_feats)
    :return: tensor (batch_size, 1, num_feats)
    """
    return data[:, 0:1, :]

def merge(channels, time):
    """
    Concatenates channels and time feature tensors along the feature dimension.

    :param channels: tensor (batch_size, seq_len, num_channels_feats)
    :param time: tensor (batch_size, seq_len, 1)
    :return: tensor (batch_size, seq_len, num_channels_feats + 1)
    """
    return torch.cat((channels, time), dim=2)