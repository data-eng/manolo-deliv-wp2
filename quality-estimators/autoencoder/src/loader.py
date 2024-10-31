import torch
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
    Retrieve and combine EEG and event data for subjects from the Bitbrain dataset. For each subject, 
    this function reads EEG signals from an EDF file and event data from a TSV file, combines them, 
    and saves the result as a CSV file in the specified output directory.

    :param base_path: Path to the base directory containing subject folders.
    :param output_path: Path to the directory where combined data will be saved.
    """
    for subject_folder in glob.glob(os.path.join(base_path, 'sub-*')):
        subject_id = os.path.basename(subject_folder)
        eeg_folder = os.path.join(subject_folder, 'eeg')

        output_file = os.path.join(output_path, f'{subject_id}.csv')

        if os.path.exists(output_file):
            continue

        if not os.path.exists(eeg_folder):
            print(f'No EEG folder found for {subject_id}. Skipping.')
            continue

        eeg_file_pattern = os.path.join(eeg_folder, f'{subject_id}_task-Sleep_acq-headband_eeg.edf')
        events_file_pattern = os.path.join(eeg_folder, f'{subject_id}_task-Sleep_acq-psg_events.tsv')

        try:
            raw = mne.io.read_raw_edf(eeg_file_pattern, preload=True)
            x_data = raw.to_data_frame()
        except Exception as e:
            print(f'Error loading EEG data for {subject_id}: {e}')
            continue

        try:
            y_data = pd.read_csv(events_file_pattern, delimiter='\t')
        except Exception as e:
            print(f'Error loading events data for {subject_id}: {e}')
            continue

        combined_data = pd.concat([x_data, y_data], axis=1)

        combined_data.to_csv(output_file, index=False)
        print(f'Saved combined data for {subject_id} to {output_file}')

class TSDataset(Dataset):
    def __init__(self, df, seq_len, X, t, y, per_epoch=True):
        """
        Initializes a time series dataset. It creates sequences from the input data by 
        concatenating features and time columns. The target variable is stored separately.

        :param df: Pandas dataframe containing the data.
        :param seq_len: Length of the input sequence (number of time steps).
        :param X: List of feature columns.
        :param t: List of time-related columns.
        :param y: List of target columns.
        :param per_epoch: Whether to create sequences in non-overlapping (True) or overlapping (False) epochs.
        """
        self.seq_len = seq_len
        self.X = pd.concat([df[X], df[t]], axis=1)
        self.y = df[y]
        self.per_epoch = per_epoch

        logger.info(f'Initializing dataset with: samples={self.num_samples}, samples/seq={seq_len}, seqs={self.num_seqs}, epochs={self.num_epochs} ')

    def __len__(self):
        """
        Returns the number of sequences in the dataset.

        :return: Length of the dataset.
        """
        return self.num_seqs

    def __getitem__(self, idx):
        """
        Retrieves a sample from the dataset at the specified index.

        :param idx: Index of the sample.
        :return: Tuple of features and target tensors.
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
        """
        Returns the total number of samples in the dataset.
        
        :return: Total number of samples.
        """
        return self.X.shape[0]
    
    @property
    def num_epochs(self):
        """
        Returns the number of full epochs available based on the dataset size.

        :return: Number of epochs.
        """
        return self.num_samples // 7680

    @property
    def max_seq_id(self):
        """
        Returns the maximum index for a sequence.

        :return: Maximum index for a sequence.
        """
        return self.num_samples - self.seq_len
    
    @property
    def num_seqs(self):
        """
        Returns the number of sequences that can be created from the dataset.

        :return: Number of sequences.
        """
        if self.per_epoch:
            return self.num_samples // self.seq_len
        else:
            return self.max_seq_id + 1

def split_data(dir, train_size=57, val_size=1, test_size=1):
    """
    Split the CSV files into training, validation, and test sets.

    :param dir: Directory containing the CSV files.
    :param train_size: Number of files for training.
    :param val_size: Number of files for validation.
    :param test_size: Number of files for testing.
    :return: Tuple of lists containing CSV file paths for train, val, and test sets.
    """
    paths = [utils.get_path(dir, filename=file) for file in os.listdir(dir)]
    logger.info(f'Found {len(paths)} files in directory: {dir} ready for splitting.')

    train_paths = paths[:train_size]
    val_paths = paths[train_size:train_size + val_size]
    test_paths = paths[train_size + val_size:train_size + val_size + test_size]

    logger.info(f'Splitting complete!')

    return (train_paths, val_paths, test_paths)

def load_file(path):
    """
    Load data from a CSV file.

    :param path: Path to the CSV file.
    :return: Tuple (X, t, y) where X contains EEG features, t contains time, and y contains labels.
    """
    df = pd.read_csv(path)

    X = df[['HB_1', 'HB_2']].values
    t = df['time'].values
    y = df['majority'].values

    return X, t, y

def combine_data(paths, seq_len=240):
    """
    Combine data from multiple CSV files into a dataframe, processing sequences and removing invalid rows.

    :param paths: List of file paths to CSV files.
    :param seq_len: Sequence length for grouping data.
    :return: Combined dataframe after processing.
    """
    dataframes = []
    total_removed_majority = 0

    logger.info(f'Combining data from {len(paths)} files.')

    for path in paths:
        X, t, y = load_file(path)

        df = pd.DataFrame(X, columns=['HB_1', 'HB_2'])
        df['majority'] = y
        df['time'] = t

        df['seq_id'] = (np.arange(len(df)) // seq_len) + 1
        df['night'] = int(os.path.basename(path).split('-')[1].split('.')[0])

        rows_before_majority_drop = df.shape[0]
        df.drop(df[df['majority'] == 8].index, inplace=True)
        total_removed_majority += (rows_before_majority_drop - df.shape[0])

        dataframes.append(df)

    df = pd.concat(dataframes, ignore_index=True)
    logger.info(f'Combined dataframe shape: {df.shape}')

    logger.info(f'Removed {total_removed_majority} rows with majority value -1.')
    
    rows_before_nan_drop = df.shape[0]
    df.dropna(inplace=True)
    logger.info(f'Removed {rows_before_nan_drop - df.shape[0]} rows with NaN values.')

    assert not df.isna().any().any(), 'NaN values found in the dataframe!'

    stats_path = utils.get_path('..', '..', 'data', filename='stats.json')
    df = utils.robust_normalize(df, exclude=['night', 'seq_id', 'time', 'majority'], path=stats_path)

    return df

def get_dataframes(paths, seq_len=240, exist=False):
    """
    Create or load dataframes for training, validation, and testing.

    :param paths: List of file paths for training, validation, and testing.
    :param exist: Boolean flag indicating if the dataframes already exist.
    :return: Tuple of dataframes for train, validation, and test sets.
    """
    dataframes = []
    names = ['train', 'val', 'test']
    weights = None

    logger.info('Creating dataframes for training, validation, and testing.')

    for paths, name in zip(paths, names):
        proc_path = utils.get_path('..', '..', 'data', 'proc', filename=f'{name}.csv')

        if exist:
            df = pd.read_csv(proc_path)
            logger.info(f'Loaded existing dataframe from {proc_path}.')
        else:
            df = combine_data(paths, seq_len)

            if name == 'train':
                logger.info('Calculating class weights from the training dataframe.')

                weights, _ = extract_weights(df, label_col='majority')

            label_mapping = get_label_mapping(weights=weights)
            df['majority'] = df['majority'].map(label_mapping)

            df.to_csv(proc_path, index=False)
            logger.info(f'Saved {name} dataframe to {proc_path}.')

        dataframes.append(df)

    logger.info('Dataframes for training, validation, and testing are ready!')

    return tuple(dataframes)

def extract_weights(df, label_col):
    """
    Calculate class weights from the training dataframe to handle class imbalance, and save them to a file.

    :param df: Dataframe containing the training data.
    :param label_col: The name of the column containing class labels.
    :return: A tuple containing a dictionary of class weights and a list of class labels if mapping is enabled.
    """
    occs = df[label_col].value_counts().to_dict()
    inverse_occs = {key: 1e-10 for key in occs.keys()}

    for key, value in occs.items():
        inverse_occs[int(key)] = 1 / (value + 1e-10)

    weights = {key: value / sum(inverse_occs.values()) for key, value in inverse_occs.items()}
    weights = dict(sorted(weights.items()))

    new_weights = {i: weights[key] for i, key in enumerate(weights.keys())}

    path = utils.get_path('..', '..', 'data', filename='weights.json')
    utils.save_json(data=new_weights, filename=path)

    return weights, new_weights

def get_label_mapping(weights):
    label_mapping = {original_label: new_index for new_index, original_label in enumerate(weights.keys())}

    return label_mapping

def create_datasets(dataframes, seq_len=7680):
    """
    Create datasets for the specified dataframes (e.g. training, validation, and testing).

    :param dataframes: Tuple of dataframes.
    :param seq_len: Sequence length for each dataset sample.
    :return: Tuple of datasets.
    """
    datasets = []

    X = ['HB_1', 'HB_2']
    t = ['time', 'seq_id', 'night']
    y = ['majority']

    logger.info('Creating datasets from dataframes.') 

    for df in dataframes:
        dataset = TSDataset(df, seq_len, X, t, y)
        datasets.append(dataset)

    logger.info(f'Datasets created successfully!')

    return tuple(datasets)

def create_dataloaders(datasets, batch_size=1, shuffle=[True, False, False], num_workers=None, drop_last=False):
    """
    Create DataLoader objects for the specified datasets, providing data in batches for training, validation, and testing.

    :param datasets: Tuple of datasets.
    :param batch_size: Batch size for the DataLoader.
    :param shuffle: List indicating whether to shuffle data for each dataset.
    :param num_workers: Number of subprocesses to use for data loading (default is all available CPU cores).
    :param drop_last: Whether to drop the last incomplete batch.
    :return: Tuple of DataLoader objects.
    """
    dataloaders = []
    cpu_cores = multiprocessing.cpu_count()

    if num_workers is None:
        num_workers = cpu_cores

    logger.info(f'System has {cpu_cores} CPU cores. Using {num_workers}/{cpu_cores} workers for data loading.')
    
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

        logger.info(f'Total batches={len(dataloader)} & full batches={full_batches}, with each full batch containing {batch_size} sequences.')
    
    logger.info('DataLoaders created successfully.')

    return tuple(dataloaders)

def separate(src, c, t):
    """
    Separates channels and time features from the source tensor.

    :param src: Tensor of shape (batch_size, seq_len, num_feats).
    :param c: Range of channel features.
    :param t: Range of time features.
    :return: Tuple of (channels, time) tensors.
    """
    channels = src[:, :, c]
    time = src[:, :, t]

    return channels, time

def aggregate_seqs(data):
    """
    Aggregates the tensor by reducing the sequence length to a single time step.

    :param data: Tensor of shape (batch_size, seq_len, num_feats).
    :return: Tensor of shape (batch_size, 1, num_feats).
    """
    return data[:, 0:1, :]

def merge(c, t):
    """
    Concatenates channel and time feature tensors along the feature dimension.

    :param c: Tensor of shape (batch_size, seq_len, num_channels_feats).
    :param t: Tensor of shape (batch_size, seq_len, num_time_feats).
    :return: Tensor of shape (batch_size, seq_len, num_channels_feats + num_time_feats).
    """
    return torch.cat((c, t), dim=2)