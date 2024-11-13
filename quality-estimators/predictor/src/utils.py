import os
import json
import numpy as np
from collections import namedtuple
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as sched
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import seaborn as sns

def get_logger(level='DEBUG'):
    """
    Create and configure a logger object with the specified logging level.

    :param level: Logging level to set for the logger. Default is 'DEBUG'.
    :return: Logger object configured with the specified logging level.
    """
    logger = logging.getLogger(__name__)

    level_name = logging.getLevelName(level)
    logger.setLevel(level_name)
    
    formatter = logging.Formatter('%(asctime)s:%(lineno)d:%(levelname)s:%(name)s:%(message)s')
    
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    
    return logger

def get_dir(*sub_dirs):
    """
    Retrieve or create a directory path based on the script's location and the specified subdirectories.

    :param sub_dirs: List of subdirectories to append to the script's directory.
    :return: Full path to the directory.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dir = os.path.join(script_dir, *sub_dirs)

    if not os.path.exists(dir):
        os.makedirs(dir)

    return dir

def get_path(*dirs, filename):
    """
    Construct a full file path by combining directory paths and a filename.

    :param dirs: List of directory paths.
    :param filename: Name of the file.
    :return: Full path to the file.
    """
    dir_path = get_dir(*dirs)
    path = os.path.join(dir_path, filename)

    return path

def get_prfs(true, pred, avg=['micro', 'macro', 'weighted'], include_support=False, zero_division=0):
    """
    Calculate Precision, Recall, F-score, and Support using specified averaging methods.

    :param true: List of true labels.
    :param pred: List of predicted labels.
    :param avg: Averaging methods to use for calculating metrics.
    :param include_support: Whether to include Support in the output.
    :param zero_division: Value to return when there is a zero division.
    :return: Dictionary containing Precision, Recall, F-score, and Support for each averaging method.
    """
    prfs = {}

    for method in avg:
        precision, recall, fscore, support = precision_recall_fscore_support(true, pred, average=method, zero_division=zero_division)

        prfs[f'precision_{method}'] = precision
        prfs[f'recall_{method}'] = recall
        prfs[f'fscore_{method}'] = fscore

        if include_support:
            prfs[f'support_{method}'] = support

    return prfs

def get_optim(name, model, lr):
    """
    Get optimizer object based on name, model, and learning rate.

    :param name: Name of the optimizer class.
    :param model: Model to optimize.
    :param lr: Learning rate for the optimizer.
    :return: Optimizer object.
    """
    optim_class = getattr(optim, name)
    optimizer = optim_class(model.parameters(), lr=lr)

    return optimizer

def get_sched(optimizer, name, **params):
    """
    Get scheduler object based on optimizer and additional parameters.

    :param optimizer: Optimizer object to schedule.
    :param name: Name of the scheduler.
    :param params: Additional parameters for the scheduler.
    :return: Scheduler object.
    """
    sched_class = getattr(sched, name)
    scheduler = sched_class(optimizer, **params)

    return scheduler

def visualize(type, values, labels, title, plot_func=None, coloring=None, names=None, classes=None, tick=False, path='static'):
    """
    Visualize (x,y) data points.

    :param type: Type of visualization ('single-plot', 'multi-plot', or 'heatmap').
    :param values: List of tuples or tuple containing the data points to visualize.
    :param labels: Tuple containing labels for the x and y axes.
    :param title: Title of the visualization.
    :param plot_func: Plotting function (optional).
    :param coloring: List or str containing colors for the plots (optional).
    :param names: List of names for the plots (optional).
    :param tick: Whether to display ticks on axes (optional).
    :param classes: List of class names for labeling (optional).
    :param path: Directory path to save the visualization.
    """
    x_label, y_label = labels
    plt.figure(figsize=(10, 6))

    if type == 'single-plot':
        x_values, y_values = values
        plot_func(x_values, y_values, color=coloring)

        if tick:
            plt.xticks(range(len(classes)), classes)
            plt.yticks(range(len(classes)), classes)

    elif type == 'multi-plot':
        x_values, y_values = values

        for i, (x_values, y_values) in enumerate(values):
            plot_func(x_values, y_values, color=coloring[i], label=names[i])
            plt.legend()

        if tick:
            plt.xticks(range(len(classes)), classes)
            plt.yticks(range(len(classes)), classes)

    elif type == 'heatmap':
        x_values, y_values = values

        cm = confusion_matrix(x_values, y_values)
        cmap = sns.blend_palette(coloring, as_cmap=True)

        sns.heatmap(cm, annot=True, fmt="d", cmap=cmap, xticklabels=classes, yticklabels=classes)

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.tight_layout()

    filename = f"{title.lower().replace(' ', '_')}.png"
    plt.savefig(os.path.join(path, filename), dpi=300)
    plt.close()

def save_json(data, filename):
    """
    Save data to a JSON file.

    :param data: Dictionary containing the data to save.
    :param filename: Name of the file to save the data into.
    """
    with open(filename, 'w') as f:
        json.dump(data, f)

class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self, weights):
        """
        Initialize the WeightedCrossEntropyLoss module.

        :param weights: dictionary
        """
        super(WeightedCrossEntropyLoss, self).__init__()
        self.weights = self.get_weights(weights)

    def get_weights(self, weights):
        """
        Extract weights from the given dictionary and convert them to a tensor.

        :param weights: dictionary
        :return: tensor
        """
        weights = [weights[i] for i in range(len(weights))]

        return torch.tensor(weights, dtype=torch.float)

    def forward(self, pred, true):
        """
        Compute the weighted cross-entropy loss.

        :param pred: tensor (batch_size * seq_len, num_classes)
        :param true: tensor (batch_size * seq_len)
        :return: tensor
        """
        if true.size(0) == 0 or pred.size(0) == 0:
            return torch.tensor(0.0, requires_grad=True, device=pred.device)

        loss = F.cross_entropy(pred, true, weight=self.weights.to(pred.device))

        return loss

def robust_normalize(df, exclude, path):
    """
    Normalize data using robust scaling (median and IQR) from precomputed stats.

    :param df: DataFrame containing the data to normalize.
    :param exclude: List of columns to exclude from normalization.
    :param path: File path to save the computed statistics.
    :return: Processed DataFrame with normalized data.
    """
    newdf = df.copy()
    
    stats = get_stats(df)
    save_json(data=stats, filename=path)
    
    for col in df.columns:
        if col not in exclude:
            median = stats[col]['median']
            iqr = stats[col]['iqr']
            
            newdf[col] = (df[col] - median) / (iqr if iqr > 0 else 1)

    return newdf

def get_stats(df):
    """
    Compute mean, standard deviation, median, and IQR for each column in the DataFrame.

    :param df: DataFrame containing the data to compute statistics for.
    :return: Dictionary containing statistics for each column.
    """
    stats = {}

    for col in df.columns:
        series = df[col]

        mean = series.mean()
        std = series.std()
        median = series.median()
        iqr = series.quantile(0.75) - series.quantile(0.25)

        stats[col] = {
            'mean': mean,
            'std': std,
            'median': median,
            'iqr': iqr
        }

    return stats
    
class BlendedLoss(nn.Module):
    def __init__(self, p=1.0, epsilon=1e-6, blend=0.01):
        """
        Initialize the BlendedLoss module.

        :param p: Power to which the differences are raised.
        :param epsilon: Small value added for numerical stability.
        :param blend: Blend factor between median and mean.
        """
        super(BlendedLoss, self).__init__()
        self.p = p
        self.epsilon = epsilon
        self.blend = blend

    def forward(self, input, target):
        """
        Compute the blended loss between the input and target.

        :param input: Tensor containing the predicted values.
        :param target: Tensor containing the target values.
        :return: Computed blended loss.
        """
        diff = torch.abs(input - target) + self.epsilon

        powered_diff = diff ** self.p
        median_diff = (1 - self.blend) * torch.median(powered_diff)
        mean_diff = self.blend * torch.mean(powered_diff)
        
        loss = median_diff + mean_diff
        
        return loss