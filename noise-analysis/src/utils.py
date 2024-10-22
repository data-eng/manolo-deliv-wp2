import os
import logging
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
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

def retain_if_greater_than_one(value):
    """
    Transformation function that retains the value if greater than 1, otherwise returns 0.
    :param value: The input value to be evaluated.
    :return: The original value if greater than 1; otherwise, returns 0.
    """
    return value if value > 1 else 0