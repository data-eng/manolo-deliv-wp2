import yaml
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from . import utils

logger = utils.get_logger(level='DEBUG')
warnings.filterwarnings("ignore", category=FutureWarning)

with open('config.yaml', 'r') as config_file:
    config = yaml.safe_load(config_file)

def average_over_segments(values, segment_size):
    """
    Calculate the average of values over specified segments.
    :param values: List of values to be averaged.
    :param segment_size: Size of each segment for averaging.
    :return: List of averaged values for each segment.
    """
    return [np.mean(values[i:i + segment_size]) for i in range(0, len(values), segment_size)]

def visualize_estimators(noise_dict, passage, shift=2, window=10000, dpi=600):
    """
    Visualize the noise values for each estimator with a shift.
    :param noise_dict: Dictionary of noise values for each estimator.
    :param passage: Information about the specific passage to visualize.
    :param window_size: Size of the window for averaging.
    :param dpi: Dots per inch for the saved figure.
    """
    features = list(noise_dict[next(iter(noise_dict))].keys())
    num_features = len(features)
    pn, pv = passage['name'], passage['value']
    
    fig, axes = plt.subplots(nrows=1, ncols=num_features, figsize=(10 * num_features, 5))
    colors = cm.viridis(np.linspace(0, 1, len(noise_dict)))
    
    for idx, feature in enumerate(features):
        ax = axes[idx] if num_features > 1 else axes
        
        for estimator_id, (noise_values, color) in zip(noise_dict.keys(), zip(noise_dict.values(), colors)):
            s = list(noise_dict.keys()).index(estimator_id) * shift

            averaged_values = average_over_segments(noise_values[feature], window)

            x_values = range(len(averaged_values))
            y_values = [abs(val) + s for val in averaged_values]

            ax.plot(x_values, y_values, label=estimator_id, color=color, linewidth=0.5)
        
        ax.set_title(f'Noise values for {feature} - {pn.capitalize()} {pv}')
        ax.set_xlabel('Time')
        ax.set_ylabel('Noise')
        ax.legend()
        ax.grid(True)

    path = utils.get_path('..', 'static', filename=f'noise_analysis_{pn}_{pv}.png')
    plt.savefig(path, dpi=dpi)
    plt.close(fig)

def apply(values, func=None):
    """
    Apply the specified transformation function to the list of values.
    :param values: List of values to be transformed.
    :param func: Name of the transformation function to apply.
    :return: Transformed list of values or the original values if function is None.
    """
    if func == "retain_if_greater_than_one":
        return [utils.retain_if_greater_than_one(val) for val in values]
    return values

def main():
    """
    Main function to ...
    """
    passage = config['passage']
    estimators = config['estimators']
    transformation = config['transformation']

    noise_dict = {}

    for id in estimators:
        csv_path = utils.get_path('..', '..', 'quality-estimators', 'data', 'proc', filename=f'estim_{id}.csv')
        
        try:
            df = pd.read_csv(csv_path)
            
            logger.debug(f"Loaded CSV for {id} from {csv_path}")
            logger.info(f"Unique {passage['name']}s in {id}.csv: {df[passage['name']].unique()}")

            df = df[df[passage['name']] == passage['value']].sort_values(by='time')
            
            noise_cols = [col for col in df.columns if col.startswith('noise_')]
            noise_dict[id] = {col.replace('noise_', ''): None for col in noise_cols}
            
            for col in noise_dict[id]:
                noise_dict[id][col] = apply(values=df[f'noise_{col}'].tolist(), func=transformation)

        except FileNotFoundError:
            logger.error(f"CSV file for {id} not found at {csv_path}")

    visualize_estimators(noise_dict, passage, window=config['window'])

if __name__ == '__main__':
    main()