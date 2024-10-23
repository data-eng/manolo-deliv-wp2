import yaml
import warnings
import numpy as np
import seaborn as sns
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from . import utils

logger = utils.get_logger(level='DEBUG')
warnings.filterwarnings("ignore", category=FutureWarning)

with open('config.yaml', 'r') as config_file:
    config = yaml.safe_load(config_file)

passage = config['passage']
estimators = config['estimators']
transformation = config['transformation']
window = config['window']

def load_noise_data():
    """
    Load noise data from CSV files for each estimator and apply transformations.

    :return: Dictionary containing noise values for each feature in each estimator.
    """
    noise_dict = {}

    for id in estimators:
        csv_path = utils.get_path('..', '..', 'quality-estimators', 'data', 'proc', filename=f'estim_{id}.csv')
        
        df = pd.read_csv(csv_path)
        
        logger.debug(f"Loaded CSV for {id} from {csv_path}")
        logger.info(f"Unique {passage['name']}s in {id}.csv: {df[passage['name']].unique()}")

        df = df[df[passage['name']] == passage['value']].sort_values(by='time')
        
        noise_cols = [col for col in df.columns if col.startswith('noise_')]
        noise_dict[id] = {col.replace('noise_', ''): None for col in noise_cols}
        
        for col in noise_dict[id]:
            noise_dict[id][col] = apply(values=df[f'noise_{col}'].tolist(), func=transformation)

    return noise_dict

def get_top_noisy_timesteps(k=0.1, exists=False):
    """
    Extract top 100*k% noisy timesteps for each feature and estimator.

    :return: Dictionary containing top 10% noisy timesteps for each feature in each estimator.
    """
    topKs = {}
    path = utils.get_path('..', '..', 'quality-estimators', 'data', 'proc', filename='topKs.npy')

    if exists and os.path.exists(path):
        topKs = np.load(path, allow_pickle=True).item()
        logger.info(f"Loaded existing top {k*100}% timesteps from {path}.")

        return topKs

    for id in estimators:
        csv_path = utils.get_path('..', '..', 'quality-estimators', 'data', 'proc', filename=f'estim_{id}.csv')
        df = pd.read_csv(csv_path)
        
        logger.debug(f"Loaded CSV for {id} from {csv_path}.")
        
        noise_cols = [col for col in df.columns if col.startswith('noise_')]
        topK = {}

        for noise_col in noise_cols:
            df_sorted = df.sort_values(by=noise_col, ascending=False)
            
            total_rows = len(df_sorted)
            k_perc = int(total_rows * k)
            logger.info(f"For {noise_col}: Total rows = {total_rows}, Top {k*100}% rows = {k_perc}.")

            df_topK = df_sorted.iloc[:k_perc]

            topK_times = df_topK['time'].to_numpy()
            feature_name = noise_col.replace('noise_', '')

            topK[feature_name] = topK_times

        topKs[id] = topK
    
    np.save(path, topKs)
    logger.info(f"Saved top {k*100}% timesteps to {path}.")

    return topKs

def average_over_segments(values, segment_size):
    """
    Calculate the average of values over specified segments.

    :param values: List of values to be averaged.
    :param segment_size: Size of each segment for averaging.
    :return: List of averaged values for each segment.
    """
    return [np.mean(values[i:i + segment_size]) for i in range(0, len(values), segment_size)]

def visualize_noise(dict, shift=2, outlier_threshold=50, dpi=600):
    """
    Visualize the noise values for each estimator with a shift.

    :param dict: Dictionary containing noise values for each feature in each estimator.
    :param shift: Value to shift the noise values for each estimator in the plot.
    :param outlier_threshold: Threshold above which noise values are considered outliers and shown as dots.
    :param dpi: Dots per inch for the saved figure.
    """
    features = list(dict[next(iter(dict))].keys())
    num_features = len(features)
    pn, pv = passage['name'], passage['value']
    
    fig, axes = plt.subplots(nrows=1, ncols=num_features, figsize=(10 * num_features, 5))
    colors = cm.viridis(np.linspace(0, 1, len(dict)))
    
    for idx, feature in enumerate(features):
        ax = axes[idx] if num_features > 1 else axes
        
        for estimator_id, (noise_values, color) in zip(dict.keys(), zip(dict.values(), colors)):
            s = list(dict.keys()).index(estimator_id) * shift

            averaged_values = average_over_segments(noise_values[feature], window)

            x_values = range(len(averaged_values))
            y_values = [abs(val) + s for val in averaged_values]

            y_values_arr = np.array(y_values)
            outliers_idx = np.where(y_values_arr > outlier_threshold)[0]

            y_values_zeroed = y_values_arr.copy()
            y_values_zeroed[outliers_idx] = 0

            ax.plot(x_values, y_values_zeroed, label=estimator_id, color=color, linewidth=0.5)

            ax.scatter(np.array(x_values)[outliers_idx], y_values_arr[outliers_idx], 
                       label=f'{estimator_id} Outliers', color=color, s=10, marker='o')
        
        ax.set_title(f'Noise values for {feature} - {pn.capitalize()} {pv}')
        ax.set_xlabel('Time')
        ax.set_ylabel('Noise')

        ax.set_ylim(0, outlier_threshold)

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

def compute_agreement(dict):
    """
    Compute the percentage agreement between pairs of estimators for each feature based on 
    the provided dict (timesteps or other metric).

    :param dict: Dictionary containing noisy timesteps for each feature in each estimator.
    :return: Dictionary showing agreement percentages between estimator pairs for each feature.
    """
    agreement_dict = {}
    
    features = list(dict[next(iter(dict))].keys())
    estimator_ids = list(dict.keys())

    for feature in features:
        agreement_dict[feature] = {}

        for i in range(len(estimator_ids)):
            est_1 = estimator_ids[i]
            times_1 = set(dict[est_1][feature])

            for j in range(i + 1, len(estimator_ids)):
                
                est_2 = estimator_ids[j]
                times_2 = set(dict[est_2][feature])
                
                common_times = times_1.intersection(times_2)
                total_times = times_1.union(times_2)

                agreement_percentage = (len(common_times) / len(total_times)) * 100 if total_times else 0
                agreement_dict[feature][(est_1, est_2)] = agreement_percentage

    return agreement_dict

def visualize_agreement(dict):
    """
    Visualize the agreement percentages as heatmaps for each feature.

    :param dict: Dictionary containing agreement percentages for each feature.
    """
    estimator_ids = config['estimators']

    num_features = len(dict)
    fig, axes = plt.subplots(1, num_features, figsize=(8 * num_features, 6))

    axes = [axes] if num_features == 1 else axes

    for ax, (feature, agreements) in zip(axes, dict.items()):
        agreement_matrix = np.ones((len(estimator_ids), len(estimator_ids)))

        for (est_1, est_2), percentage in agreements.items():
            idx_1 = estimator_ids.index(est_1)
            idx_2 = estimator_ids.index(est_2)

            agreement_matrix[idx_1, idx_2] = percentage/100
            agreement_matrix[idx_2, idx_1] = percentage/100

        sns.heatmap(agreement_matrix, annot=True, fmt=".2f", cmap='coolwarm',
                    xticklabels=estimator_ids, yticklabels=estimator_ids,
                    cbar_kws={'label': 'Agreement Percentage (%)'}, ax=ax)
        
        ax.set_title(f'{feature}')
        ax.set_xlabel('Estimators')
        ax.set_ylabel('Estimators')
    
    plt.tight_layout()

    path = utils.get_path('..', 'static', filename='agreement_heatmap.png')
    plt.savefig(path)
    plt.close(fig)

    logger.info(f"Saved agreement heatmap to {path}.")

def main():
    """
    Main function to execute the noise analysis workflow.

    It loads noise data from CSV files, visualizes noise values for each estimator,
    extracts the top 10% noisy timesteps, computes the agreement percentages
    between estimator pairs for each feature, and visualizes these agreements as heatmaps.
    """
    noise_dict = load_noise_data()
    visualize_noise(dict=noise_dict)

    tops5s = get_top_noisy_timesteps(k=0.05, exists=True)
    agreement_dict = compute_agreement(dict=tops5s)
    visualize_agreement(dict=agreement_dict)

if __name__ == '__main__':
    main()