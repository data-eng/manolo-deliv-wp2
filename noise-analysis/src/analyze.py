import yaml
import warnings
import numpy as np
import seaborn as sns
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from . import utils

logger = utils.get_logger(level='INFO')
warnings.filterwarnings("ignore", category=FutureWarning)

with open('config.yaml', 'r') as config_file:
    config = yaml.safe_load(config_file)

passage = config['passage']
estimators = config['estimators']
thresholds = config['thresholds']
window = config['window']
perc = config['perc']

def load_noise_data():
    """
    Load noise data from CSV files for each estimator.

    :return: Dictionary containing noise values for each feature in each estimator.
    """
    noise_dict = {}

    for id in estimators:
        csv_path = utils.get_path('..', '..', 'quality-estimators', 'data', 'proc', filename=f'estim_{id}.csv')
        
        df = pd.read_csv(csv_path)
        
        logger.info(f"Loaded CSV for {id} from {csv_path}")
        logger.info(f"Unique {passage['name']}s in {id}.csv: {df[passage['name']].unique()}")

        df = df[df[passage['name']].isin(passage['values'])].sort_values(by='time')
        
        noise_cols = [col for col in df.columns if col.startswith('noise_')]
        noise_dict[id] = {col.replace('noise_', ''): None for col in noise_cols}

    return noise_dict

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

def suggest_thresholds(values):
    """
    Calculate three meaningful thresholds based on noise values statistics.

    :param values: List of noise values for a specific channel.
    :return: Tuple of three thresholds based on percentiles.
    """
    low = np.percentile(values, 95)
    mid = np.percentile(values, 97)
    high = np.percentile(values, 99)
    
    return low, mid, high

def estimate_binary_noise():
    """
    Estimate binary noise values for each estimator and save the results to CSV files. 
    Loads data, applies segment averaging, computes thresholds, and outputs binary noise estimates.
    """
    noise_cols = ['noise_HB_1', 'noise_HB_2']
    df8 = None
    
    csv8_path = utils.get_path('..', '..', 'quality-estimators', 'data', 'proc', filename=f'test8.csv')

    if os.path.exists(csv8_path):
        df8 = pd.read_csv(csv8_path)

        logger.info(f"Loaded CSV from {csv8_path} that contains data from label-8 outliers.")

        for noise_col, noise_value in zip(noise_cols, [96839061427, 72416093869]):
            df8[noise_col] = noise_value
    else:
        logger.info(f"{csv8_path} does not exist. Skipping df8 concatenation.")

    pname = passage['name']

    for id in estimators:
        df_avg = pd.DataFrame()

        threshold = thresholds[id]

        lows, mids, highs = {col: [] for col in noise_cols}, {col: [] for col in noise_cols}, {col: [] for col in noise_cols}

        csv_path = utils.get_path('..', '..', 'quality-estimators', 'data', 'proc', filename=f'estim_{id}.csv')
        df = pd.read_csv(csv_path)

        logger.info(f"Loaded CSV for {id} from {csv_path}.")
        logger.info(f"Unique {passage['name']}s in {id}.csv: {df[passage['name']].unique()}")

        if df8 is None:
            df8 = pd.DataFrame(columns=df.columns)

        if id == 'lstm_ae' or id == 'conv_lstm_ae' or id == 'attn_ae' or id == 'transformer':
            df = pd.concat([df, df8], ignore_index=True).sort_values(by='time')

        for p in passage['values']:
            df_p_avg = pd.DataFrame()

            df_p = df[df[pname] == p]

            for idx, col in enumerate(noise_cols):
                df_p_avg[col] = average_over_segments(df_p[col].tolist(), segment_size=window)

                low_threshold, mid_threshold, high_threshold = suggest_thresholds(df_p_avg[col].tolist())

                lows[col].append(low_threshold)
                mids[col].append(mid_threshold)
                highs[col].append(high_threshold)

                df_p_avg[f"binary_{col}"] = [utils.binary(val, threshold[idx]) for val in df_p_avg[col].tolist()]

            df_p_avg['id'] = np.arange(1, len(df_p_avg) + 1)
            df_p_avg[pname] = df_p[pname].iloc[0]

            df_avg = pd.concat([df_avg, df_p_avg], ignore_index=True)

        logger.info(f"Aggregated DataFrame for {id} using window size {window}, with {len(df_avg)} rows.")

        mean_lows = {col: sum(lows[col]) / len(lows[col]) if lows[col] else 'N/A' for col in lows}
        mean_mids = {col: sum(mids[col]) / len(mids[col]) if mids[col] else 'N/A' for col in mids}
        mean_highs = {col: sum(highs[col]) / len(highs[col]) if highs[col] else 'N/A' for col in highs}

        logger.info(f"Suggested thresholds for {id}:")
        logger.info(f"Low: [{mean_lows['noise_HB_1']:.2f}, {mean_lows['noise_HB_2']:.2f}]")
        logger.info(f"Mid: [{mean_mids['noise_HB_1']:.2f}, {mean_mids['noise_HB_2']:.2f}]")
        logger.info(f"High: [{mean_highs['noise_HB_1']:.2f}, {mean_highs['noise_HB_2']:.2f}]")

        agg_csv_path = utils.get_path('..', '..', 'quality-estimators', 'data', 'proc', filename=f'agg_{id}.csv')
        #df_avg.to_csv(agg_csv_path, index=False)

        logger.info(f"Saved aggregated data to {agg_csv_path} with {len(df_avg)} rows.")

        for col in noise_cols:
            df_avg.drop(columns=[col], inplace=True)
            df_avg.rename(columns={f'binary_{col}': col}, inplace=True)
        
        threshold_str = f'_{ "_".join(map(str, threshold))}' if threshold is not None else ''
        binary_csv_path = utils.get_path('..', '..', 'quality-estimators', 'data', 'proc', 
                                 filename=f'bin_{id}{threshold_str}.csv')

        #df_avg.to_csv(binary_csv_path, index=False)

        logger.info(f"Saved binary aggregated data to {binary_csv_path} with {len(df_avg)} rows.")

def get_top_noisy_timesteps(type='agg'):
    """
    Extract the top 100*k% of noisy timesteps for each feature and estimator, based on noise values 
    from either the aggregated or binary data.

    :param type: Specifies whether to use 'agg' (aggregated) or 'bin' (binary) data.
    :return: Dictionary containing top noisy timesteps for each feature in each estimator.
    """
    topKs = {}
    noise_cols = ['noise_HB_1', 'noise_HB_2']

    path = utils.get_path('..', '..', 'quality-estimators', 'data', 'proc', filename='topKs.npy')

    for id in estimators:
        csv_path = utils.get_path('..', '..', 'quality-estimators', 'data', 'proc', filename=f'agg_{id}.csv')
        
        df = pd.read_csv(csv_path)
        topK = {}

        for noise_col in noise_cols:
            binary_noise_col = 'binary_' + noise_col

            if type == "bin":
                df_filtered = df[df[binary_noise_col] == 1]

            df_sorted = df_filtered.sort_values(by=noise_col, ascending=False)
 
            total_rows = len(df_sorted)
            k_perc = int(total_rows * perc)
            logger.info(f"Estimator {id} - {noise_col}: Total rows = {total_rows}, Top {perc*100}% rows = {k_perc}.")

            df_topK = df_sorted.iloc[:k_perc]

            topK_times = df_topK['id'].to_numpy()
            feature_name = noise_col.replace('noise_', '')

            topK[feature_name] = topK_times

        topKs[id] = topK
    
    np.save(path, topKs)
    logger.info(f"Saved top {perc*100}% timesteps to {path}.")

    return topKs

def average_over_segments(values, segment_size):
    """
    Calculate the average of values over specified segments.
    
    The last segment may have fewer elements than segment_size.
    """
    values_array = np.array(values)
    num_segments = len(values_array) // segment_size

    if num_segments > 0:
        averages = values_array[:num_segments * segment_size].reshape(-1, segment_size).mean(axis=1)
    else:
        averages = values_array

    return averages.tolist()

def get_fscore(times_1, times_2):
    """
    Compute the F-score between two sets of timestamps.

    :param times_1: Set of timestamps for the first estimator.
    :param times_2: Set of timestamps for the second estimator.
    :return: F-score between the two sets.
    """
    true_positives = len(times_1.intersection(times_2))
    predicted_positives = len(times_2)
    actual_positives = len(times_1)

    precision = true_positives / predicted_positives if predicted_positives > 0 else 0
    recall = true_positives / actual_positives if actual_positives > 0 else 0

    fscore = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return fscore

def compute_agreement():
    """
    Compute the percentage agreement between pairs of estimators for each feature based on 
    the provided dict (timesteps or other metric).

    :return: Dictionary showing agreement percentages between estimator pairs for each feature.
    """
    path = utils.get_path('..', '..', 'quality-estimators', 'data', 'proc', filename='topKs.npy')
    
    if os.path.exists(path):
        dict = np.load(path, allow_pickle=True).item()
        logger.info(f"Loaded topK values from {path}")
    else:
        logger.warning(f"{path} does not exist. Ensure the topK values are generated.")
        return {}
    
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

                logger.debug(f"Feature {feature} - Estimator {est_2}: Length of times_2 = {len(times_2)}")
                
                fscore_12 = get_fscore(times_1, times_2)
                fscore_21 = get_fscore(times_2, times_1)

                agreement_dict[feature][(est_1, est_2)] = fscore_12
                agreement_dict[feature][(est_2, est_1)] = fscore_21

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

        for (est_1, est_2), score in agreements.items():
            idx_1 = estimator_ids.index(est_1)
            idx_2 = estimator_ids.index(est_2)

            agreement_matrix[idx_1, idx_2] = score

        sns.heatmap(agreement_matrix, annot=True, fmt=".4f", cmap='RdBu',
                    xticklabels=estimator_ids, yticklabels=estimator_ids,
                    cbar_kws={'label': 'Agreement Score'}, ax=ax)
        
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
    extracts the top 100*k% noisy timesteps, computes the agreement percentages
    between estimator pairs for each feature, and visualizes these agreements as heatmaps.
    """
    #noise_dict = load_noise_data()
    #visualize_noise(dict=noise_dict)

    estimate_binary_noise()
    topKs = get_top_noisy_timesteps(type='bin')

    agreement_dict = compute_agreement()
    visualize_agreement(dict=agreement_dict)

if __name__ == '__main__':
    main()