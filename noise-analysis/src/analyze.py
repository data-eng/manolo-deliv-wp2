from pyrsistent import l
import yaml
import warnings
import numpy as np
import seaborn as sns
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.metrics import precision_score, recall_score

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

def suggest_thresholds(values, channel):
    """
    Calculate three meaningful thresholds based on noise values statistics.

    :param values: List of noise values for a specific channel.
    :return: Tuple of three thresholds based on percentiles.
    """
    if channel == "noise_HB_1":
        low = np.percentile(values, 98.00)
        mid = np.percentile(values, 98.61)
        high = np.percentile(values, 99.00)

    elif channel == "noise_HB_2":
        low = np.percentile(values, 96.00)
        mid = np.percentile(values, 96.71)
        high = np.percentile(values, 97.00)
    
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

    all_data = {col: [] for col in noise_cols}

    for id in estimators:
        df_avg, df_avg_sug = pd.DataFrame(), pd.DataFrame()

        threshold = thresholds[id]

        lows, mids, highs = {col: [] for col in noise_cols}, {col: [] for col in noise_cols}, {col: [] for col in noise_cols}

        csv_path = utils.get_path('..', '..', 'quality-estimators', 'data', 'proc', filename=f'estim_{id}.csv')
        df = pd.read_csv(csv_path)

        logger.info(f"Loaded CSV for {id} from {csv_path}.")
        logger.info(f"Unique {passage['name']}s in {id}.csv: {df[passage['name']].unique()}")

        for col in noise_cols:
            column_zeros = (df[col] == 0).sum()
            logger.info(f"Total zeros in column {col}: {column_zeros} out of {len(df)} values.")

        if df8 is None:
            df8 = pd.DataFrame(columns=df.columns)

        if id != 'mne' and id != 'cusum' and id != 'page_hinkley' and id != 'kl' and id != 'adwin' and id != 'pca':
            df = pd.concat([df, df8], ignore_index=True).sort_values(by='time')

        for p in passage['values']:
            df_p = df[df[pname] == p].sort_values(by='time')

            for col in noise_cols:
                all_data[col].extend(df_p[col].tolist())
        
        logger.info(f"Aggregated data across passages for {id}.")

        for idx, col in enumerate(noise_cols):
            df_avg_sug[col] = average_over_segments(all_data[col], segment_size=window)

            low_threshold, mid_threshold, high_threshold = suggest_thresholds(df_avg_sug[col].tolist(), channel=col)

            lows[col].append(low_threshold)
            mids[col].append(mid_threshold)
            highs[col].append(high_threshold)

        for p in passage['values']:
            df_p_avg = pd.DataFrame()

            df_p = df[df[pname] == p].sort_values(by='time')

            for idx, col in enumerate(noise_cols):
                df_p_avg[col] = average_over_segments(df_p[col].tolist(), segment_size=window)

                """if id in ['adwin', 'pca', 'kl', 'page_hinkley', 'cusum']:
                    df_p_avg[f"binary_{col}"] = [0 if val == 0 else 1 for val in df_p_avg[col].tolist()]
                else:
                    """
                df_p_avg[f"binary_{col}"] = [utils.binary(val, threshold[idx]) for val in df_p_avg[col].tolist()]

            df_p_avg['id'] = np.arange(1, len(df_p_avg) + 1)
            df_p_avg[pname] = df_p[pname].iloc[0]

            df_avg = pd.concat([df_avg, df_p_avg], ignore_index=True)

        logger.info(f"Aggregated DataFrame for {id} using window size {window}, with {len(df_avg)} rows.")

        mean_lows = {col: sum(lows[col]) / len(lows[col]) if lows[col] else 'N/A' for col in lows}
        mean_mids = {col: sum(mids[col]) / len(mids[col]) if mids[col] else 'N/A' for col in mids}
        mean_highs = {col: sum(highs[col]) / len(highs[col]) if highs[col] else 'N/A' for col in highs}

        logger.info(f"Suggested thresholds for {id}:")
        logger.info(f"Low: [{mean_lows['noise_HB_1']:.10f}, {mean_lows['noise_HB_2']:.10f}]")
        logger.info(f"Mid: [{mean_mids['noise_HB_1']:.10f}, {mean_mids['noise_HB_2']:.10f}]")
        logger.info(f"High: [{mean_highs['noise_HB_1']:.10f}, {mean_highs['noise_HB_2']:.10f}]")

        agg_csv_path = utils.get_path('..', '..', 'quality-estimators', 'data', 'proc', filename=f'agg_{id}.csv')
        df_avg.to_csv(agg_csv_path, index=False)

        logger.info(f"Saved aggregated data to {agg_csv_path} with {len(df_avg)} rows.")

        for col in noise_cols:
            df_avg.drop(columns=[col], inplace=True)
            df_avg.rename(columns={f'binary_{col}': col}, inplace=True)
        
        threshold_str = f'_{ "_".join(map(str, threshold))}' if threshold is not None else ''
        binary_csv_path = utils.get_path('..', '..', 'quality-estimators', 'data', 'proc', 
                                 filename=f'bin_{id}{threshold_str}.csv')

        df_avg.to_csv(binary_csv_path, index=False)

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
    fig, axes = plt.subplots(1, num_features, figsize=(10 * num_features, 8))

    axes = [axes] if num_features == 1 else axes

    for ax, (feature, agreements) in zip(axes, dict.items()):
        agreement_matrix = np.ones((len(estimator_ids), len(estimator_ids)))

        for (est_1, est_2), score in agreements.items():
            idx_1 = estimator_ids.index(est_1)
            idx_2 = estimator_ids.index(est_2)

            agreement_matrix[idx_1, idx_2] = score

        sns.heatmap(agreement_matrix, annot=True, fmt=".2f", cmap='RdBu',
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

def noise_as_lines():
    """
    Plot the noise values for each feature (noise_HB_1 and noise_HB_2) as lines.
    One figure with two subplots for each feature.
    Three lines for each estimator (one for each binary CSV file).
    Adds a slight vertical shift to separate lines for better visibility.
    """
    noise_cols = ['noise_HB_1', 'noise_HB_2']
    
    colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
        '#bcbd22', '#17becf', '#8B0000', '#FF6347', '#FFD700', '#98FB98', '#6495ED', '#DC143C'
    ]
  
    vertical_shift = 0.05
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    for i, col in enumerate(noise_cols):
        ax = axes[i]
        
        for idx, estimator in enumerate(estimators):
            threshold = thresholds[estimator]

            threshold_str = f'_{ "_".join(map(str, threshold))}' if threshold is not None else ''
            bin_csv_path = utils.get_path('..', '..', 'quality-estimators', 'data', 'proc', 
                                 filename=f'bin_{estimator}{threshold_str}.csv')
            
            if os.path.exists(bin_csv_path):
                df = pd.read_csv(bin_csv_path)
                
                df = df.sort_values(by=['night', 'id'])
                
                noise_values = df[col].values

                zero_count = (noise_values == 0).sum()
                total_count = len(noise_values)
                zero_ratio = zero_count / total_count
                logger.info(f"Feature '{col}', Estimator '{estimator}': {zero_count}/{total_count} values are exactly zero ({zero_ratio:.2%}).")
                
                shifted_noise_values = noise_values + (vertical_shift * idx)
                
                ax.plot(shifted_noise_values, label=estimator, color=colors[idx], linewidth=2)
                
            else:
                logger.warning(f"CSV file for {estimator} not found: {bin_csv_path}")
        
        ax.set_title(f'Noise values for {col}')
        ax.set_xlabel('Time (rows)')
        ax.set_ylabel(f'Noise ({col})')
        ax.legend(loc='upper right')
        ax.grid(True)
    
    output_path = utils.get_path('..', 'static', filename='noise_values_line_plot.png')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)
    
    logger.info(f"Saved noise line plot to {output_path}.")

def compare_with_mne():
    """
    Compare the results each estimator with the ground thruth MNE estimations. Calculate precision and 
    recall for noise features (noise_HB_1 and noise_HB_2) and save the results as CSV tables.

    :return: A dictionary containing the results for each estimator, with precision and recall values.
    """
    bins_dir = utils.get_dir('..', '..', 'quality-estimators', 'data', 'proc','bins')
    result_dir = utils.get_dir('..', '..', 'quality-estimators', 'data', 'proc', 'bins' ,'results')
    
    estimators = [
        "adwin", "attn_ae_a", "attn_ae_r", "classifier", "conv_lstm_ae",
        "cusum", "kl", "lstm_ae", "page_hinkley", "pca", "predictor_a", "predictor_r"
    ]
    
    bin_files = [
        f for f in os.listdir(bins_dir)
        if f.endswith('_10.csv') and 'mne' not in f
    ]
    
    results = {}
    
    for bin_file in bin_files:
        estimator_name = next((est for est in estimators if f"_{est}_" in bin_file), None)
        print("The name of the estimator is: ", estimator_name)
        print("Bin file is: ", bin_file)
        
        if estimator_name is None:
            print(f"Warning: Estimator name not found in the file {bin_file}. Skipping.")
            continue

        mne_file = f'bin_mne_0.6744791667_0.2682291667_10.csv'
        mne_df = pd.read_csv(os.path.join(bins_dir, mne_file))
        
        bin_df = pd.read_csv(os.path.join(bins_dir, bin_file))
  
        estimator_results = []
        
        for night in bin_df['night'].unique():
            bin_night_df = bin_df[bin_df['night'] == night]
            mne_night_df = mne_df[mne_df['night'] == night]

            night_precisions = []
            night_recalls = []

            for col in ['noise_HB_1', 'noise_HB_2']:
                true_labels = mne_night_df[col]
                pred_labels = bin_night_df[col]
                
                tp = np.sum((true_labels == 1) & (pred_labels == 1))
                fp = np.sum((true_labels == 0) & (pred_labels == 1))
                
                if tp + fp == 0:
                    precision = -200
                else:
                    precision = precision_score(true_labels, pred_labels, zero_division=0)

                fn = np.sum((true_labels == 1) & (pred_labels == 0))
                if tp + fn == 0:
                    recall = -100
                else:
                    recall = recall_score(true_labels, pred_labels, zero_division=0)
                
                night_precisions.append(precision)
                night_recalls.append(recall)
            
            estimator_results.append({
                'N': night,
                'P1': night_precisions[0],
                'R1': night_recalls[0],
                'P2': night_precisions[1],
                'R2': night_recalls[1],
                'P': sum(night_precisions) / len(night_precisions),
                'R': sum(night_recalls) / len(night_recalls)
            })
        
        estimator_df = pd.DataFrame(estimator_results)

        estimator_df = estimator_df[~estimator_df['N'].isin([26, 30, 33])]
        
        result_path = os.path.join(result_dir, f'{estimator_name}_results.csv')
        estimator_df.to_csv(result_path, index=False)
        results[estimator_name] = estimator_df
        
    return results

def main():
    """
    Main function to execute the noise analysis workflow.

    It loads noise data from CSV files, visualizes noise values for each estimator,
    extracts the top 100*k% noisy timesteps, computes the agreement percentages
    between estimator pairs for each feature, and visualizes these agreements as heatmaps.
    """
    noise_dict = load_noise_data()
    visualize_noise(dict=noise_dict)

    estimate_binary_noise()
    topKs = get_top_noisy_timesteps(type='bin')

    agreement_dict = compute_agreement()
    visualize_agreement(dict=agreement_dict)

    noise_as_lines()
    compare_with_mne()

if __name__ == '__main__':
    main()