import torch
from tqdm import tqdm
import warnings
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import theilslopes, pearsonr

from .. import utils
from ..loader import *
from .model import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f'Device is {device}')

warnings.filterwarnings("ignore", category=FutureWarning)

def plot_signals(signals, batch, outlier_threshold, dpi=1200):
    num_features = signals[0][0].shape[-1]
    
    X, X_dec = signals[batch]
    fig, axes = plt.subplots(1, num_features, figsize=(10 * num_features, 5))

    for j in range(num_features):
        ax = axes[j] if num_features > 1 else axes

        X_feat = X[:, j].cpu().numpy().reshape(-1)
        X_dec_feat = X_dec[:, j].cpu().numpy().reshape(-1)

        outliers_X = np.where(np.abs(X_feat) > outlier_threshold)[0]
        outliers_X_dec = np.where(np.abs(X_dec_feat) > outlier_threshold)[0]

        X_feat[outliers_X] = 0
        X_dec_feat[outliers_X_dec] = 0

        ax.plot(X_feat, label='Raw Signal', color='C1', linewidth=0.5)
        ax.plot(X_dec_feat, label='Decoded Signal', color='C0', linewidth=0.5)

        ax.set_ylim(-outlier_threshold, outlier_threshold)
        ax.scatter(outliers_X, np.zeros_like(outliers_X), label='Raw Outliers', s=30, color='C1')
        ax.scatter(outliers_X_dec, np.zeros_like(outliers_X_dec), label='Decoded Outliers', s=30, color='C0')

        ax.set_title(f'Batch {batch} - Feature {j+1}')
        ax.set_xlabel('Samples')
        ax.set_ylabel('Amplitude')
        ax.legend()

    plt.tight_layout()

    path = utils.get_path('static', 'autoencoder', 'signals', filename=f'batch_{batch}.png')
    plt.savefig(path, dpi=dpi)
    plt.close(fig)

def compare_bands(signals, batch, threshold, num_bands=5, dpi=1200):
    num_features = signals[0][0].shape[-1]

    X, X_dec = signals[batch]
    fig, axes = plt.subplots(1, num_features, figsize=(10 * num_features, 5))

    for j in range(num_features):
        ax = axes[j] if num_features > 1 else axes

        X_feat = X[:, j].cpu().numpy().reshape(-1)
        X_dec_feat = X_dec[:, j].cpu().numpy().reshape(-1)

        lower_outliers_X = X_feat[X_feat < -threshold]
        upper_outliers_X = X_feat[X_feat > threshold]

        lower_outliers_X_dec = X_dec_feat[X_dec_feat < -threshold]
        upper_outliers_X_dec = X_dec_feat[X_dec_feat > threshold]

        X_feat_no_outliers = X_feat[(X_feat >= -threshold) & (X_feat <= threshold)]
        X_dec_feat_no_outliers = X_dec_feat[(X_dec_feat >= -threshold) & (X_dec_feat <= threshold)]

        min_val = min(X_feat_no_outliers.min(), X_dec_feat_no_outliers.min())
        max_val = max(X_feat_no_outliers.max(), X_dec_feat_no_outliers.max())
        
        bands = np.linspace(min_val, max_val, num_bands + 1)

        raw_counts, _ = np.histogram(X_feat_no_outliers, bins=bands)
        dec_counts, _ = np.histogram(X_dec_feat_no_outliers, bins=bands)

        raw_counts = np.concatenate(([len(lower_outliers_X)], raw_counts, [len(upper_outliers_X)]))
        dec_counts = np.concatenate(([len(lower_outliers_X_dec)], dec_counts, [len(upper_outliers_X_dec)]))

        bar_width = 0.35
        indices = np.arange(num_bands + 2)

        ax.bar(indices, raw_counts, bar_width, label='Raw Signal', color='C1', linewidth=0.5)
        ax.bar(indices + bar_width, dec_counts, bar_width, label='Decoded Signal', color='C0', linewidth=0.5)

        ax.set_title(f'Batch {batch} - Feature {j+1}')
        ax.set_xlabel('Bands')
        ax.set_ylabel('Sample Count')

        band_labels = [f'(-∞, {-threshold})'] + [f'[{bands[i]:.2f}, {bands[i+1]:.2f})' for i in range(num_bands)] + [f'({threshold}, ∞)']
        ax.set_xticks(indices + bar_width / 2)
        ax.set_xticklabels(band_labels)

        ax.legend()

    plt.tight_layout()

    path = utils.get_path('static', 'autoencoder', 'metrics', filename=f'batch_{batch}_bands.png')
    plt.savefig(path, dpi=dpi)
    plt.close(fig)

def calculate_zero_crossings(signals, batch, num_points=100, dpi=1200):
    num_features = signals[0][0].shape[-1]
    
    X, X_dec = signals[batch]
    fig, axes = plt.subplots(1, num_features, figsize=(10 * num_features, 5))

    for j in range(num_features):
        ax = axes[j] if num_features > 1 else axes

        X_feat = X[:, j].cpu().numpy().reshape(-1)
        X_dec_feat = X_dec[:, j].cpu().numpy().reshape(-1)

        zero_crossings_raw = np.where(np.diff(np.sign(X_feat)))[0]
        zero_crossings_dec = np.where(np.diff(np.sign(X_dec_feat)))[0]

        smoothed_raw = np.zeros_like(X_feat)
        smoothed_dec = np.zeros_like(X_dec_feat)

        window_size_raw = max(1, len(zero_crossings_raw) // num_points)
        window_size_dec = max(1, len(zero_crossings_dec) // num_points)

        for i in range(0, len(zero_crossings_raw), window_size_raw):
            end = min(i + window_size_raw, len(zero_crossings_raw))
            smoothed_raw[zero_crossings_raw[i:end]] = np.mean(X_feat[zero_crossings_raw[i:end]])

        for i in range(0, len(zero_crossings_dec), window_size_dec):
            end = min(i + window_size_dec, len(zero_crossings_dec))
            smoothed_dec[zero_crossings_dec[i:end]] = np.mean(X_dec_feat[zero_crossings_dec[i:end]])

        ax.plot(smoothed_raw, label='Raw Signal', color='C1', linewidth=0.5)
        ax.plot(smoothed_dec, label='Decoded Signal', color='C0', linewidth=0.5)

        ax.set_title(f'Batch {batch} - Feature {j+1}')
        ax.set_xlabel('Samples')
        ax.set_ylabel('Amplitude')
        ax.legend()

    plt.tight_layout()

    path = utils.get_path('static', 'autoencoder', 'metrics', filename=f'batch_{batch}_zero_crossings.png')
    plt.savefig(path, dpi=dpi)
    plt.close(fig)

def theil_slope_intercept(signals, batch, dpi=1200):
    num_features = signals[0][0].shape[-1]
    
    X, X_dec = signals[batch]
    fig, axes = plt.subplots(1, num_features, figsize=(10 * num_features, 5))

    for j in range(num_features):
        ax = axes[j] if num_features > 1 else axes

        X_feat = X[:, j].cpu().numpy().reshape(-1)
        X_dec_feat = X_dec[:, j].cpu().numpy().reshape(-1)

        slope_raw, intercept_raw, _, _ = theilslopes(X_feat)
        slope_dec, intercept_dec, _, _ = theilslopes(X_dec_feat)

        ax.plot(X_feat - (slope_raw * np.arange(len(X_feat)) + intercept_raw), label='Raw Detrended', linewidth=0.5, color='C1')
        ax.plot(X_dec_feat - (slope_dec * np.arange(len(X_dec_feat)) + intercept_dec), label='Decoded Detrended', linewidth=0.5, color='C0')

        ax.set_title(f'Batch {batch} - Feature {j+1}')
        ax.set_xlabel('Samples')
        ax.set_ylabel('Amplitude')
        ax.legend()

    plt.tight_layout()

    path = utils.get_path('static', 'autoencoder', 'metrics', filename=f'batch_{batch}_detrended.png')
    plt.savefig(path, dpi=dpi)
    plt.close(fig)

def calculate_rms(signals, batch, dpi=1200):
    num_features = signals[0][0].shape[-1]

    X, X_dec = signals[batch]
    fig, axes = plt.subplots(1, num_features, figsize=(10 * num_features, 5))

    for j in range(num_features):
        ax = axes[j] if num_features > 1 else axes

        X_feat = X[:, j].cpu().numpy().reshape(-1)
        X_dec_feat = X_dec[:, j].cpu().numpy().reshape(-1)

        differences = X_feat - X_dec_feat
        squared_differences = np.square(differences)

        ax.plot(squared_differences, label='Squared Differences', linewidth=0.5, color='C2')

        ax.set_title(f'Batch {batch} - Feature {j+1}')
        ax.set_xlabel('Samples')
        ax.set_ylabel('Squared Difference')
        ax.legend()

    plt.tight_layout()

    path = utils.get_path('static', 'autoencoder', 'metrics', filename=f'batch_{batch}_rms.png')
    plt.savefig(path, dpi=dpi)
    plt.close(fig)

def collect_metrics(signals, num_bands=5, threshold=10):
    num_features = signals[0][0].shape[-1]

    all_raw_signals = [[] for _ in range(num_features)]
    all_decoded_signals = [[] for _ in range(num_features)]

    for X, X_dec in signals:
        for j in range(num_features):
            all_raw_signals[j].append(X[:, j].cpu().numpy().reshape(-1))
            all_decoded_signals[j].append(X_dec[:, j].cpu().numpy().reshape(-1))
    
    metrics = {}

    for j in range(num_features):
        raw_signals_concat = np.concatenate(all_raw_signals[j])
        decoded_signals_concat = np.concatenate(all_decoded_signals[j])

        min_val = min(raw_signals_concat.min(), decoded_signals_concat.min())
        max_val = max(raw_signals_concat.max(), decoded_signals_concat.max())
        bands = np.linspace(min_val, max_val, num_bands + 1)

        lower_outliers_raw = raw_signals_concat[raw_signals_concat < -threshold]
        upper_outliers_raw = raw_signals_concat[raw_signals_concat > threshold]

        lower_outliers_decoded = decoded_signals_concat[decoded_signals_concat < -threshold]
        upper_outliers_decoded = decoded_signals_concat[decoded_signals_concat > threshold]

        raw_counts, _ = np.histogram(raw_signals_concat, bins=bands)
        dec_counts, _ = np.histogram(decoded_signals_concat, bins=bands)

        raw_counts = np.concatenate(([len(lower_outliers_raw)], raw_counts, [len(upper_outliers_raw)]))
        dec_counts = np.concatenate(([len(lower_outliers_decoded)], dec_counts, [len(upper_outliers_decoded)]))

        band_rate = (raw_counts - dec_counts) / raw_counts

        zero_crossings_raw = np.sum(np.where(np.diff(np.sign(raw_signals_concat)))[0])
        zero_crossings_decoded = np.sum(np.where(np.diff(np.sign(decoded_signals_concat)))[0])

        zero_crossings_rate = (zero_crossings_decoded / zero_crossings_raw) if zero_crossings_raw > 0 else 0

        differences = raw_signals_concat - decoded_signals_concat
        rms_value = np.sqrt(np.mean(np.square(differences)))

        pearson_corr, _ = pearsonr(raw_signals_concat, decoded_signals_concat)

        metrics[f'Feature_{j+1} Differences'] = {
            'band_rate': band_rate.tolist(),
            'zero_crossings_rate': float(zero_crossings_rate),
            'rms_value': float(rms_value),
            'pearson_correlation': float(pearson_corr)
        }

    fn = utils.get_path('static', 'autoencoder', filename='concat_signals_metrics.json')
    utils.save_json(data=metrics, filename=fn)

    logger.info(f'Metrics saved to {fn}.')

def test(data, criterion, model, visualize=False):
    mfn = utils.get_path('models', filename='autoencoder.pth')

    model.load_state_dict(torch.load(mfn))
    model.to(device)
    model.eval()

    batches = len(data)
    total_test_loss = 0.0
    signals = []

    progress_bar = tqdm(enumerate(data), total=batches, desc=f'Evaluation', leave=True)

    with torch.no_grad():
        for _, (X, _) in progress_bar:
            X = X[:, :, :2].to(device)

            X_dec, _ = model(X)

            test_loss = criterion(X_dec, X)

            total_test_loss += test_loss.item()
            progress_bar.set_postfix(Loss=test_loss.item())

            signals.append((X, X_dec))

        if visualize:
            plot_signals(signals, batch=4, outlier_threshold=10)
            compare_bands(signals, batch=4, threshold=10, num_bands=5)
            calculate_zero_crossings(signals, batch=4, num_points=100)
            theil_slope_intercept(signals, batch=4)
            calculate_rms(signals, batch=4)

            collect_metrics(signals, num_bands=5, threshold=10)

        avg_test_loss = total_test_loss / batches

    logger.info(f'\nTesting complete!\nTesting Loss: {avg_test_loss:.6f}\n')

def main():
    samples, chunks = 7680, 32
    seq_len = samples // chunks

    bitbrain_dir = utils.get_dir('data', 'bitbrain')
    raw_dir = utils.get_dir('data', 'raw')

    get_boas_data(base_path=bitbrain_dir, output_path=raw_dir)
    
    datapaths = split_data(dir=raw_dir, train_size=43, val_size=3, test_size=10)
    
    _, _, test_df = get_dataframes(datapaths, samples=samples, seq_len=seq_len, exist=True)

    datasets = create_datasets(dataframes=(test_df,), seq_len=seq_len)

    dataloaders = create_dataloaders(datasets, batch_size=512, drop_last=True)

    model = Autoencoder(seq_len=seq_len,
                        num_feats=2, 
                        latent_seq_len=1, 
                        latent_num_feats=16, 
                        hidden_size=32, 
                        num_layers=1,
                        dropout=0.05)
    
    test(data=dataloaders[0],
         criterion=utils.BlendedLoss(p=1.0, blend=0.1),
         model=model,
         visualize=True)

if __name__ == '__main__':
    main()