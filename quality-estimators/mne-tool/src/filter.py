import warnings
import pandas as pd
import mne
from . import utils

logger = utils.get_logger(level='DEBUG')

warnings.filterwarnings("ignore", category=FutureWarning)

def main():
    """
    Main function to filter data in test.csv using MNE's filter_data function
    and save it as estim_mne.csv.
    """
    csv_path = utils.get_path('..', '..', 'data', 'proc', filename='test.csv')
    mne_csv_path = utils.get_path('..', '..', 'data', 'proc', filename='estim_mne.csv')

    df = pd.read_csv(csv_path)
    logger.debug(f"Loaded CSV from {csv_path}")

    cols = (['HB_1', 'HB_2'], ['time', 'seq_id', 'night'], ['majority'])
    data_columns = cols[0]
    metadata_columns = cols[1] + cols[2]

    data = df[data_columns].to_numpy().T
    
    sfreq = utils.get_fs(path=csv_path)
    logger.info(f"Sampling frequency is: {sfreq} Hz.")

    filtered_data = mne.filter.filter_data(data, sfreq=sfreq, l_freq=None, h_freq=None)
    filtered_df = pd.DataFrame(filtered_data.T, columns=data_columns)

    for col in data_columns:
        df[f'noise_{col}'] = df[col] - filtered_df[col]

    result_dict = {
        'features': df[data_columns].to_dict(orient='list'),
        'metadata': df[metadata_columns].to_dict(orient='list'),
        'noise': {f'noise_{col}': df[f'noise_{col}'].tolist() for col in data_columns}
    }

    final_df = pd.concat([filtered_df, df[['noise_HB_1', 'noise_HB_2']] + df[metadata_columns]], axis=1)
    final_df.to_csv(mne_csv_path, index=False)
    logger.info(f"Filtered data and noise columns saved to {mne_csv_path}.")

    logger.debug(f"Resulting dictionary: {result_dict}")

if __name__ == '__main__':
    main()
