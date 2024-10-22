import yaml
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import theilslopes, pearsonr
from . import utils

logger = utils.get_logger(level='DEBUG')
warnings.filterwarnings("ignore", category=FutureWarning)

with open('config.yaml', 'r') as config_file:
    config = yaml.safe_load(config_file)

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
            
            noise_cols = [col for col in df.columns if col.startswith('noise_')]
            noise_dict[id] = {col.replace('noise_', ''): None for col in noise_cols}
            
            for col in noise_dict[id]:
                noise_dict[id][col] = apply(values=df[f'noise_{col}'].tolist(), func=transformation)

        except FileNotFoundError:
            logger.error(f"CSV file for {id} not found at {csv_path}")

if __name__ == '__main__':
    main()
