"""
Run this script to perform feature engineering seperately on train / test / val
    input: df_train_cleaned.csv, df_val_cleaned.csv, df_test_cleaned.csv
    output: df_train_final.csv, df_val_final.csv, df_test_final.csv
"""

import os
import sys
from pathlib import Path
import yaml
import pandas as pd

import logging
log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_fmt)
logger = logging.getLogger("build_features")

def project_path(level):
    depth = ['..'] * level
    depth = '/'.join(depth)
    module_path = os.path.abspath(depth)
    # print(module_path)
    if module_path not in sys.path:
        sys.path.append(module_path)
    
    return module_path

# set root directory for importing modules
root = project_path(0)

# get config file
with open(Path(root, 'src', 'config', 'config.yaml'), 'r') as file:
    config = yaml.load(file, Loader=yaml. FullLoader)

from src.features.feature_builder import FeatureBuilder

# Splits data into train / test / split
def run():

    # check root directory
    logger.info('Root Directory: ' + root)

    # read in cleaned data
    df_train_cleaned_path = Path(root, config['data_interim_dir'], 'df_train_cleaned.csv')
    df_val_cleaned_path = Path(root, config['data_interim_dir'], 'df_val_cleaned.csv')
    df_test_cleaned_path = Path(root, config['data_interim_dir'], 'df_test_cleaned.csv')

    logger.info('Read in df_train_cleaned: ' + str(df_train_cleaned_path))
    logger.info('Read in df_val_cleaned: ' + str(df_val_cleaned_path))
    logger.info('Read in df_test_cleaned: ' + str(df_test_cleaned_path))

    df_train_cleaned = pd.read_csv(df_train_cleaned_path)
    df_val_cleaned = pd.read_csv(df_val_cleaned_path)
    df_test_cleaned = pd.read_csv(df_test_cleaned_path)

    # Build features
    logger.info('Building Features...')
    feature_builder = FeatureBuilder()
    feature_builder.fit(df_train_cleaned)

    df_train_final = feature_builder.transform(df_train_cleaned)
    df_val_final = feature_builder.transform(df_val_cleaned)
    df_test_final = feature_builder.transform(df_test_cleaned)

    logger.info('Building Features Complete')

    logger.info(
        'df_train_final shape: ' + str(df_train_final.shape) + '  |   ' + 
        'df_val_final shape: ' + str(df_val_final.shape) + '  |   ' + 
        'df_test_final shape: ' + str(df_test_final.shape)
    )

    # write to processed folder
    df_train_final.to_csv(Path(root, config['data_processed_dir'], 'df_train_final.csv'), index = False)
    df_val_final.to_csv(Path(root, config['data_processed_dir'], 'df_val_final.csv'), index = False)
    df_test_final.to_csv(Path(root, config['data_processed_dir'], 'df_test_final.csv'), index = False)
    
    logger.info('Processed data folder: ')
    logger.info(os.listdir(Path(root, config['data_processed_dir'])))

if __name__ == "__main__":
    run()