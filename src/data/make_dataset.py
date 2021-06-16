"""
Run this script to perform data cleaning / formatting on full dataset
Excludes feature engineering - done seperately on train / val / test
    input: raw data
    output: df_train_cleaned.csv, df_val_cleaned.csv, df_test_cleaned.csv
"""

from sklearn.model_selection import train_test_split
import os
import sys
from pathlib import Path
import pandas as pd
import yaml
import copy

import logging
log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_fmt)
logger = logging.getLogger("make_dataset")

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
    config = yaml.load(file, Loader=yaml.FullLoader)

from src.data.data_cleaner import DataCleaner

# Splits data into train / test / split
def run(config):

    config = copy.deepcopy(config)

    # check root directory
    logger.info('Root Directory: ' + root)

    # read in raw data
    raw_data_path =  Path(root, config['data_raw_dir'], config['data_raw_filename'])
    logger.info('Read in raw data: ' + str(raw_data_path))
    data_raw = pd.read_csv(raw_data_path)

    # preprocess data
    logger.info('Processing data...')

    if config['deploy']:

        # if in deployment mode, do not do train / test / split, just return cleaned data
        
        # attach root to model_dir (use predictor dir for deployment)
        config['predictor_config']['model_dir'] = str(Path(root, config['predictor_config']['model_dir'], 'trained_model'))
        
        # load preprocessor class
        data_cleaner = DataCleaner(
            model_dir=config['predictor_config']['model_dir'], 
            random_state=config['random_state'])

        data_processed = data_cleaner.clean_data(
            df = data_raw,
            min_tokens=config['preprocess_config']['min_tokens'], 
            max_tokens=config['preprocess_config']['max_tokens'], 
            subsample=config['preprocess_config']['subsample'])
        
        return data_processed

    else: 

        # load preprocessor class
        data_cleaner = DataCleaner(
            model_dir=config['trainer_config']['model_dir'], 
            random_state=config['random_state'])

        data_processed = data_cleaner.clean_data(
            df = data_raw,
            min_tokens=config['preprocess_config']['min_tokens'], 
            max_tokens=config['preprocess_config']['max_tokens'], 
            subsample=config['preprocess_config']['subsample'])

        # train_test_split
        logger.info('Split Datasets...')

        df_train_cleaned, df_test_cleaned = train_test_split(
            data_processed, 
            train_size=config['train_size'], 
            random_state=config['random_state'], 
            shuffle=True,
            stratify=data_processed['target'])

        df_val_cleaned, df_test_cleaned = train_test_split(
            df_test_cleaned, 
            train_size=config['val_size'] / (config['val_size'] + config['test_size']), 
            random_state=config['random_state'], 
            shuffle=True, 
            stratify=df_test_cleaned['target'])

        logger.info(
            'df_train_cleaned shape: ' + str(df_train_cleaned.shape) + '  |   ' + 
            'df_val_cleaned shape: ' + str(df_val_cleaned.shape) + '  |   ' + 
            'df_test_cleaned shape: ' + str(df_test_cleaned.shape)
        )

        # write to interim folder
        df_train_cleaned.to_csv(Path(root, config['data_interim_dir'], 'df_train_cleaned.csv'), index = False)
        df_val_cleaned.to_csv(Path(root, config['data_interim_dir'], 'df_val_cleaned.csv'), index = False)
        df_test_cleaned.to_csv(Path(root, config['data_interim_dir'], 'df_test_cleaned.csv'), index = False)
        
        logger.info('Interim data folder: ')
        logger.info(os.listdir(Path(root, config['data_interim_dir'])))

        return None


if __name__ == "__main__":
    run(config)