"""
Evaluate model using test set
inputs: df_test_final.csv
model outputs: 'models/trained_model"
training viz outputs: 'reports'
"""

import os
import sys
from pathlib import Path
import yaml
import json
import pandas as pd
import copy

import matplotlib
import matplotlib.pyplot as plt

import logging
log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_fmt)
logger = logging.getLogger("train_model")

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

# import custom modules
import src.models.training as training

# main function
def run(config, df=None):
    
    config = copy.deepcopy(config)

    # check root directory
    logger.info('Root Directory: ' + root)

    # attach root to model_dir
    config['predictor_config']['model_dir'] = str(Path(root, config['predictor_config']['model_dir'], 'trained_model'))

    # load trained model
    logger.info('Load trained Model...')
    predictor = training.Predictor(**config['predictor_config'])

    # print predictor parameters
    logger.info('Predictor params: ')
    for k, v in config['predictor_config'].items():
        logger.info(k + ': ' + str(v))

    if config['deploy']:

        # if deployment mode, make predictions on df
        df = df.copy()

        # make predictions of df using best threshold
        logger.info('Make Predictions...')
        pred_dict = predictor.predict(df, threshold=predictor.best_threshold)

        # convert to dataframe
        pred_df = pd.DataFrame(pred_dict)

        logger.info('Predictions Complete')

        return pred_df

    else:

        # read in processed data
        df_test_final_path = Path(root, config['data_processed_dir'], 'df_val_final.csv')
        logger.info('Read in df_test_final: ' + str(df_test_final_path))
        df_test_final = pd.read_csv(df_test_final_path)

        # make predictions of test set using best threshold
        logger.info('Make Predictions...')
        pred_dict = predictor.predict(df_test_final, threshold=predictor.best_threshold)

        # convert to dataframe
        pred_df = pd.DataFrame(pred_dict)

        # output to 'models/predictions'
        pred_dir = Path(root, config['predictions_dir'])
        pred_dir.mkdir(parents=True, exist_ok=True)
        pred_df.to_csv(Path(str(pred_dir), 'df_test_pred.csv'), index=False)

        logger.info('Predictions folder: ')
        logger.info(os.listdir(Path(root, config['predictions_dir'])))

        logger.info('Predictions Complete')

        return None


if __name__ == "__main__":
    run(config)