"""
Trains model using training and validation set
inputs: df_train_final.csv, df_val_final.csv
model outputs: 'models/trained_model"
training viz outputs: 'reports"
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
def run(config):
    
    config = copy.deepcopy(config)

    # check root directory
    logger.info('Root Directory: ' + root)

    # read in processed data
    df_train_final_path = Path(root, config['data_processed_dir'], 'df_train_final.csv')
    df_val_final_path = Path(root, config['data_processed_dir'], 'df_val_final.csv')

    logger.info('Read in df_train_final: ' + str(df_train_final_path))
    logger.info('Read in df_val_final: ' + str(df_val_final_path))

    df_train_final = pd.read_csv(df_train_final_path)
    df_val_final = pd.read_csv(df_val_final_path)

    # attach root to out_dir
    config['trainer_config']['out_dir'] = str(Path(root, config['trainer_config']['out_dir']))

    # initialize model
    logger.info('Initialize model...')
    trainer = training.Trainer(**config['trainer_config'])

    # print training parameters
    logger.info('Training params: ')
    for k, v in config['trainer_config'].items():
        logger.info(k + ': ' + str(v))

    # Begin Training
    logger.info('Train model...')
    trainer.fit(df_train_final, df_val_final)

    logger.info('Training Complete')

    # dump training history (convert to float)
    logger.info('Dump Training History...')
    history = {k: [float(s) for s in v] for k, v in trainer.history.items()}
    json.dump(history, open(Path(root, config['reports_dir'], 'training_history.json'), 'w'), indent=4)

    # dump training plots
    logger.info('Dump Training Plots...')
    for metric in trainer.metrics_names:
        save_img_name = 'training_' + metric + '.png'
        epochs = len(history['train_' + metric])

        plt.plot(list(range(epochs)), history['train_' + metric], label = 'train ' + metric)
        plt.plot(list(range(epochs)), history['val_' + metric], label = 'val ' + metric)
        plt.legend()

        plt.savefig(Path(root, config['figures_dir'], save_img_name))

        # clear figure after plotting
        plt.clf()


if __name__ == "__main__":
    run(config)