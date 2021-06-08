import os
import sys
from pathlib import Path
import yaml
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from pprint import pformat

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
root = project_path(2)

# get config file
with open(Path(root, 'src', 'config', 'config.yaml'), 'r') as file:
    config = yaml.load(file, Loader=yaml. FullLoader)

# import custom modules
import src.features.preprocess as preprocess
import src.models.training as training

# helper functions
def preprocess_data(in_path, out_path):
    
    # load preprocessor class
    data_cleaner = preprocess.DataCleaner(
        model_dir=config['trainer_config']['model_dir'], 
        random_state=config['random_state'])

    data_raw = pd.read_csv(in_path)

    data_processed = data_cleaner.clean_data(
        df = data_raw,
        min_tokens=config['preprocess_config']['min_tokens'], 
        max_tokens=config['preprocess_config']['max_tokens'], 
        subsample=config['preprocess_config']['subsample'])

    data_processed.to_csv(out_path)

    return data_processed


# main function
def train_model():
    
    # check root directory
    logger.info('Root Directory: ' + root)

    # preprocess data
    logger.info('Processing data: ')

    data_processed = \
    preprocess_data(
        in_path = Path(root, config['data_raw_dir'], config['data_raw_filename']), 
        out_path = Path(root, config['data_processed_dir'], config['data_processed_filename'])
        )

    logger.info('Processed data folder: ')
    logger.info(os.listdir(Path(root, config['data_processed_dir'])))

    # train_test_split
    logger.info('Split Datasets: ')

    df_train, df_test = train_test_split(
        data_processed, 
        train_size=config['train_size'], 
        random_state=config['random_state'], 
        shuffle=True,
        stratify=data_processed['target'])

    df_val, df_test = train_test_split(
        df_test, 
        train_size=config['val_size'] / (config['val_size'] + config['test_size']), 
        random_state=config['random_state'], 
        shuffle=True, 
        stratify=df_test['target'])

    logger.info(
        'Train samples: ' + str(df_train.shape[0]) + '  |   ' + 
        'Val samples: ' + str(df_val.shape[0]) + '  |   ' + 
        'Test samples: ' + str(df_test.shape[0])
    )

    # attach root to out_dir
    config['trainer_config']['out_dir'] = str(Path(root, config['trainer_config']['out_dir']))

    # initialize model
    logger.info('Initialize model: ')
    trainer = training.Trainer(**config['trainer_config'])

    # print training parameters
    logger.info('Training params: ')
    for k, v in config['trainer_config'].items():
        logger.info(k + ': ' + str(v))

    # Begin Training
    logger.info('Train model: ')
    trainer.fit(df_train, df_val)

    # dump training history (convert to float)
    history = {k: [float(s) for s in v] for k, v in trainer.history.items()}
    json.dump(history, open(Path(root, config['reports_dir'], 'training_history.json'), 'w'), indent=4)

    # dump training plots
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
    train_model()