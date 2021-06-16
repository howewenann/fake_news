"""
Create classification report, confusion matrix, AUROC, AUPRC, Attention weights
inputs: df_test_pred.csv
training viz outputs: 'reports/figures'
dataset with viz data: 'models/predictions'
"""

import os
import sys
from pathlib import Path
import yaml
import json
import pandas as pd
import copy

from sklearn.metrics import classification_report

import logging
log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_fmt)
logger = logging.getLogger("visualize")

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
from src.visualization.visualizer import AttentionVisualizer

# helper functions
def subset_data_results(df, target, pred, ascending, head=30):

    df = df.copy()
    df_out = \
        df[(df['target'] == target) & (df['pred'] == pred)]\
            .sort_values('pred_proba', ascending = ascending).head(head)

    return df_out


def create_html_fig(l, l_names, out_dir):
    for i in range(len(l)):

        data = l[i]
        file_path = Path(out_dir, l_names[i] + '.html')
        logger.info('HTML File: ' + str(file_path))

        with open(file_path, 'w') as file:
            file.write('<br><br>'.join(data.html))

    return None


def run(config, df=None):

    config = copy.deepcopy(config)

    # check root directory
    logger.info('Root Directory: ' + root)

    # attach root to model_dir
    config['predictor_config']['model_dir'] = str(Path(root, config['predictor_config']['model_dir'], 'trained_model'))

    # initialize attention visualizer
    attention_visualizer = AttentionVisualizer(config['predictor_config']['model_dir'])

    if config['deploy']:

        # if deployment mode, create html and append to df
        df = df.copy()

        # generate attention visualization
        logger.info('Generate Attention Visualization...')
        df = attention_visualizer.visualize_attention(df)

        return df

    else:

        # read in processed data
        df_test_pred_path = Path(root, config['predictions_dir'], 'df_test_pred.csv')
        logger.info('Read in df_test_pred: ' + str(df_test_pred_path))
        df_test_pred = pd.read_csv(df_test_pred_path)

        # Create confusion matrix
        logger.info('Generate Confusion Matrix...')
        cm = pd.crosstab(
            'Class ' + df_test_pred['target'].astype(str), 
            'Class ' + df_test_pred['pred'].astype(str)
            )
        cm.to_html(Path(config['figures_dir'], 'confusion_matrix.html'))

        # create classification report
        logger.info('Generate Classification Report...')
        report = classification_report(df_test_pred['target'], df_test_pred['pred'], output_dict=True)
        report_df = pd.DataFrame(report)
        report_df.to_html(Path(config['figures_dir'], 'classification_report.html'))

        # generate attention visualization
        logger.info('Generate Attention Visualization...')
        df_test_pred = attention_visualizer.visualize_attention(df_test_pred)

        # get all data types
        true_positives = subset_data_results(df_test_pred, target=1, pred=1, ascending=False, head=30)
        true_negatives = subset_data_results(df_test_pred, target=0, pred=0, ascending=False, head=30)
        false_positives = subset_data_results(df_test_pred, target=0, pred=1, ascending=False, head=30)
        false_negatives = subset_data_results(df_test_pred, target=1, pred=0, ascending=False, head=30)

        # output attention visualizations
        logger.info('Output Attention Visualization...')
        create_html_fig(
            l = [true_positives, true_negatives, false_positives, false_negatives], 
            l_names = ['true_positives', 'true_negatives', 'false_positives', 'false_negatives'],
            out_dir = str(Path(root, config['figures_dir']))
            )

        return None


if __name__ == "__main__":
    run(config)