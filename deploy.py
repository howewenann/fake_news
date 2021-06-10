"""
Run this script to execure E2E process from raw to prediction
"""
from pathlib import Path
import yaml
import sys
import os

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
with open(Path(root, 'src', 'config', 'config_deploy.yaml'), 'r') as file:
    config = yaml.load(file, Loader=yaml. FullLoader)

# import modules
import src.data.make_dataset
import src.features.build_features
import src.models.train_model
import src.models.predict_model
import src.visualization.visualize

def run():
    df = src.data.make_dataset.run(config)
    df = src.features.build_features.run(config, df)
    df = src.models.predict_model.run(config, df)
    df = src.visualization.visualize.run(config, df)
    
    # write to predictions folder
    df.to_csv(Path(root, config['predictions_dir'], 'df_test_pred_deploy.csv'), index=False)

if __name__ == "__main__":
    run()
