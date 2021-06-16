"""
Run this script to execure E2E process for training / evaluation
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
with open(Path(root, 'src', 'config', 'config.yaml'), 'r') as file:
    config = yaml.load(file, Loader=yaml.FullLoader)

# import modules
import src.data.make_dataset
import src.features.build_features
import src.models.train_model
import src.models.predict_model
import src.visualization.visualize

def run():
    src.data.make_dataset.run(config)
    src.features.build_features.run(config)
    src.models.train_model.run(config)
    src.models.predict_model.run(config)
    src.visualization.visualize.run(config)

if __name__ == "__main__":
    run()