"""
Run this script to execure E2E process for training / evaluation
"""

import src.data.make_dataset
import src.features.build_features
import src.models.train_model
import src.models.eval_model
import src.visualization.visualize

if __name__ == "__main__":
    src.data.make_dataset.run()
    src.features.build_features.run()
    src.models.train_model.run()
    src.models.eval_model.run()
    src.visualization.visualize.run()