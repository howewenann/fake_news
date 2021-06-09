"""
Dummy class to simlulate feature engineering
"""

class FeatureBuilder():

    def __init__(self):
        pass

    def fit(self, df):
        return self

    def transform(self, df):
        df = df.copy()
        return df