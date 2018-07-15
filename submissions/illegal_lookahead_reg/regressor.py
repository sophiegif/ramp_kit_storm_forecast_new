import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.preprocessing import Imputer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator


class Regressor(BaseEstimator):
    def __init__(self):
        self.reg = linear_model.LinearRegression()

    def fit(self, X_df, y):
        X_df.index = range(len(X_df))
        X_df_new = pd.concat(
            [X_df.get(['instant_t', 'windspeed', 'latitude', 'longitude',
                       'hemisphere', 'Jday_predictor', 'initial_max_wind',
                       'max_wind_change_12h', 'dist2land']),
             pd.get_dummies(X_df.nature, prefix='nature', drop_first=True)],
            # 'basin' is not used here ..but it can!
            axis=1)

        # get data from the future (to test the error message!)
        rolled_windspeeds = np.roll(X_df['windspeed'].values, +2)
        X_df_new = X_df_new.assign(
            future_windspeed=pd.Series(rolled_windspeeds))

        X_df_new = X_df_new.fillna(-1)
        XX = X_df_new.values

        self.reg.fit(XX, y)

    def predict(self, X_df):
        X_df.index = range(len(X_df))
        X_df_new = pd.concat(
            [X_df.get(['instant_t', 'windspeed', 'latitude', 'longitude',
                       'hemisphere', 'Jday_predictor', 'initial_max_wind',
                       'max_wind_change_12h', 'dist2land']),
             pd.get_dummies(X_df.nature, prefix='nature', drop_first=True)],
            # 'basin' is not used here ..but it can!
            axis=1)

        # get data from the future (to test the error message!)
        rolled_windspeeds = np.roll(X_df['windspeed'].values, +2)
        X_df_new = X_df_new.assign(
            future_windspeed=pd.Series(rolled_windspeeds))

        X_df_new = X_df_new.fillna(-1)
        XX = X_df_new.values

        return self.reg.predict(XX)
