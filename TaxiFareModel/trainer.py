from encoders import DistanceTransformer
from encoders import TimeFeaturesEncoder
from data import get_data
from utils import compute_rmse
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np


class Trainer():
    def __init__(self):
        df = get_data()
        y = df.pop('fare_amount')
        X = df

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def set_pipeline(self):
        '''returns a pipelined model'''
        dist_pipe = Pipeline([
        ('dist_trans', DistanceTransformer()),
        ('stdscaler', StandardScaler())
        ])
        time_pipe = Pipeline([
        ('time_enc', TimeFeaturesEncoder('pickup_datetime')),
        ('ohe', OneHotEncoder(handle_unknown='ignore'))
        ])
        preproc_pipe = ColumnTransformer([
        ('distance', dist_pipe, ["pickup_latitude", "pickup_longitude", 'dropoff_latitude', 'dropoff_longitude']),
        ('time', time_pipe, ['pickup_datetime'])
        ], remainder="drop")
        pipe = Pipeline([
        ('preproc', preproc_pipe),
        ('linear_model', LinearRegression())
        ])
        return pipe


    def train(self):
        '''returns a trained pipelined model'''
        return self.set_pipeline().fit(self.X_train, self.y_train)


    def evaluate(self):
        '''returns the value of the RMSE'''
        y_pred = self.train().predict(self.X_test)
        rmse = compute_rmse(y_pred, self.y_test)
        print(rmse)
        return rmse


if __name__ == "__main__":
    trainer = Trainer()
    trainer.train()
    print(trainer.evaluate())
