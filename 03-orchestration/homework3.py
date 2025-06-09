#!/usr/bin/env python
# coding: utf-8

import pandas as pd

import pickle

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import root_mean_squared_error

import xgboost as xgb

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll import scope

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error

from pathlib import Path

models_folder = Path('models')
models_folder.mkdir(exist_ok=True)


import mlflow


mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("nyc-taxi-experiment2")

# print("**************")

def read_dataframe(year, month):
    url = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{month:02d}.parquet'
    print("Reading data from:", url)
    df = pd.read_parquet(url)

    print(len(df), "rows in the dataset")

    # df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    # df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)
    df['duration'] = (df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]).dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)

    # df['PU_DO'] = df['PULocationID'] + '_' + df['DOLocationID']

    print("size of dataframe:", df.shape[0]* df.shape[1], "elements")

    return df

def create_X(df, dv=None):

    categorical = ['PULocationID', 'DOLocationID']
    numerical = ['trip_distance']

    dicts = df[categorical + numerical].to_dict(orient='records')

    if dv is None:
        dv = DictVectorizer(sparse=True)
        X = dv.fit_transform(dicts)
    else:
        X = dv.transform(dicts)

    return X, dv

def train_model(X_train, y_train, X_val, y_val, dv):
    
    with mlflow.start_run() as run:
        mlflow.set_tag("model", "liner regression")

        mlflow.log_params(best_params)

    lr = LinearRegression()
    lr.fit(X_train, y_train)

    print(f"Intercept of model:" ,lr.intercept_)

    y_pred = lr.predict(X_val)
    rmse = root_mean_squared_error(y_val, y_pred)
    mlflow.log_metric("rmse", rmse)

    with open("models/preprocessor.b", "wb") as f_out:
        pickle.dump(dv, f_out)
    mlflow.log_artifact("models/preprocessor.b", artifact_path="preprocessor")

    mlflow.log_model(lr, artifact_path="models_mlflow")

    return run.info.run_id

def run(year, month):
    df_train = read_dataframe(year=year, month=month)

    if month == 12:
        year += 1
        month = 1
    else:
        month += 1

    print("****************")

    df_val = read_dataframe(year=year, month=month)

    X_train, dv = create_X(df_train)
    X_val, _ = create_X(df_val, dv=dv)

    y_train = df_train['duration'].values
    y_val = df_val['duration'].values

    run_id = train_model(X_train, y_train, X_val, y_val, dv)

    return run_id
    # print("****************")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train a model to predict taxi trip duration")
    parser.add_argument('--year', type=int, default=2023, help='Year of the data')
    parser.add_argument('--month', type=int, default=1, help='Month of the data')
    args = parser.parse_args()

    run_id = run(year=args.year, month=args.month)
    #save run_id to a file
    with open("run_id.txt", "w") as f:
        f.write(run_id)

