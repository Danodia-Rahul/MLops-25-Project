import mlflow
import pandas as pd
import numpy as np
import pickle
from prefect import flow, task
import xgboost as xgb
from sklearn.metrics import root_mean_squared_error
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer



@task()
def setup_mlflow():

    tracking_uri = 'http://127.0.0.1:5000'
    experiment_name = 'Kaggle Season-4 Episode-12'
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)


@task()
def prepare_data(path='./Data/train.csv'):

    target = 'Premium Amount'

    features = ['Age', 'Annual Income', 'Number of Dependents',
                'Occupation', 'Credit Score', 'Property Type']

    df = pd.read_csv(path)
    df = df[features + [target]].copy()

    categorical = df.select_dtypes(include=['object']).columns.tolist()

    cat_col_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
    ])

    transformer = ColumnTransformer(transformers=[
        ('cat', cat_col_transformer, categorical)
    ], remainder='passthrough')

    transformed = transformer.fit_transform(df[features])

    return transformed, df[target]

@task()
def train_model(X, y):

    with mlflow.start_run():

        model = xgb.XGBRegressor()
        model.load_model('./model.xgb')

        preds = model.predict(X)
        rmse = root_mean_squared_error(y, preds)
        mlflow.log_metric("rmse", rmse)

        mlflow.log_artifact('preprocessor.bin')

@flow()
def run():
     
    setup_mlflow()
    X_train, y_train = prepare_data()
    train_model(X_train, y_train)



if __name__ == '__main__':
    run()

