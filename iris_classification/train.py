import argparse
import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression


def train(train_dataframe: pd.DataFrame):

    X = train_dataframe.drop(["species"], axis=1).values
    y = train_dataframe["species"].values

    linear_regressor = LinearRegression(solver="sublinear")
    trained_model = linear_regressor.fit(X, y)

    return trained_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dataset-path")
    parser.add_argument("--model-save-path")
    args = parser.parse_args()

    train_dataframe = pd.read_csv(args.train_dataset_path)
    trained_model = train(train_dataframe)

    joblib.dump(trained_model, args.model_save_path)
