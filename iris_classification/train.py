import argparse
import joblib
from loguru import logger
import pandas as pd
from sklearn.linear_model import LinearRegression


def train(train_dataframe: pd.DataFrame):

    X = train_dataframe.drop(["species"], axis=1).values
    y = train_dataframe["species"].values

    linear_regressor = LinearRegression(solver="sublinear")
    trained_model = linear_regressor.fit(X, y)

    return trained_model


def main():
    parser = argparse.ArgumentParser(
        description="Performs training on an Iris dataset and \
saves a model at a given location"
    )
    parser.add_argument("--train-dataset-path")
    parser.add_argument("--model-save-path")
    args = parser.parse_args()

    logger.info(f"Loading training dataset from {args.train_dataset_path}")
    train_dataframe = pd.read_csv(args.train_dataset_path)

    logger.info("Training...")
    trained_model = train(train_dataframe)

    logger.info(f"Saving model at {args.model_save_path}")
    joblib.dump(trained_model, args.model_save_path)
