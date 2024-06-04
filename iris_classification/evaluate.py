import argparse

import joblib
import pandas as pd
from loguru import logger
from sklearn.metrics import accuracy_score, precision_score, recall_score


def evaluate(iris_model, test_dataframe):
    """Evaluate model on test data, and print the evaluation
    metrics."""

    X_test = test_dataframe.drop("species", axis=1).values
    y_test = test_dataframe["species"].values

    predictions = iris_model.predict(X_test)

    accuracy = accuracy_score(y_true=y_test, y_pred=predictions)
    precision = precision_score(
        y_true=y_test,
        y_pred=predictions,
        average="binary",
    )
    recall = recall_score(y_true=y_test, y_pred=predictions, average="binary")

    metrics = {"accuracy": accuracy, "precision": precision, "recall": recall}
    print(metrics)


def main():
    parser = argparse.ArgumentParser(
        description="Retrieves a model from a given location, apply it on a \
test dataset and prints some evaluation metrics"
    )
    parser.add_argument("--model-path", "-m")
    parser.add_argument("--test-dataset-path", "-d")
    args = parser.parse_args()

    logger.info(f"Loading model from {args.model_path}")
    iris_model = joblib.load(args.model_path)
    logger.info(f"Loading test dataset from {args.test_dataset_path}")
    test_dataframe = pd.read_csv(args.test_dataset_path)

    logger.info("Evaluating the model...")
    evaluate(iris_model, test_dataframe)
