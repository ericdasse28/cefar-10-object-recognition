import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def prepare(path_to_iris_data: Path) -> pd.DataFrame:
    """Perform label encoding on the target variable of the original Iris
    dataset and splits it into training and test datasets"""

    iris_data = pd.read_csv(path_to_iris_data)

    # Column names were missing. Add them for better understandability
    iris_data.columns = [
        "sepal_length_cm",
        "sepal_width_cm",
        "petal_length_cm",
        "petal_width_cm",
        "species",
    ]

    # Turn target variable into integer
    encoder = LabelEncoder()
    iris_data["species"] = encoder.fit_transform(iris_data["species"])

    # Split into training and test data
    train_dataset, test_dataset = train_test_split(
        iris_data, test_size=0.33, random_state=4
    )

    return train_dataset, test_dataset


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "filename",
        help="Location of the original Iris dataset.",
    )
    parser.add_argument(
        "--save-folder",
        help="Folder where the training and test datasets are saved.",
    )
    args = parser.parse_args()

    train_dataset, test_dataset = prepare(args.filename)
    datasets_save_folder = Path(args.save_folder)
    train_dataset.to_csv(datasets_save_folder / "train.csv", index=False)
    test_dataset.to_csv(datasets_save_folder / "test.csv", index=False)
