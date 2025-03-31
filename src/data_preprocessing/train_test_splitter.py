import argparse
import logging
import os
from typing import Tuple

import pandas as pd
from pandas import DataFrame
from sklearn.model_selection import train_test_split

from src.config import CSV_FILE_PATH, PROCESSED_DATA_FOLDER_PATH


def save_train_test_validation_csvs(
    data_splits: dict,
) -> Tuple[str, str, str]:
    logging.info("Saving training, validation, and testing CSVs...")

    train_csv_path = os.path.join(PROCESSED_DATA_FOLDER_PATH, "training.csv")
    val_csv_path = os.path.join(PROCESSED_DATA_FOLDER_PATH, "validation.csv")
    test_csv_path = os.path.join(PROCESSED_DATA_FOLDER_PATH, "testing.csv")

    pd.DataFrame(
        {"Sequence": data_splits["X_train"], "IsGene": data_splits["y_train"]}
    ).to_csv(train_csv_path, index=False)
    pd.DataFrame(
        {"Sequence": data_splits["X_val"], "IsGene": data_splits["y_val"]}
    ).to_csv(val_csv_path, index=False)
    pd.DataFrame(
        {"Sequence": data_splits["X_test"], "IsGene": data_splits["y_test"]}
    ).to_csv(test_csv_path, index=False)

    return train_csv_path, val_csv_path, test_csv_path


def train_test_validation_split(
    csv_file_path: str, test_size: float = 0.20, validation_size: float = 0.20
):
    genes_data_frame: DataFrame = pd.read_csv(csv_file_path)
    sequences = genes_data_frame["Sequence"].tolist()
    labels = genes_data_frame["IsGene"].values

    logging.info("Splitting data into training, validation, and testing sets...")
    X_train, X_temp, y_train, y_temp = train_test_split(
        sequences, labels, test_size=(test_size + validation_size), random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=(test_size / (test_size + validation_size)),
        random_state=42,
    )
    logging.info("Data split complete.")

    return (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv_file_path",
        type=str,
        required=False,
        default=f"{CSV_FILE_PATH}",
        help="Path to the CSV file containing sequences",
    )
    args = parser.parse_args()

    X_train, X_val, X_test, y_train, y_val, y_test = train_test_validation_split(
        args.csv_file_path
    )

    save_train_test_validation_csvs(
        {
            "X_train": X_train,
            "X_val": X_val,
            "X_test": X_test,
            "y_train": y_train,
            "y_val": y_val,
            "y_test": y_test,
        }
    )
