import logging
from typing import Tuple

import pandas as pd
from pandas import DataFrame
from sklearn.model_selection import train_test_split

from src.data_preprocessing.feature_extractor import extract_features


def train_test_validation_split(
    csv_file_path: str, test_size: float = 0.20, validation_size: float = 0.20
) -> Tuple[DataFrame, DataFrame, DataFrame, DataFrame, DataFrame, DataFrame]:
    genes_data_frame: DataFrame = pd.read_csv(csv_file_path)
    sequences = genes_data_frame["Sequence"].values
    labels = genes_data_frame["IsGene"].values

    logging.info("Extracting features...")
    X = extract_features(sequences)
    logging.info("Features extracted.")

    logging.info("Splitting data into training, validation, and testing sets...")
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, labels, test_size=(test_size + validation_size), random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=(test_size / (test_size + validation_size)),
        random_state=42,
    )
    logging.info("Data split complete.")

    return X_train, X_val, X_test, y_train, y_val, y_test
