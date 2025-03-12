import logging

import pandas as pd
from pandas import DataFrame
from sklearn.model_selection import train_test_split as tts

from src.data_preprocessing.feature_extractor import extract_features


def train_test_split(csv_file_path: str, test_size: float = 0.15):
    genes_data_frame: DataFrame = pd.read_csv(csv_file_path)
    sequences = genes_data_frame["Sequence"].values
    labels = genes_data_frame["IsGene"].values

    logging.info("Extracting features...")
    X = extract_features(sequences)
    logging.info("Features extracted.")
    logging.info("Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = tts(X, labels, test_size=0.15, random_state=42)
    logging.info("Data split complete.")

    return X_train, X_test, y_train, y_test
