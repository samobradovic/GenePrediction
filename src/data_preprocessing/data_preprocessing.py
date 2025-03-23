import logging
from typing import List, Tuple

import pandas as pd
from pandas import DataFrame
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src.config import KMER_SIZE, VECTORIZER_FILE_PATH
from src.helpers.filesystem_helper import save_pkl


def train_test_validation_split(
    csv_file_path: str, test_size: float = 0.20, validation_size: float = 0.20
) -> Tuple[DataFrame, DataFrame, DataFrame, DataFrame, DataFrame, DataFrame]:
    """
    Splits the data from a CSV file into training, validation, and test sets.

    Args:
        csv_file_path (str): The path to the CSV file containing the sequences and
            labels.
        test_size (float): The proportion of the dataset to include in the test split.
            Default is 0.20.
        validation_size (float): The proportion of the dataset to include in the
            validation split. Default is 0.20.

    Returns:
        Tuple[DataFrame, DataFrame, DataFrame, DataFrame, DataFrame, DataFrame]:
            - X_train: The training feature set.
            - X_val: The validation feature set.
            - X_test: The test feature set.
            - y_train: The training labels.
            - y_val: The validation labels.
            - y_test: The test labels.
    """
    genes_data_frame: DataFrame = pd.read_csv(csv_file_path)
    sequences = genes_data_frame["Sequence"].tolist()
    labels = genes_data_frame["IsGene"].values

    logging.info("Extracting features...")
    X = extract_kmer_features(sequences, save_vectorizer=True)
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


def get_kmers(sequence: str, k: int) -> List[str]:
    """
    Generate k-mers from a given sequence.
    K-mer explanation: https://en.wikipedia.org/wiki/K-mer

    Args:
        sequence (str): The input sequence string.
        k (int): The length of each k-mer.

    Returns:
        List[str]: A list of k-mers extracted from the sequence.
    """
    return [sequence[i : i + k] for i in range(len(sequence) - k + 1)]


def extract_kmer_features(
    sequences: List[str], save_vectorizer: bool = True
):  # Todo add return type
    """
    Extract k-mer features from sequences and save the vectorizer if required.

    Args:
        sequences (List[str]): A list of sequences to extract k-mers from.
        save_vectorizer (bool): Whether to save the vectorizer to a file.
            Default is True.

    Returns:
        X: The k-mer feature matrix for the sequences.
    """
    logging.info("Generating k-mers...")
    kmers_list = [
        " ".join(get_kmers(seq, KMER_SIZE))
        for seq in tqdm(sequences, desc="Generating k-mers")
    ]

    logging.info("Converting k-mers to a numerical feature matrix...")
    # with open("data/models/count_vectorizer.pkl", "rb") as f:
    # vectorizer = dill.load(f)
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(kmers_list)

    if save_vectorizer:
        try:
            save_pkl(vectorizer, VECTORIZER_FILE_PATH)
            logging.info(f"Vectorizer saved to {VECTORIZER_FILE_PATH}")
        except Exception as e:
            logging.error(f"Failed to save vectorizer: {e}")

    return X
