import logging
import os
from typing import List, Tuple

import numpy as np
import pandas as pd
from pandas import DataFrame
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src.config import CSV_FILE_PATH, KMER_SIZE
from src.helpers.filesystem_helper import get_versioned_file_path


def save_train_test_validation_csvs(
    data_splits: dict,
) -> Tuple[str, str, str]:
    logging.info("Saving training, validation, and testing CSVs...")

    csv_file_path = get_versioned_file_path(CSV_FILE_PATH)

    train_csv_path = os.path.join(csv_file_path, "train.csv")
    val_csv_path = os.path.join(csv_file_path, "validation.csv")
    test_csv_path = os.path.join(csv_file_path, "test.csv")

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


def fit_vectorizer(train_sequences: List[str]) -> Tuple[np.ndarray, CountVectorizer]:
    logging.info("Generating k-mers for training data...")
    train_kmers = [
        " ".join(get_kmers(seq, KMER_SIZE))
        for seq in tqdm(train_sequences, desc="Generating training k-mers")
    ]
    logging.info("Fitting vectorizer on kmer training data...")
    vectorizer = CountVectorizer()
    X_train = csr_matrix(vectorizer.fit_transform(train_kmers)).toarray()
    return X_train, vectorizer


def extract_kmer_features_for_sequence(
    sequence: List[str], vectorizer: CountVectorizer, sequence_name: str = ""
) -> np.ndarray:
    kmers = [
        " ".join(get_kmers(seq, KMER_SIZE))
        for seq in tqdm(
            sequence, desc=f"Generating k-mers for sequence: {sequence_name}"
        )
    ]
    logging.info(f"Transforming sequence: {sequence_name} using the vectorizer...")
    return csr_matrix(vectorizer.transform(kmers)).toarray()
