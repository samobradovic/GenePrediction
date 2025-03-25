import logging
from typing import List, Tuple

import numpy as np
import pandas as pd
from pandas import DataFrame
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src.config import KMER_SIZE


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


def extract_kmer_features(
    train_sequences: List[str],
    val_sequences: List[str],
    test_sequences: List[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, CountVectorizer]:
    logging.info("Generating k-mers for training data...")
    train_kmers = [
        " ".join(get_kmers(seq, KMER_SIZE))
        for seq in tqdm(train_sequences, desc="Generating training k-mers")
    ]

    logging.info("Generating k-mers for validation data...")
    val_kmers = [
        " ".join(get_kmers(seq, KMER_SIZE))
        for seq in tqdm(val_sequences, desc="Generating validation k-mers")
    ]

    logging.info("Generating k-mers for test data...")
    test_kmers = [
        " ".join(get_kmers(seq, KMER_SIZE))
        for seq in tqdm(test_sequences, desc="Generating test k-mers")
    ]

    logging.info("Fitting vectorizer on training data...")
    vectorizer = CountVectorizer()
    X_train = csr_matrix(
        vectorizer.fit_transform(train_kmers)
    ).toarray()  # Not sure if this is correct
    X_val = csr_matrix(vectorizer.transform(val_kmers)).toarray()
    X_test = csr_matrix(vectorizer.transform(test_kmers)).toarray()

    return X_train, X_val, X_test, vectorizer
