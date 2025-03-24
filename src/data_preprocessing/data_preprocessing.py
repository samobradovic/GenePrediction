import logging
from typing import List, Tuple

import numpy as np
import pandas as pd
from pandas import DataFrame
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src.config import KMER_SIZE, VECTORIZER_FILE_PATH
from src.helpers.filesystem_helper import save_pkl


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

    logging.info("Extracting features...")
    X_train_vectorized, X_val_vectorized, X_test_vectorized = extract_kmer_features(
        X_train, X_val, X_test
    )
    logging.info("Features extracted.")

    return (
        X_train_vectorized,
        X_val_vectorized,
        X_test_vectorized,
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
    save_vectorizer: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract k-mer features from sequences and save the vectorizer if required.

    Args:
        train_sequences (List[str]): A list of training sequences to extract k-mers
          from.
        val_sequences (List[str]): A list of validation sequences to extract k-mers
        from.
        test_sequences (List[str]): A list of test sequences to extract k-mers from.
        save_vectorizer (bool): Whether to save the vectorizer to a file. Default is
          True.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: The k-mer feature matrices for the
        training, validation, and test sequences as dense arrays.
    """
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
    X_train = csr_matrix(vectorizer.fit_transform(train_kmers)).toarray()
    X_val = csr_matrix(vectorizer.transform(val_kmers)).toarray()
    X_test = csr_matrix(vectorizer.transform(test_kmers)).toarray()

    if save_vectorizer:
        try:
            save_pkl(vectorizer, VECTORIZER_FILE_PATH)
            logging.info(f"Vectorizer saved to {VECTORIZER_FILE_PATH}")
        except Exception as e:
            logging.error(f"Failed to save vectorizer: {e}")

    return X_train, X_val, X_test
