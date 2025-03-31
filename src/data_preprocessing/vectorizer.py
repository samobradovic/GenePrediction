import logging
from pathlib import Path
from typing import List, Tuple

import dvc.api
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm

from helpers.filesystem_helper import save_pkl


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
        " ".join(get_kmers(seq, params["kmer_size"]))
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
        " ".join(get_kmers(seq, params["kmer_size"]))
        for seq in tqdm(
            sequence, desc=f"Generating k-mers for sequence: {sequence_name}"
        )
    ]
    logging.info(f"Transforming sequence: {sequence_name} using the vectorizer...")
    return csr_matrix(vectorizer.transform(kmers)).toarray()


params = dvc.api.params_show(stages="vectorizer")
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
if __name__ == "__main__":
    X_train = pd.read_csv(Path(params["paths"]["training_csv_file"]))[
        "Sequence"
    ].tolist()
    X_test = pd.read_csv(Path(params["paths"]["testing_csv_file"]))["Sequence"].tolist()
    X_val = pd.read_csv(Path(params["paths"]["validation_csv_file"]))[
        "Sequence"
    ].tolist()

    X_train_vectorized, vectorizer = fit_vectorizer(X_train)
    X_val_vectorized = extract_kmer_features_for_sequence(
        X_val, vectorizer, "Validation"
    )
    X_test_vectorized = extract_kmer_features_for_sequence(
        X_test, vectorizer, "Testing"
    )

    logging.info("Saving vectorized data to .npy files...")
    np.save(Path(params["paths"]["training_npy_file"]), X_train_vectorized)
    np.save(Path(params["paths"]["validation_npy_file"]), X_val_vectorized)
    np.save(Path(params["paths"]["testing_npy_file"]), X_test_vectorized)
    logging.info("Vectorized data saved successfully.")

    logging.info("Saving vectorizer to .pkl file...")
    save_pkl(vectorizer, Path(params["paths"]["vectorizer_file"]))
    logging.info("Vectorizer saved successfully.")
