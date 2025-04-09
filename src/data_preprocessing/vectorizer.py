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


def vectorize_sequences(
    vectorizer: CountVectorizer,
    X_train: List[str],
    X_test: List[str],
    X_val: List[str],
    kmer_size: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Vectorize k-mer sequences using a CountVectorizer.

    Args:
        vectorizer (CountVectorizer): The CountVectorizer instance.
        train_kmers (List[str]): Training k-mers.
        test_kmers (List[str]): Testing k-mers.
        val_kmers (List[str]): Validation k-mers.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: Vectorized training, testing,
          and validation data.
    """
    train_kmers = [
        " ".join(get_kmers(seq, kmer_size))
        for seq in tqdm(X_train, desc="Generating training k-mers")
    ]
    test_kmers = [
        " ".join(get_kmers(seq, kmer_size))
        for seq in tqdm(X_test, desc="Generating testing k-mers")
    ]
    val_kmers = [
        " ".join(get_kmers(seq, kmer_size))
        for seq in tqdm(X_val, desc="Generating validation k-mers")
    ]

    X_train_vectorized = csr_matrix(vectorizer.fit_transform(train_kmers)).toarray()
    X_test_vectorized = csr_matrix(vectorizer.transform(test_kmers)).toarray()
    X_val_vectorized = csr_matrix(vectorizer.transform(val_kmers)).toarray()
    return X_train_vectorized, X_test_vectorized, X_val_vectorized


def vectorize(
    training_csv_path: Path,
    testing_csv_path: Path,
    validation_csv_path: Path,
    training_npy_path: Path,
    testing_npy_path: Path,
    validation_npy_path: Path,
    vectorizer_path: Path,
    kmer_size: int,
) -> None:
    """
    Main function to vectorize sequences and save the results.

    Args:
        training_csv_path (Path): Path to training CSV file.
        testing_csv_path (Path): Path to testing CSV file.
        validation_csv_path (Path): Path to validation CSV file.
        training_npy_path (Path): Path to save training data.
        testing_npy_path (Path): Path to save testing data.
        validation_npy_path (Path): Path to save validation data.
        vectorizer_path (Path): Path to save the vectorizer.
        kmer_size (int): Size of k-mers.
    """
    X_train = pd.read_csv(training_csv_path)["Sequence"].tolist()
    X_test = pd.read_csv(testing_csv_path)["Sequence"].tolist()
    X_val = pd.read_csv(validation_csv_path)["Sequence"].tolist()

    vectorizer = CountVectorizer()
    X_train_vectorized, X_test_vectorized, X_val_vectorized = vectorize_sequences(
        vectorizer, X_train, X_test, X_val, kmer_size
    )

    logging.info("Saving vectorized data to .npy files...")
    np.save(training_npy_path, X_train_vectorized)
    np.save(testing_npy_path, X_test_vectorized)
    np.save(validation_npy_path, X_val_vectorized)
    logging.info("Vectorized data saved successfully.")

    save_pkl(vectorizer, vectorizer_path)


if __name__ == "__main__":
    params = dvc.api.params_show(stages="vectorization")
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    vectorize(
        training_csv_path=Path(params["paths"]["training_csv_file"]),
        testing_csv_path=Path(params["paths"]["testing_csv_file"]),
        validation_csv_path=Path(params["paths"]["validation_csv_file"]),
        training_npy_path=Path(params["paths"]["training_npy_file"]),
        testing_npy_path=Path(params["paths"]["testing_npy_file"]),
        validation_npy_path=Path(params["paths"]["validation_npy_file"]),
        vectorizer_path=Path(params["paths"]["vectorizer_file"]),
        kmer_size=params["kmer_size"],
    )
