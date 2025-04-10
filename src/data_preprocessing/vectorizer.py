import logging
from pathlib import Path
from typing import List, Tuple

import dvc.api
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm

from src.helpers.filesystem_helper import save_pkl


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
    if k <= 0 or k > len(sequence):
        raise ValueError(
            "k must be a positive integer less than or equal to the sequence length."
        )
    if len(sequence) == 0:
        raise ValueError("The input sequence is empty.")
    return [sequence[i : i + k] for i in range(len(sequence) - k + 1)]


def vectorize_sequences(
    vectorizer: CountVectorizer,
    X_train: List[str],
    X_test: List[str],
    X_val: List[str],
    kmer_size: int,
) -> Tuple[csr_matrix, csr_matrix, csr_matrix]:
    """
    Vectorize k-mer sequences using a CountVectorizer.

    Args:
        vectorizer (CountVectorizer): The CountVectorizer instance.
        train_kmers (List[str]): Training k-mers.
        test_kmers (List[str]): Testing k-mers.
        val_kmers (List[str]): Validation k-mers.

    Returns:
        Tuple[csr_matrix, csr_matrix, csr_matrix]: Vectorized training, testing,
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

    X_train_vectorized = csr_matrix(vectorizer.fit_transform(train_kmers))
    X_test_vectorized = csr_matrix(vectorizer.transform(test_kmers))
    X_val_vectorized = csr_matrix(vectorizer.transform(val_kmers))
    return X_train_vectorized, X_test_vectorized, X_val_vectorized


def validate_and_extract_sequences(csv_path: Path, file_type: str) -> List[str]:
    """
    Validate the CSV file and extract the 'Sequence' column as a list.

    Args:
        csv_path (Path): Path to the CSV file.
        file_type (str): Type of the file (e.g., 'Training', 'Testing', 'Validation').

    Returns:
        List[str]: List of sequences from the 'Sequence' column.

    Raises:
        ValueError: If the file is empty or does not contain the 'Sequence' column.
    """
    df = pd.read_csv(csv_path)
    if df.empty:
        logging.error(f"{file_type} CSV is empty.")
        raise ValueError(f"{file_type} CSV is empty.")
    if "Sequence" not in df.columns:
        logging.error(f"{file_type} CSV does not contain 'Sequence' column.")
        raise ValueError(f"{file_type} CSV does not contain 'Sequence' column.")
    return df["Sequence"].tolist()


def vectorize(
    training_csv_path: Path,
    testing_csv_path: Path,
    validation_csv_path: Path,
    training_vectorized_path: Path,
    testing_vectorized_path: Path,
    validation_vectorized_path: Path,
    vectorizer_path: Path,
    kmer_size: int,
) -> None:
    """
    Main function to vectorize sequences and save the results.

    Args:
        training_csv_path (Path): Path to training CSV file.
        testing_csv_path (Path): Path to testing CSV file.
        validation_csv_path (Path): Path to validation CSV file.
        training_vectorized_path (Path): Path to save training data.
        testing_vectorized_path (Path): Path to save testing data.
        validation_vectorized_path (Path): Path to save validation data.
        vectorizer_path (Path): Path to save the vectorizer.
        kmer_size (int): Size of k-mers.
    """

    X_train = validate_and_extract_sequences(training_csv_path, "Training")
    X_test = validate_and_extract_sequences(testing_csv_path, "Testing")
    X_val = validate_and_extract_sequences(validation_csv_path, "Validation")

    vectorizer = CountVectorizer()
    X_train_vectorized, X_test_vectorized, X_val_vectorized = vectorize_sequences(
        vectorizer, X_train, X_test, X_val, kmer_size
    )

    save_pkl(X_train_vectorized, training_vectorized_path)
    save_pkl(X_test_vectorized, testing_vectorized_path)
    save_pkl(X_val_vectorized, validation_vectorized_path)
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
        training_vectorized_path=Path(params["paths"]["training_vectorized_file"]),
        testing_vectorized_path=Path(params["paths"]["testing_vectorized_file"]),
        validation_vectorized_path=Path(params["paths"]["validation_vectorized_file"]),
        vectorizer_path=Path(params["paths"]["vectorizer_file"]),
        kmer_size=params["kmer_size"],
    )
