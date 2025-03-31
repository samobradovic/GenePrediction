# import argparse
# import logging
# from typing import List, Tuple

# import numpy as np
# from scipy.sparse import csr_matrix
# from sklearn.feature_extraction.text import CountVectorizer
# from tqdm import tqdm

# from src.config import CSV_FILE_PATH, KMER_SIZE


# def get_kmers(sequence: str, k: int) -> List[str]:
#     """
#     Generate k-mers from a given sequence.
#     K-mer explanation: https://en.wikipedia.org/wiki/K-mer

#     Args:
#         sequence (str): The input sequence string.
#         k (int): The length of each k-mer.

#     Returns:
#         List[str]: A list of k-mers extracted from the sequence.
#     """
#     return [sequence[i : i + k] for i in range(len(sequence) - k + 1)]


# def fit_vectorizer(train_sequences: List[str]) -> Tuple[np.ndarray, CountVectorizer]:
#     logging.info("Generating k-mers for training data...")
#     train_kmers = [
#         " ".join(get_kmers(seq, KMER_SIZE))
#         for seq in tqdm(train_sequences, desc="Generating training k-mers")
#     ]
#     logging.info("Fitting vectorizer on kmer training data...")
#     vectorizer = CountVectorizer()
#     X_train = csr_matrix(vectorizer.fit_transform(train_kmers)).toarray()
#     return X_train, vectorizer


# def extract_kmer_features_for_sequence(
#     sequence: List[str], vectorizer: CountVectorizer, sequence_name: str = ""
# ) -> np.ndarray:
#     kmers = [
#         " ".join(get_kmers(seq, KMER_SIZE))
#         for seq in tqdm(
#             sequence, desc=f"Generating k-mers for sequence: {sequence_name}"
#         )
#     ]
#     logging.info(f"Transforming sequence: {sequence_name} using the vectorizer...")
#     return csr_matrix(vectorizer.transform(kmers)).toarray()


# if __name__ == "__main__":
#     logging.basicConfig(
#         level=logging.DEBUG,
#         format="%(asctime)s - %(levelname)s - %(message)s",
#     )

#     parser = argparse.ArgumentParser()
#     parser.add_argument(
#         "--csv_file_path",
#         type=str,
#         required=False,
#         default=f"{CSV_FILE_PATH}",
#         help="Path to the CSV file containing sequences",
#     )
#     args = parser.parse_args()

#     X_train, X_val, X_test, y_train, y_val, y_test = train_test_validation_split(
#         args.csv_file_path
#     )

#     save_train_test_validation_csvs(
#         {
#             "X_train": X_train,
#             "X_val": X_val,
#             "X_test": X_test,
#             "y_train": y_train,
#             "y_val": y_val,
#             "y_test": y_test,
#         }
#     )
