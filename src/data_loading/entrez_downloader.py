import argparse
import logging
import os
import time
from typing import List

from Bio import Entrez
from Bio.Entrez.Parser import DictionaryElement

from src.config import (
    ENTREZ_EMAIL,
    FILE_FORMAT,
    GENES_RAW_DATA_FOLDER_PATH,
    GENOMES_RAW_DATA_FOLDER_PATH,
    MAX_GENES_SIZE_GB,
    MAX_GENOMES_SIZE_GB,
    NUMBER_OF_GENES_BIG_DATASET,
    NUMBER_OF_GENES_SMALL_DATASET,
    NUMBER_OF_GENOMES_BIG_DATASET,
    NUMBER_OF_GENOMES_SMALL_DATASET,
)
from src.helpers.filesystem_helper import get_folder_size

RETRY_COUNT = 15
RETRY_DELAY_SECONDS = 3
BATCH_SIZE_GENES = 5000
BATCH_SIZE_GENOMES = 5

Entrez.email = ENTREZ_EMAIL


def get_ids(max_results: int, search_query: str) -> List[str]:
    handle = Entrez.esearch(
        db="nucleotide", term=search_query, retmax=max_results, sorted="relevance"
    )
    record = Entrez.read(handle)
    handle.close()
    if isinstance(record, DictionaryElement) and "IdList" in record:
        id_list = record["IdList"]
        logging.info(f"Found {len(id_list)} ids")
        if id_list:
            return record["IdList"]
    logging.error(f"No ids found for the given search query: {search_query}")
    return []


def download_batch(
    file_path: str,
    batch_ids: List[str],
    batch_number: int,
) -> None:
    if os.path.exists(file_path):
        logging.info(f"Batch {batch_number} already exists, skipping download.")
        return

    for attempt in range(RETRY_COUNT):
        try:
            logging.info(f"Downloading batch {batch_number} (Attempt {attempt + 1})...")
            handle = Entrez.efetch(
                db="nucleotide",
                id=",".join(batch_ids),
                rettype=FILE_FORMAT,
                retmode="text",
            )

            with open(file_path, "w") as f:
                f.write(handle.read())
            logging.info(f"Batch {batch_number} downloaded successfully.")
            handle.close()
            return
        except Exception as e:
            logging.error(
                f"Failed to download batch {batch_number} on attempt {attempt + 1}: {e}"
            )

            if attempt < RETRY_COUNT - 1:
                time.sleep(RETRY_DELAY_SECONDS)
            else:
                logging.error(
                    f"Failed to download batch {batch_number} after {RETRY_COUNT} ",
                    "attempts.",
                )
                raise e


def download_data(
    number_of_datapoints: int,
    search_query: str,
    folder_path: str,
    max_size_gb: int,
    batch_size: int,
) -> None:
    logging.info("Downloading data from Entrez...")
    ids = get_ids(
        number_of_datapoints,
        search_query,
    )

    for i in range(0, len(ids), batch_size):
        os.makedirs(folder_path, exist_ok=True)
        folder_size = get_folder_size(folder_path)
        if folder_size > max_size_gb:
            logging.warning(f"Folder size exceeds {max_size_gb}GB. Stopping download.")
            break

        batch_ids = ids[i : i + batch_size]
        batch_number = i // batch_size + 1
        file_path = os.path.join(folder_path, f"batch_{batch_number}.{FILE_FORMAT}")
        download_batch(file_path, batch_ids, batch_number)


def download(dataset_size: str = "small") -> None:
    dataset_size = dataset_size.lower()
    if dataset_size == "small":
        number_of_genes = NUMBER_OF_GENES_SMALL_DATASET
        number_of_genomes = NUMBER_OF_GENOMES_SMALL_DATASET
    elif dataset_size == "big":
        number_of_genes = NUMBER_OF_GENES_BIG_DATASET
        number_of_genomes = NUMBER_OF_GENOMES_BIG_DATASET
    elif dataset_size == "max":
        number_of_genes = -1
        number_of_genomes = -1
    else:
        raise ValueError(f"Invalid dataset size: {dataset_size}")

    logging.info(f"Downloading data for dataset size: {dataset_size}.")
    download_data(
        number_of_genes,
        "Bacteria[Organism] AND gene[Title]",
        GENES_RAW_DATA_FOLDER_PATH,
        MAX_GENES_SIZE_GB,
        BATCH_SIZE_GENES,
    )
    download_data(
        number_of_genomes,
        "Bacteria[Organism] AND complete genome[Title]",
        GENOMES_RAW_DATA_FOLDER_PATH,
        MAX_GENOMES_SIZE_GB,
        BATCH_SIZE_GENOMES,
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_size",
        type=str,
        required=True,
        help="Dataset size: small, big, or max",
    )
    args = parser.parse_args()

    download(args.dataset_size)
