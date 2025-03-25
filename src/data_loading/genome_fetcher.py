import logging
import os
import time
from typing import List

from Bio import Entrez, SeqIO
from Bio.Entrez.Parser import DictionaryElement
from Bio.SeqRecord import SeqRecord

from src.config import (
    ENTREZ_EMAIL,
    FILE_FORMAT,
    GENOMES_RAW_DATA_FILE_PATH,
    MAX_GENOMES,
    MAX_GENOMES_SIZE_GB,
    NUMBER_OF_GENOMES_BIG_DATASET,
    NUMBER_OF_GENOMES_SMALL_DATASET,
)
from src.helpers.filesystem_helper import get_folder_size

RETRY_COUNT = 15
RETRY_DELAY_SECONDS = 3

Entrez.email = ENTREZ_EMAIL


def get_genome_ids(max_results: int = MAX_GENOMES) -> List[str]:
    search_query = 'Bacteria[Organism] AND "complete genome"[Title]'
    handle = Entrez.esearch(
        db="nucleotide", term=search_query, retmax=max_results, sorted="relevance"
    )
    record = Entrez.read(handle)
    handle.close()
    if isinstance(record, DictionaryElement) and "IdList" in record:
        id_list = record["IdList"]
        logging.info(f"Found {len(id_list)} bacterial genomes")
        if id_list:
            return record["IdList"]
    logging.error("No bacterial genomes found")
    return []


def download_genome(genome_id: str) -> None:
    filename = os.path.join(
        GENOMES_RAW_DATA_FILE_PATH, f"genome_{genome_id}.{FILE_FORMAT}"
    )

    if os.path.exists(filename):
        logging.info(f"Genome {genome_id} already exists, skipping download.")
        return

    for attempt in range(RETRY_COUNT):
        try:
            logging.info(f"Downloading genome {genome_id}...")

            handle = Entrez.efetch(
                db="nucleotide", id=genome_id, rettype=FILE_FORMAT, retmode="text"
            )
            with open(filename, "w") as f:
                f.write(handle.read())
            logging.info(f"Genome saved to {filename}")
            handle.close()
            return
        except Exception as e:
            logging.error(
                f"Failed to download genome {genome_id}",
                f"on attempt {attempt + 1}: {e}",
            )
            if attempt < RETRY_COUNT - 1:
                time.sleep(RETRY_DELAY_SECONDS)
            else:
                logging.error(f"Exceeded maximum retries for genome {genome_id}")


def download_genomes(subset_size: int) -> None:
    logging.info("Downloading genomes from Entrez...")
    genome_ids = get_genome_ids(subset_size if subset_size > 0 else MAX_GENOMES)

    for genome_id in genome_ids:
        folder_size = get_folder_size(GENOMES_RAW_DATA_FILE_PATH)
        if folder_size > MAX_GENOMES_SIZE_GB:
            logging.warning(
                f"Folder size exceeds {MAX_GENOMES_SIZE_GB}GB. Stopping download."
            )
            break
        download_genome(genome_id)


def load_genomes(dataset_size: str, regenerate_data: bool = False) -> List[SeqRecord]:
    dataset_size = dataset_size.lower()
    if dataset_size == "small":
        subset_size = NUMBER_OF_GENOMES_SMALL_DATASET
    elif dataset_size == "big":
        subset_size = NUMBER_OF_GENOMES_BIG_DATASET
    elif dataset_size == "max":
        subset_size = -1
    else:
        raise ValueError(f"Invalid dataset size: {dataset_size}")

    if regenerate_data:
        download_genomes(subset_size)

    genomes = []
    logging.info(f"Loading genomes from {GENOMES_RAW_DATA_FILE_PATH}...")
    for i, file in enumerate(os.listdir(GENOMES_RAW_DATA_FILE_PATH)):
        if subset_size > 0 and i >= subset_size:
            logging.info(f"Loaded subset of: {subset_size} genes")
            break
        if file.endswith(FILE_FORMAT):
            try:
                filepath = os.path.join(GENOMES_RAW_DATA_FILE_PATH, file)
                logging.info(f"Loading genome from {filepath}.")
                genome = next(SeqIO.parse(filepath, FILE_FORMAT))
                genomes.append(genome)
            except Exception as e:
                logging.error(f"Failed to load genome from {file}: {e}")
    logging.info(f"Successfully loaded {len(genomes)} genomes.")
    return genomes


# if __name__ == "__main__":
#     download_genomes()
