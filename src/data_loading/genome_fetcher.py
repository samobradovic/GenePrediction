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


def download_genomes() -> None:
    logging.info("Downloading genes from Entrez...")
    genome_ids = get_genome_ids()
    for genome_id in genome_ids:
        folder_size = get_folder_size(GENOMES_RAW_DATA_FILE_PATH)
        if folder_size > MAX_GENOMES_SIZE_GB:
            logging.warning(
                f"Folder size exceeds {MAX_GENOMES_SIZE_GB}GB. Stopping download."
            )
            break

        filename = f"{GENOMES_RAW_DATA_FILE_PATH}/genome_{genome_id}.{FILE_FORMAT}"
        if os.path.exists(filename):
            logging.info(f"Genome {genome_id} already exists, skipping download.")
            continue

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
                break
            except Exception as e:
                logging.error(
                    f"Failed to download genome {genome_id} on attempt {attempt + 1}: {e}"
                )
                if attempt < RETRY_COUNT - 1:
                    time.sleep(RETRY_DELAY_SECONDS)
                else:
                    logging.error(f"Exceeded maximum retries for genome {genome_id}")


def load_genomes(
    load_from_files: bool = False, subset_size: int = -1
) -> List[SeqRecord]:

    if not load_from_files:
        download_genomes()

    genomes = []
    for i, file in enumerate(os.listdir(GENOMES_RAW_DATA_FILE_PATH)):
        if subset_size > 0 and i >= subset_size:
            logging.info(f"Loaded subset of: {subset_size} genes")
            break
        if file.endswith(FILE_FORMAT):
            logging.info(f"Loading genome from {file}")
            genomes.append(
                next(SeqIO.parse(f"{GENOMES_RAW_DATA_FILE_PATH}/{file}", FILE_FORMAT))
            )
    return genomes


# if __name__ == "__main__":
#     download_genomes()
