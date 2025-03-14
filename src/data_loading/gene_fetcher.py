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
    GENES_RAW_DATA_FILE_PATH,
    MAX_GENES,
    MAX_GENES_SIZE_GB,
)
from src.helpers.filesystem_helper import get_folder_size

RETRY_COUNT = 15
RETRY_DELAY_SECONDS = 3
BATCH_SIZE = 5000

Entrez.email = ENTREZ_EMAIL


def get_gene_ids(max_results: int = MAX_GENES) -> List[str]:
    search_query = "bacteria[Organism] AND gene[Title]"
    handle = Entrez.esearch(
        db="nucleotide", term=search_query, retmax=max_results, sorted="relevance"
    )
    record = Entrez.read(handle)
    handle.close()
    if isinstance(record, DictionaryElement) and "IdList" in record:
        id_list = record["IdList"]
        logging.info(f"Found {len(id_list)} bacterial genes")
        if id_list:
            return record["IdList"]
    logging.error("No bacterial genes found")
    return []


def download_genes() -> None:
    logging.info("Downloading genes from Entrez...")
    gene_ids = get_gene_ids()
    for i in range(0, len(gene_ids), BATCH_SIZE):
        folder_size = get_folder_size(GENES_RAW_DATA_FILE_PATH)
        if folder_size > MAX_GENES_SIZE_GB:
            logging.warning(
                f"Folder size exceeds {MAX_GENES_SIZE_GB}GB. Stopping download."
            )
            break

        batch_ids = gene_ids[i : i + BATCH_SIZE]
        filename = f"{GENES_RAW_DATA_FILE_PATH}/genes_batch_{i // BATCH_SIZE + 1}.{FILE_FORMAT}"
        if os.path.exists(filename):
            logging.info(
                f"Batch {i // BATCH_SIZE + 1} already exists, skipping download."
            )
            continue

        for attempt in range(RETRY_COUNT):
            try:
                logging.info(f"Downloading genes batch {i // BATCH_SIZE + 1}...")
                handle = Entrez.efetch(
                    db="nucleotide",
                    id=",".join(batch_ids),
                    rettype=FILE_FORMAT,
                    retmode="text",
                )
                with open(filename, "w") as f:
                    f.write(handle.read())
                logging.info(f"Genes batch saved to {filename}")
                handle.close()
                break
            except Exception as e:
                logging.error(
                    f"Failed to download genes batch {i // BATCH_SIZE + 1} on attempt {attempt + 1}: {e}"
                )
                if attempt < RETRY_COUNT - 1:
                    time.sleep(RETRY_DELAY_SECONDS)
                else:
                    logging.error(
                        f"Exceeded maximum retries for genes batch {i // BATCH_SIZE + 1}"
                    )


def load_genes(load_from_files: bool = False, subset_size: int = -1) -> List[SeqRecord]:
    if not load_from_files:
        download_genes()

    genes = []
    for i, file in enumerate(os.listdir(GENES_RAW_DATA_FILE_PATH)):
        if subset_size > 0 and i >= subset_size:
            logging.info(f"Loaded subset of: {subset_size} genes")
            break
        if file.endswith(FILE_FORMAT):
            logging.info(f"Loading genes from {file}")
            genes.extend(SeqIO.parse(f"{GENES_RAW_DATA_FILE_PATH}/{file}", FILE_FORMAT))

    return genes


# if __name__ == "__main__":
# download_genes()
