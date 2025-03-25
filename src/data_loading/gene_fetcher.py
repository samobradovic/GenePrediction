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
    NUMBER_OF_GENES_BIG_DATASET,
    NUMBER_OF_GENES_SMALL_DATASET,
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


def download_gene_batch(batch_ids: List[str], batch_number: int) -> None:
    """Download a batch of genes by IDs."""
    filename = os.path.join(
        GENES_RAW_DATA_FILE_PATH, f"genes_batch_{batch_number}.{FILE_FORMAT}"
    )

    if os.path.exists(filename):
        logging.info(f"Batch {batch_number} already exists, skipping download.")
        return

    for attempt in range(RETRY_COUNT):
        try:
            logging.info(
                f"Downloading genes batch {batch_number} (Attempt {attempt + 1})..."
            )
            handle = Entrez.efetch(
                db="nucleotide",
                id=",".join(batch_ids),
                rettype=FILE_FORMAT,
                retmode="text",
            )
            with open(filename, "w") as f:
                f.write(handle.read())
            logging.info(f"Genes batch {batch_number} saved to {filename}.")
            handle.close()
            return
        except Exception as e:
            logging.error(
                f"Failed to download genes batch {batch_number} on attempt",
                f"{attempt + 1}: {e}",
            )
            if attempt < RETRY_COUNT - 1:
                time.sleep(RETRY_DELAY_SECONDS)
            else:
                logging.error(
                    f"Exceeded maximum retries for genes batch {batch_number}."
                )


def download_genes(subset_size: int) -> None:
    logging.info("Downloading genes from Entrez...")
    gene_ids = get_gene_ids(subset_size if subset_size > 0 else MAX_GENES)

    for i in range(0, len(gene_ids), BATCH_SIZE):
        folder_size = get_folder_size(GENES_RAW_DATA_FILE_PATH)
        if folder_size > MAX_GENES_SIZE_GB:
            logging.warning(
                f"Folder size exceeds {MAX_GENES_SIZE_GB}GB. Stopping download."
            )
            break

        batch_ids = gene_ids[i : i + BATCH_SIZE]
        batch_number = i // BATCH_SIZE + 1
        download_gene_batch(batch_ids, batch_number)


def load_genes(dataset_size: str, regenerate_data: bool = False) -> List[SeqRecord]:
    dataset_size = dataset_size.lower()
    if dataset_size == "small":
        subset_size = NUMBER_OF_GENES_SMALL_DATASET
    elif dataset_size == "big":
        subset_size = NUMBER_OF_GENES_BIG_DATASET
    elif dataset_size == "max":
        subset_size = -1
    else:
        raise ValueError(f"Invalid dataset size: {dataset_size}")

    if regenerate_data:
        logging.info(f"Regenerating data for dataset size: {dataset_size}.")
        download_genes(subset_size)

    genes = []
    total_genes_loaded = 0
    logging.info(f"Loading genes from {GENES_RAW_DATA_FILE_PATH}...")
    for file in os.listdir(GENES_RAW_DATA_FILE_PATH):
        if file.endswith(FILE_FORMAT):
            try:
                filepath = os.path.join(GENES_RAW_DATA_FILE_PATH, file)
                logging.info(f"Loading genes from {filepath}.")
                for gene in SeqIO.parse(filepath, FILE_FORMAT):
                    genes.append(gene)
                    total_genes_loaded += 1
                    if subset_size > 0 and total_genes_loaded >= subset_size:
                        logging.info(f"Loaded subset of {subset_size} genes.")
                        return genes
            except Exception as e:
                logging.error(f"Failed to load genes from {file}: {e}")
    logging.info(f"Successfully loaded {len(genes)} genes.")
    return genes


# if __name__ == "__main__":
# download_genes()
