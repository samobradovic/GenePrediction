import logging
import os
import time
from pathlib import Path
from typing import List

import dvc.api
from Bio import Entrez

from src.helpers.filesystem_helper import get_folder_size


def get_ids(max_results: int, search_query: str) -> List[str]:
    """Fetches a list of nucleotide IDs from Entrez based on a search query.

    Args:
        max_results (int): Maximum number of results to fetch.
        search_query (str): Query string to search in the nucleotide database.

    Returns:
        List[str]: List of nucleotide IDs.

    Raises:
        ValueError: If no IDs are found or the response format is invalid.
    """
    try:
        with Entrez.esearch(
            db="nucleotide", term=search_query, retmax=max_results, sort="relevance"
        ) as handle:
            record = Entrez.read(handle)

        if not isinstance(record, dict) or "IdList" not in record:
            logging.error(f"Unexpected response format: {record}")
            raise ValueError("Unexpected response format from Entrez API.")

        id_list = record["IdList"]

        if not isinstance(id_list, list) or not all(
            isinstance(i, str) for i in id_list
        ):
            logging.error(f"Invalid IdList format: {id_list}")
            raise ValueError("Invalid IdList format from Entrez API.")

        if not id_list:
            logging.warning("No IDs found in the response.")
            raise ValueError("No IDs found in the response.")

        logging.info(f"Found {len(id_list)} IDs.")
        return id_list

    except Exception as e:
        logging.error(f"Error fetching IDs: {e}")
        raise


def download_batch(
    file_path: str,
    batch_ids: List[str],
    batch_number: int,
) -> None:
    if os.path.exists(file_path):
        logging.info(f"Batch {batch_number} already exists, skipping download.")
        return

    for attempt in range(params["retry_entrez_request"]["count"]):
        try:
            logging.info(f"Downloading batch {batch_number} (Attempt {attempt + 1})...")
            handle = Entrez.efetch(
                db="nucleotide",
                id=",".join(batch_ids),
                rettype=params["file_format"],
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

            if attempt < params["retry_entrez_request"]["count"] - 1:
                time.sleep(params["retry_entrez_request"]["delay_seconds"])
            else:
                logging.error(
                    f"Failed to download batch {batch_number} after",
                    f"{params['retry_entrez_request']['count']} attempts.",
                )
                raise e


def download_data(
    number_of_datapoints: int,
    search_query: str,
    folder_path: Path,
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
        file_path = os.path.join(
            folder_path, f"batch_{batch_number}.{params['file_format']}"
        )
        download_batch(file_path, batch_ids, batch_number)


def download(dataset_size: str) -> None:
    dataset_size = dataset_size.lower()
    number_of_genes = params["dataset_sizes"][f"{dataset_size}"]["genes"]
    number_of_genomes = params["dataset_sizes"][f"{dataset_size}"]["genomes"]

    logging.info(f"Downloading data for dataset size: {dataset_size}.")
    download_data(
        number_of_genes,
        "Bacteria[Organism] AND gene[Title]",
        Path(params["paths"]["genes_raw_data_folder"]),
        params["max_sizes_gb"]["genes"],
        params["batch_sizes"]["genes"],
    )
    download_data(
        number_of_genomes,
        "Bacteria[Organism] AND complete genome[Title]",
        Path(params["paths"]["genomes_raw_data_folder"]),
        params["max_sizes_gb"]["genomes"],
        params["batch_sizes"]["genomes"],
    )


params = dvc.api.params_show(stages="download_genomes")
Entrez.email = params["entrez_email"]
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
if __name__ == "__main__":
    download(params["dataset_size"])
