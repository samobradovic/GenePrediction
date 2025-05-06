import logging
import os
import time
from pathlib import Path
from typing import List

import dvc.api
from Bio import Entrez

from src.helpers.filesystem_helper import get_folder_size


def get_ids_from_file(file_path: Path) -> List[str]:
    try:
        with open(
            file_path,
            "r",
        ) as f:
            return f.read().splitlines()
    except FileNotFoundError:
        logging.error(f"File not found: {file_path}.")
        raise
    except Exception as e:
        logging.error(f"Error reading file {file_path}: {e}.")
        raise


def download_batch(
    file_path: str,
    batch_ids: List[str],
    batch_number: int,
    retry_count: int,
    delay_seconds: int,
    file_format: str,
) -> None:
    """Downloads a batch of nucleotide data from Entrez and saves it to a file.

    The function attempts to download data for a given batch of IDs from Entrez.
    It retries the download if it fails, up to a specified number of attempts.
    If the batch already exists at the given file path, the download is skipped.

    Args:
        file_path (str): The path to save the downloaded data.
        batch_ids (List[str]): A list of nucleotide IDs to fetch from Entrez.
        batch_number (int): The batch number, used for logging.
        retry_count (int): The number of times to retry the download if it fails.
        delay_seconds (int): The number of seconds to wait before retrying.
        file_format (str): The format of the file to be downloaded (e.g., "fasta").

    Raises:
        Exception: If the download fails after the specified number of retries.
    """
    if not batch_ids:
        logging.warning(f"Batch {batch_number} is empty, skipping download.")
        return
    if os.path.exists(file_path):
        logging.info(
            f"Batch {batch_number} already exists at {file_path}, skipping download."
        )
        return

    for attempt in range(retry_count):
        try:
            logging.info(f"Downloading batch {batch_number} (Attempt {attempt + 1})...")

            handle = Entrez.efetch(
                db="nucleotide",
                id=",".join(batch_ids),
                rettype=file_format,
                retmode="text",
            )

            with open(file_path, "w") as f:
                f.write(handle.read())
            logging.info(
                f"Batch {batch_number} downloaded successfully and saved "
                f"to {file_path}.",
            )
            return
        except Exception as e:
            logging.error(
                f"Failed to download batch {batch_number} (Attempt {attempt + 1}): {e}"
            )

            if attempt < retry_count - 1:
                logging.info(
                    f"Retrying in {delay_seconds} seconds...",
                )
                time.sleep(delay_seconds)
            else:
                logging.error(
                    f"Failed to download batch {batch_number} after {retry_count} "
                    "attempts.",
                )
                raise e
        finally:
            if "handle" in locals() and handle:
                handle.close()


def download_data(
    ids: List[str],
    folder_path: Path,
    max_size_gb: int,
    batch_size: int,
    file_format: str,
    batch_retry_count: int,
    batch_delay_seconds: int,
) -> None:
    """Download data from Entrez and save it in batches.

    Args:
        number_of_datapoints (int): Number of data points to download.
        search_query (str): Search query to get IDs from Entrez.
        folder_path (Path): Folder where the data will be saved.
        max_size_gb (int): Maximum size (in GB) the folder can reach before stopping.
        batch_size (int): Size of each batch for downloading.
        file_format (str): Format of the file to be downloaded (e.g., "fasta").
        batch_retry_count (int): Number of retries for each batch download.
        batch_delay_seconds (int): Delay in seconds between retries.
    """
    logging.info("Starting data download from Entrez...")
    os.makedirs(folder_path, exist_ok=True)

    for i in range(0, len(ids), batch_size):
        folder_size = get_folder_size(folder_path)
        if folder_size > max_size_gb:
            logging.warning(f"Folder size exceeds {max_size_gb}GB. Stopping download.")
            break

        batch_ids = ids[i : i + batch_size]
        batch_number = i // batch_size + 1
        file_path = os.path.join(folder_path, f"batch_{batch_number}.{file_format}")
        download_batch(
            file_path,
            batch_ids,
            batch_number,
            batch_retry_count,
            batch_delay_seconds,
            file_format,
        )

    logging.info(f"Download completed. {len(ids)} IDs processed.")


if __name__ == "__main__":
    params = dvc.api.params_show(stages="download_genomes")
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    Entrez.email = params["entrez_email"]

    dataset_size = params["dataset_size"].lower()

    logging.info(f"Downloading data for dataset size: {dataset_size}.")

    gene_ids = get_ids_from_file(Path(params["paths"]["entrez_gene_ids_raw_data"]))
    genome_ids = get_ids_from_file(Path(params["paths"]["entrez_genome_ids_raw_data"]))

    download_data(
        ids=gene_ids,
        folder_path=Path(params["paths"]["genes_raw_data_folder"]),
        max_size_gb=params["max_sizes_gb"]["genes"],
        batch_size=params["batch_sizes"]["genes"],
        batch_retry_count=params["retry_entrez_request"]["count"],
        batch_delay_seconds=params["retry_entrez_request"]["delay_seconds"],
        file_format=params["file_format"],
    )
    download_data(
        ids=genome_ids,
        folder_path=Path(params["paths"]["genomes_raw_data_folder"]),
        max_size_gb=params["max_sizes_gb"]["genomes"],
        batch_size=params["batch_sizes"]["genomes"],
        batch_retry_count=params["retry_entrez_request"]["count"],
        batch_delay_seconds=params["retry_entrez_request"]["delay_seconds"],
        file_format=params["file_format"],
    )
