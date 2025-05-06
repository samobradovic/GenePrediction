import logging
import os
from pathlib import Path

import dvc.api
from Bio import Entrez


def download_ids(max_results: int, search_query: str, file_path: Path) -> None:
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
        logging.info(f"Fetching {max_results} IDs with query: {search_query}")
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

        os.makedirs(file_path.parent, exist_ok=True)
        with open(file_path, "w") as f:
            for id in id_list:
                f.write(f"{id}\n")

    except Exception as e:
        logging.error(f"Error fetching IDs: {e}")
        raise


if __name__ == "__main__":
    params = dvc.api.params_show(stages="fetch_ids")
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    Entrez.email = params["entrez_email"]

    dataset_size = params["dataset_size"].lower()

    logging.info(f"Downloading data for dataset size: {dataset_size}.")
    download_ids(
        max_results=params["dataset_sizes"][f"{dataset_size}"]["genes"],
        search_query=params["entrez_search_queries"]["genes"],
        file_path=Path(params["paths"]["raw"]["gene_ids_file"]),
    )
    download_ids(
        max_results=params["dataset_sizes"][f"{dataset_size}"]["genomes"],
        search_query=params["entrez_search_queries"]["genomes"],
        file_path=Path(params["paths"]["raw"]["genome_ids_file"]),
    )
