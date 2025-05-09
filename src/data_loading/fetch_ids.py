import logging
import math
import os
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import dvc.api
from Bio import Entrez


def fetch_ids(max_results: int, search_query: str, offset: int) -> List[str]:
    """Fetches a list of nucleotide IDs from Entrez based on a search query.

    Args:
        max_results (int): Maximum number of results to fetch.
        search_query (str): Query string to search in the nucleotide database.
        offset (int): Offset for Entrez search (retstart).

    Returns:
        List[str]: List of nucleotide IDs.

    Raises:
        ValueError: If no IDs are found or the response format is invalid.
    """
    try:
        logging.info(
            f"Fetching {max_results} IDs with query: {search_query}, offset: {offset}"
        )
        with Entrez.esearch(
            db="nucleotide",
            term=search_query,
            retmax=max_results,
            retstart=offset,
            sort="relevance",
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
            raise ValueError("No IDs found in the response.")

        logging.info(f"Found {len(id_list)} IDs.")

        return id_list

    except Exception as e:
        logging.error(f"Error fetching IDs: {e}")
        raise


def get_tax_ids_for_ids(ids: List[str]) -> List[int]:
    """
    Fetches tax IDs for a list of nucleotide IDs using Entrez esummary.

    Args:
        ids (List[str]): List of nucleotide IDs.
    Returns:
        List[int]: List of tax IDs corresponding to the nucleotide IDs.
    Raises:
        ValueError: If no tax IDs are found or the response format is invalid.
    """
    try:
        logging.info(f"Fetching summaries for {len(ids)} IDs.")
        with Entrez.esummary(
            db="nucleotide", id=",".join(ids), retmode="xml"
        ) as handle:
            records = Entrez.read(handle)

        if not isinstance(records, list):
            logging.error(f"Unexpected records format: {records}")
            raise ValueError("Unexpected records format from Entrez API.")

        tax_ids = [
            int(record["TaxId"])
            for record in records
            if isinstance(record, dict) and "TaxId" in record
        ]

        if not tax_ids:
            raise ValueError("No tax IDs found in the response.")

        logging.info(f"Found {len(tax_ids)} tax IDs.")
        return tax_ids

    except Exception as e:
        logging.error(f"Error fetching summaries: {e}")
        raise


def normalized_shannon_diversity(list_values: List[int]) -> float:
    # TODO This can be moved into C lang. Untested.
    count = Counter(list_values)
    length_of_list = len(list_values)
    proportions = [freq / length_of_list for freq in count.values()]

    shannon = -sum(p * math.log(p) for p in proportions if p > 0)

    richness = len(count)
    if richness <= 1:
        return 0.0

    return shannon / math.log(richness)


def round_robin_by_tax_id(
    id_tax_pairs: List[Tuple[str, int]], subset_length: int
) -> List[Tuple[str, int]]:
    """
    Selects a subset of IDs with diverse taxonomies using a round-robin approach.
    The function ensures that the selected IDs are evenly distributed across different
    tax IDs, up to the specified subset length.
    Args:
        id_tax_pairs (List[Tuple[str, int]]): List of tuples containing IDs
            and their corresponding tax IDs.
        subset_length (int): Desired length of the subset.
    Returns:
        List[Tuple[str, int]]: A list of tuples containing selected IDs and
            their corresponding tax IDs.

    """

    tax_groups: Dict[int, List[str]] = {}
    for id, tax_id in id_tax_pairs:
        if tax_id not in tax_groups:
            tax_groups[tax_id] = []
        tax_groups[tax_id].append(id)

    selected_ids: List[str] = []
    tax_ids: List[int] = []

    while len(selected_ids) < subset_length and tax_groups:
        available_tax_ids = [tid for tid, ids in tax_groups.items() if ids]

        if not available_tax_ids:
            break

        for tax_id in available_tax_ids:
            if len(selected_ids) >= subset_length:
                break

            selected_id = tax_groups[tax_id].pop(0)
            selected_ids.append(selected_id)
            tax_ids.append(tax_id)

            if not tax_groups[tax_id]:
                del tax_groups[tax_id]

    return list(zip(selected_ids, tax_ids, strict=True))


def download_ids(
    max_results: int, search_query: str, file_path: Path, min_h_value: float
) -> None:
    """
    Downloads IDs and their taxonomies in batches, until we find a subset with
    diversity greater than or equal to min_h_value.
    Uses fetch_ids_with_taxonomies to get both IDs and taxonomy IDs in one call.

    Args:
        max_results (int): Maximum number of IDs to fetch in each batch.
        search_query (str): Search query for Entrez.
        file_path (Path): Path to save the IDs.
        min_h_value (float): Minimum diversity score to achieve.
    """
    diversity_score: float = 0.0
    id_tax_pairs: List[Tuple[str, int]] = []
    most_diverse_subset: List[Tuple[str, int]] = []
    offset = 0

    while diversity_score <= min_h_value:
        new_ids = fetch_ids(max_results, search_query, offset)
        new_tax_ids = get_tax_ids_for_ids(new_ids)
        id_tax_pairs.extend(zip(new_ids, new_tax_ids, strict=True))
        most_diverse_subset = round_robin_by_tax_id(id_tax_pairs, max_results)
        diversity_score = normalized_shannon_diversity(
            [tax_id for _, tax_id in most_diverse_subset]
        )

        logging.info(f"Best diversity score of current batch: {diversity_score}")
        offset += max_results

    logging.info(
        f"Found {len(most_diverse_subset)} IDs with diversity score {diversity_score}."
    )
    logging.info(most_diverse_subset[0:10])

    os.makedirs(file_path.parent, exist_ok=True)
    with open(file_path, "w") as f:
        for id, _ in most_diverse_subset:
            f.write(f"{id}\n")


if __name__ == "__main__":
    params = dvc.api.params_show(stages="fetch_ids")
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    Entrez.email = params["entrez_email"]

    dataset_size = params["dataset_size"].lower()

    logging.info(f"Downloading data for dataset size: {dataset_size}.")
    min_h_genomes = (
        params["taxonomic_diversity"]["genomes"]["normalized_h_min"]
        if params["taxonomic_diversity"]["genomes"]["enforced"]
        else 0.0
    )
    min_h_genes = (
        params["taxonomic_diversity"]["genes"]["normalized_h_min"]
        if params["taxonomic_diversity"]["genes"]["enforced"]
        else 0.0
    )
    download_ids(
        max_results=params["dataset_sizes"][f"{dataset_size}"]["genes"],
        search_query=params["entrez_search_queries"]["genes"],
        file_path=Path(params["paths"]["raw"]["gene_ids_file"]),
        min_h_value=min_h_genes,
    )
    download_ids(
        max_results=params["dataset_sizes"][f"{dataset_size}"]["genomes"],
        search_query=params["entrez_search_queries"]["genomes"],
        file_path=Path(params["paths"]["raw"]["genome_ids_file"]),
        min_h_value=min_h_genomes,
    )
