import logging
import random
from pathlib import Path
from typing import List

import dvc.api
import pandas as pd
from tqdm import tqdm

from src.helpers.csv_helper import (
    copy_csv_file,
    get_iterable_series_sequence_lengths_from_csv,
    write_string_sequences_to_csv,
)
from src.helpers.fasta_helper import get_string_sequences_from_directory


def get_random_gene_frame(genome: str, random_length: int) -> str:
    genome_length = len(genome)
    random_start_index = random.randint(0, genome_length - random_length)
    random_end_index = min(random_start_index + random_length, genome_length)
    return genome[random_start_index:random_end_index]


def generate_random_non_genes(
    random_length: int, number_of_genes: int, genomes: List[str]
) -> List[str]:
    logging.info("Generating non-genes...")

    non_genes = []

    for _ in tqdm(
        range(number_of_genes // 2), desc="Generating random non-genes", unit="gene"
    ):
        genome = random.choice(genomes)
        non_gene_candidate = get_random_gene_frame(genome, random_length)
        non_genes.append(non_gene_candidate)

    logging.info("Non-genes generated.")
    return non_genes


def generate_non_genes(
    genes_csv_file: Path, genomes_raw_data_folder: Path, combined_csv_file: Path
):
    gene_lengths: pd.Series[int] = get_iterable_series_sequence_lengths_from_csv(
        genes_csv_file,
    )
    random_length: int = gene_lengths.sample(1).iloc[0]
    number_of_genes: int = len(gene_lengths)

    genome_sequences: List[str] = get_string_sequences_from_directory(
        genomes_raw_data_folder,
    )
    random_non_genes: List[str] = generate_random_non_genes(
        random_length, number_of_genes, genome_sequences
    )

    copy_csv_file(
        source_csv_file=genes_csv_file,
        destination_csv_file=combined_csv_file,
    )
    write_string_sequences_to_csv(
        sequences=random_non_genes,
        csv_destination_path=combined_csv_file,
        write_header=False,
        is_real_gene=False,
    )


if __name__ == "__main__":
    params = dvc.api.params_show(stages="non_gene_generation")
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    generate_non_genes(
        genes_csv_file=Path(params["paths"]["genes_csv_file"]),
        genomes_raw_data_folder=Path(params["paths"]["genomes_raw_data_folder"]),
        combined_csv_file=Path(params["paths"]["csv_file"]),
    )
