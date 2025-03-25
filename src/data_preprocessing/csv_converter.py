import logging
import os
import random
import time
from typing import List

from Bio.SeqRecord import SeqRecord
from tqdm import tqdm

from src.config import CSV_FILE_PATH


def write_sequences_to_csv(
    gene_sequences: List[str], non_gene_sequences: List[str]
) -> str:
    logging.info("Writing sequences to CSV...")

    current_time = str(time.time())  # TODO Change this to v1.0 or other versioning
    csv_file_path = os.path.join(CSV_FILE_PATH, current_time)
    os.makedirs(csv_file_path, exist_ok=True)
    file_path = os.path.join(csv_file_path, "raw_data.csv")

    with open(file_path, "w") as f:
        f.write("IsGene,Sequence\n")
        for sequence in gene_sequences:
            f.write(f"1,{sequence}\n")
        for sequence in non_gene_sequences:
            f.write(f"0,{sequence}\n")

    logging.info("Sequences written to CSV.")
    return file_path


def generate_non_genes(
    gene_sequences: List[str], genomes: List[SeqRecord]
) -> List[str]:
    logging.info("Generating non-genes...")

    non_genes = []

    random_length = get_random_length_from_gene_distribution(gene_sequences)

    for _ in tqdm(
        range(len(gene_sequences)), desc="Generating non-genes", unit="genome"
    ):
        genome = get_random_genome(genomes)

        non_gene_candidate = get_random_gene_frame(str(genome.seq), random_length)

        # if non_gene_candidate not in gene_sequences: <- this takes too long
        non_genes.append(non_gene_candidate)

    logging.info("Non-genes generated.")
    return non_genes


def get_random_length_from_gene_distribution(gene_sequences: List[str]) -> int:
    # gene_lengths = [len(gene) for gene in gene_sequences if len(gene) > 2]
    gene_lengths = [len(gene) for gene in gene_sequences]
    return random.choice(gene_lengths)


def get_random_gene_frame(genome: str, random_length: int) -> str:
    genome_length = len(genome)
    random_start_index = random.randint(0, genome_length - random_length)
    random_end_index = random_start_index + random_length
    return genome[random_start_index:random_end_index]


# Todo - Make this a specific genome, not a random one
def get_random_genome(genomes: List[SeqRecord]) -> SeqRecord:
    if not genomes:
        raise ValueError("The genomes list is empty.")
    return random.choice(genomes)


def generate_raw_csv(genes: List[SeqRecord], genomes: List[SeqRecord]) -> str:
    if not genomes:
        logging.error("The genomes list is empty.")
        raise ValueError("The genomes list is empty.")
    if not genes:
        logging.error("The genes list is empty.")
        raise ValueError("The genes list is empty.")

    # TODO this should not be here
    # Filter out genomes that are too short
    # genomes = [genome for genome in genomes if len(genome) > 100000]
    # Filter out genes that are too long
    # genes = [gene for gene in genes if len(gene) < 80000]

    gene_sequences = [str(gene.seq) for gene in genes if gene.seq]
    non_gene_sequences = generate_non_genes(gene_sequences, genomes)

    return write_sequences_to_csv(gene_sequences, non_gene_sequences)
