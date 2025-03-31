import logging
import random
from typing import List

from tqdm import tqdm


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
def get_random_genome(genomes: List[str]) -> str:
    if not genomes:
        raise ValueError("The genomes list is empty.")
    return random.choice(genomes)


def generate_non_genes(gene_sequences: List[str], genomes: List[str]) -> List[str]:
    logging.info("Generating non-genes...")

    non_genes = []

    random_length = get_random_length_from_gene_distribution(gene_sequences)

    for _ in tqdm(range(len(gene_sequences)), desc="Generating non-genes", unit="gene"):
        genome = get_random_genome(genomes)

        non_gene_candidate = get_random_gene_frame(genome, random_length)

        # if non_gene_candidate not in gene_sequences: <- this takes too long
        non_genes.append(non_gene_candidate)

    logging.info("Non-genes generated.")
    return non_genes
