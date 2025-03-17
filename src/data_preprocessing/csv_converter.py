import logging
import os
import random
from typing import List

from Bio.SeqRecord import SeqRecord
from tqdm import tqdm

from src.config import CSV_PROCESSED_FILE_PATH


def write_sequences_to_csv(is_gene: bool, sequences: List[str]) -> None:
    if not os.path.exists(CSV_PROCESSED_FILE_PATH):
        with open(CSV_PROCESSED_FILE_PATH, "w") as f:
            f.write("IsGene,Sequence\n")
            for sequence in sequences:
                f.write(f"{int(is_gene)},{sequence}\n")
    else:
        with open(CSV_PROCESSED_FILE_PATH, "a") as f:
            for sequence in sequences:
                f.write(f"{int(is_gene)},{sequence}\n")


def generate_non_genes(
    gene_sequences: List[str], genomes: List[SeqRecord]
) -> List[str]:
    non_genes = []

    random_length = get_random_length_from_gene_distribution(gene_sequences)

    for _ in tqdm(
        range(len(gene_sequences)), desc="Generating non-genomes", unit="genome"
    ):  # seems to be about 130 genome/s

        genome = get_random_genome(genomes)

        non_gene_candidate = get_random_gene_frame(
            gene_sequences, str(genome.seq), random_length
        )

        # if non_gene_candidate not in gene_sequences:
        non_genes.append(non_gene_candidate)

    return non_genes


def get_gene_length_distribution(gene_sequences: List[str]) -> List[int]:
    gene_lengths = [len(gene) for gene in gene_sequences if len(gene) > 2]
    return gene_lengths


def get_random_length_from_gene_distribution(gene_sequences: List[str]) -> int:
    gene_lengths = get_gene_length_distribution(gene_sequences)
    return random.choice(gene_lengths)


def get_random_gene_frame(
    gene_sequences: List[str], genome: str, random_length: int
) -> str:

    genome_length = len(genome)
    random_start_index = random.randint(0, genome_length - random_length)
    random_end_index = random_start_index + random_length
    return genome[random_start_index:random_end_index]


# Todo - Make this a specific genome, not a random one
def get_random_genome(genomes: List[SeqRecord]) -> SeqRecord:
    if not genomes:
        raise ValueError("The genomes list is empty.")
    return random.choice(genomes)


def save_genes_to_csv(genes: List[SeqRecord]) -> List[str]:
    logging.info("Saving genes to CSV...")
    gene_sequences: List[str] = [str(gene.seq) for gene in genes if gene.seq]
    write_sequences_to_csv(is_gene=True, sequences=gene_sequences)
    logging.info("Genes saved to CSV.")
    return gene_sequences


def save_non_genes_to_csv(
    gene_sequences: List[str], genomes: List[SeqRecord]
) -> List[str]:
    logging.info("Generating non-genes...")
    random_non_genes = generate_non_genes(gene_sequences, genomes)
    logging.info("Saving non-genes to CSV...")
    write_sequences_to_csv(is_gene=False, sequences=random_non_genes)
    logging.info("Non-genes saved to CSV.")
    return random_non_genes


def generate_train_csv(genes: List[SeqRecord], genomes: List[SeqRecord]) -> None:
    if not genomes:
        logging.error("The genomes list is empty.")
        raise ValueError("The genomes list is empty.")
    if not genes:
        logging.error("The genes list is empty.")
        raise ValueError("The genes list is empty.")

    # Filter out genomes that are too short
    genomes = [genome for genome in genomes if len(genome) > 100000]

    # Filter out genes that are too long
    genes = [gene for gene in genes if len(gene) < 80000]

    gene_sequences = save_genes_to_csv(genes)
    save_non_genes_to_csv(gene_sequences, genomes)
