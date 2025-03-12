import os
import random
from typing import List

from Bio.SeqRecord import SeqRecord

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

    for _ in range(len(gene_sequences)):

        genome = get_random_genome(genomes)

        non_gene_candidate = get_random_gene_frame(gene_sequences, str(genome.seq))

        if non_gene_candidate not in gene_sequences:
            non_genes.append(non_gene_candidate)

    return non_genes


def get_gene_length_distribution(gene_sequences: List[str]) -> List[int]:
    gene_lengths = [len(gene) for gene in gene_sequences]
    return gene_lengths


def get_random_length_from_gene_distribution(gene_sequences: List[str]) -> int:
    gene_lengths = get_gene_length_distribution(gene_sequences)
    return random.choice(gene_lengths)


def get_random_gene_frame(gene_sequences: List[str], genome: str) -> str:
    random_length = get_random_length_from_gene_distribution(gene_sequences)
    genome_length = len(genome)
    random_start_index = random.randint(0, genome_length - random_length)
    random_end_index = random_start_index + random_length
    return genome[random_start_index:random_end_index]


# Todo - Make this a specific genome, not a random one
def get_random_genome(genomes: List[SeqRecord]) -> SeqRecord:
    return random.choice(genomes)


def save_genes_to_csv(genes: List[SeqRecord]) -> List[str]:
    gene_sequences: List[str] = [str(gene.seq) for gene in genes]
    write_sequences_to_csv(is_gene=True, sequences=gene_sequences)
    return gene_sequences


def save_non_genes_to_csv(
    gene_sequences: List[str], genomes: List[SeqRecord]
) -> List[str]:
    random_non_genes = generate_non_genes(gene_sequences, genomes)
    write_sequences_to_csv(is_gene=False, sequences=random_non_genes)
    return random_non_genes


def generate_train_csv(genes: List[SeqRecord], genomes: List[SeqRecord]) -> None:
    gene_sequences = save_genes_to_csv(genes)
    save_non_genes_to_csv(gene_sequences, genomes)
