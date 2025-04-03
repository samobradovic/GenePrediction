import logging
import os
import random
from pathlib import Path
from typing import List

import dvc.api
from Bio import SeqIO
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


def write_sequences_to_csv(
    gene_sequences: List[str], non_gene_sequences: List[str]
) -> None:
    logging.info("Writing sequences to CSV...")

    with open(Path(params["paths"]["csv_file"]), "w") as f:
        f.write("IsGene,Sequence\n")
        for sequence in gene_sequences:
            f.write(f"1,{sequence}\n")
        for sequence in non_gene_sequences:
            f.write(f"0,{sequence}\n")

    logging.info("Sequences written to CSV.")


def get_string_sequences_from_directory(directory: Path) -> List[str]:
    sequences = []
    if os.path.isdir(directory):
        for file in os.listdir(directory):
            if file.endswith(".fasta") or file.endswith(".fa"):
                records = SeqIO.parse(os.path.join(directory, file), "fasta")
                for record in records:
                    sequences.append(str(record.seq))
    else:
        logging.error(f"Path {directory} is not a directory.")
    return sequences


params = dvc.api.params_show(stages="non_gene_generation")
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
if __name__ == "__main__":
    gene_sequences = []
    with open(Path(params["paths"]["genes_csv_file"]), "r") as f:
        next(f)
        for line in f:
            gene_sequences.append(line.strip().split(",")[1])

    genomes = get_string_sequences_from_directory(
        Path(params["paths"]["genomes_raw_data_folder"]),
    )

    non_genes = generate_non_genes(gene_sequences, genomes)

    write_sequences_to_csv(gene_sequences, non_genes)
