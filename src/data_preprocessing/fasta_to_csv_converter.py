import argparse
import logging
import os
from typing import List

from Bio import SeqIO

from src.config import (
    CSV_FILE_PATH,
    GENES_RAW_DATA_FOLDER_PATH,
    GENOMES_RAW_DATA_FOLDER_PATH,
)
from src.data_preprocessing.non_gene_generator import generate_non_genes


def write_sequences_to_csv(
    gene_sequences: List[str], non_gene_sequences: List[str]
) -> str:
    logging.info("Writing sequences to CSV...")

    with open(CSV_FILE_PATH, "w") as f:
        f.write("IsGene,Sequence\n")
        for sequence in gene_sequences:
            f.write(f"1,{sequence}\n")
        for sequence in non_gene_sequences:
            f.write(f"0,{sequence}\n")

    logging.info("Sequences written to CSV.")
    return CSV_FILE_PATH


def get_string_sequences_from_directory(directory: str) -> List[str]:
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


def convert(raw_genomes_path: str, raw_genes_path: str) -> str:
    logging.info("Converting FASTA files to CSV...")

    genes = get_string_sequences_from_directory(raw_genes_path)
    genomes = get_string_sequences_from_directory(raw_genomes_path)

    non_gene_seqences = generate_non_genes(genes, genomes)

    return write_sequences_to_csv(genes, non_gene_seqences)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw_genomes_path",
        type=str,
        required=False,
        default=f"{GENOMES_RAW_DATA_FOLDER_PATH}",
        help="Path to the raw genomes folder",
    )
    parser.add_argument(
        "--raw_genes_path",
        type=str,
        required=False,
        default=f"{GENES_RAW_DATA_FOLDER_PATH}",
        help="Path to the raw genes folder",
    )
    args = parser.parse_args()

    convert(args.raw_genomes_path, args.raw_genes_path)
