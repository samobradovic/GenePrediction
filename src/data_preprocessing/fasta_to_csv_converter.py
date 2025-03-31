import logging
import os
from pathlib import Path
from typing import List

import dvc.api
from Bio import SeqIO

from src.data_preprocessing.non_gene_generator import generate_non_genes


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


def convert(raw_genomes_path: Path, raw_genes_path: Path) -> None:
    logging.info("Converting FASTA files to CSV...")

    genes = get_string_sequences_from_directory(raw_genes_path)
    genomes = get_string_sequences_from_directory(raw_genomes_path)

    non_gene_seqences = generate_non_genes(genes, genomes)

    write_sequences_to_csv(genes, non_gene_seqences)


params = dvc.api.params_show(stages="csv_converter")
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
if __name__ == "__main__":
    convert(
        Path(params["paths"]["genomes_raw_data_folder"]),
        Path(params["paths"]["genes_raw_data_folder"]),
    )
