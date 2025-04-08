import logging
import os
from pathlib import Path
from typing import List

from Bio import SeqIO


def get_string_sequences_from_directory(directory: Path) -> List[str]:
    sequences = []
    if os.path.isdir(directory):
        for file in os.listdir(directory):
            if file.endswith(".fasta") or file.endswith(".fa"):
                try:
                    records = SeqIO.parse(file, "fasta")
                    for record in records:
                        sequences.append(str(record.seq))
                except Exception as e:
                    logging.error(f"Error processing file {file}: {e}")
    else:
        logging.error(f"Path {directory} is not a directory.")

    if not sequences:
        logging.warning(f"No sequences found in {directory}.")

    return sequences
