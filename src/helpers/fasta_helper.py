import logging
import os
from pathlib import Path
from typing import List

from Bio import SeqIO


def get_string_sequences_from_directory(directory: Path) -> List[str]:
    sequences = []
    logging.info(f"Reading sequences from directory: {directory}")
    if os.path.isdir(directory):
        for file in os.listdir(directory):
            if file.endswith(".fasta") or file.endswith(".fa"):
                file_path = os.path.join(directory, file)
                logging.info(f"Processing file: {file_path}")
                try:
                    records = SeqIO.parse(file_path, "fasta")
                    for record in records:
                        sequences.append(str(record.seq))
                except Exception as e:
                    logging.error(f"Error processing file {file}: {e}")
                    raise e
    else:
        logging.error(f"Path {directory} is not a directory.")

    if not sequences:
        logging.warning(f"No sequences found in {directory}.")

    return sequences
