import logging
from pathlib import Path
from typing import List

import pandas as pd


def write_string_sequences_to_csv(
    sequences: List[str],
    csv_destination_path: Path,
    write_header: bool,
    is_real_gene: bool,
) -> None:
    logging.info("Writing sequences to CSV...")

    try:
        with open(csv_destination_path, "w" if write_header else "a") as f:
            if write_header:
                f.write("IsGene,Sequence\n")
            for sequence in sequences:
                f.write(f"{1 if is_real_gene else 0},{sequence}\n")
    except Exception as e:
        logging.error(f"Error writing to CSV: {e}")
        raise e

    logging.info("Sequences written to CSV.")


def copy_csv_file(
    source_csv_file: Path,
    destination_csv_file: Path,
) -> None:
    logging.info(
        f"Copying CSV file from {source_csv_file} to {destination_csv_file}..."
    )
    try:
        df = pd.read_csv(source_csv_file)
        df.to_csv(destination_csv_file, index=False)
    except Exception as e:
        logging.error(f"Error copying CSV file: {e}")
        raise e
    logging.info("CSV file copied successfully.")


def get_iterable_series_sequence_lengths_from_csv(
    csv_file: Path,
) -> pd.Series:
    df = pd.read_csv(csv_file)
    if "Sequence" not in df.columns:
        raise ValueError("The required 'Sequence' column is missing in the CSV file.")
    return df["Sequence"].str.len()
