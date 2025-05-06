import logging
from pathlib import Path
from typing import List

import dvc.api

from src.helpers.csv_helper import write_string_sequences_to_csv
from src.helpers.fasta_helper import (
    get_string_sequences_from_directory,
)


def convert_fasta_folder_into_csv(
    source_directory: Path, csv_destination_path: Path
) -> None:
    """Converts FASTA files in a directory to a CSV file.

    Args:
        source_directory (Path): Path to the directory containing FASTA files.
        csv_destination_path (Path): Path where the CSV file will be saved.
    """

    logging.info("Converting FASTA files to CSV...")
    try:
        sequences: List[str] = get_string_sequences_from_directory(source_directory)
        write_string_sequences_to_csv(
            sequences=sequences,
            csv_destination_path=csv_destination_path,
            write_header=True,
            is_real_gene=True,
        )
    except Exception as e:
        logging.error(f"Error during conversion: {e}")
        raise e


if __name__ == "__main__":
    params = dvc.api.params_show(stages="csv_conversion")
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    convert_fasta_folder_into_csv(
        Path(params["paths"]["raw"]["genes_dir"]),
        Path(params["paths"]["raw"]["genes_csv"]),
    )
