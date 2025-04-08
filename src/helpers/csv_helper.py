import logging
from pathlib import Path
from typing import List


def write_string_sequences_to_csv(
    sequences: List[str],
    csv_destination_path: Path,
    write_header: bool,
    is_real_gene: bool,
) -> None:
    logging.info("Writing sequences to CSV...")

    try:
        with open(csv_destination_path, "w") as f:
            if write_header:
                f.write("IsGene,Sequence\n")
            for sequence in sequences:
                f.write(f"{1 if is_real_gene else 0},{sequence}\n")
    except Exception as e:
        logging.error(f"Error writing to CSV: {e}")
        raise e

    logging.info("Sequences written to CSV.")
