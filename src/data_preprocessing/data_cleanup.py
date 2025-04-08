import logging
from pathlib import Path

import dvc.api
import pandas as pd


def remove_duplicates():
    logging.info("Removing duplicates from CSV...")

    df = pd.read_csv(Path(params["paths"]["csv_file"]))
    df.drop_duplicates(subset=["Sequence"], inplace=True)
    df.to_csv(Path(params["paths"]["csv_file_clean"]), index=False)

    logging.info("Duplicates removed from CSV.")


def remove_sequences_by_length(min_length: int = 50, max_length: int = 1000):
    logging.info(
        f"Removing sequences not within length range {min_length}-{max_length} "
        "from CSV..."
    )

    df = pd.read_csv(Path(params["paths"]["csv_file"]))
    df = df[df["Sequence"].str.len().between(min_length, max_length)]
    df.to_csv(Path(params["paths"]["csv_file_clean"]), index=False)

    logging.info(
        f"Sequences not within length range {min_length}-{max_length} removed from CSV."
    )


params = dvc.api.params_show(stages="data_cleanup")
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
if __name__ == "__main__":
    remove_duplicates()
    remove_sequences_by_length(min_length=50, max_length=10000)
