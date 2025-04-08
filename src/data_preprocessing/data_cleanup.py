import logging
from pathlib import Path

import dvc.api
import pandas as pd
from pandas import DataFrame


def clean_data(
    source_path: Path,
    destination_path: Path,
    min_length: int = 50,
    max_length: int = 1000,
) -> DataFrame:
    """Cleans the data by removing duplicates and filtering sequences by length.

    Args:
        source_path (Path): Path to the source CSV file.
        destination_path (Path): Path to the destination CSV file.
        min_length (int): Minimum length of sequences to keep.
        max_length (int): Maximum length of sequences to keep.

    Returns:
        DataFrame: The cleaned DataFrame.

    Raises:
        FileNotFoundError: If the source file does not exist.
        ValueError: If the input CSV is invalid or missing required columns.
    """
    if min_length > max_length:
        raise ValueError("min_length cannot be greater than max_length.")

    try:
        df: DataFrame = pd.read_csv(source_path)
    except FileNotFoundError:
        logging.error(f"Source file not found: {source_path}")
        raise
    except pd.errors.EmptyDataError:
        logging.error(f"Source file is empty or invalid: {source_path}")
        raise

    if "Sequence" not in df.columns:
        raise ValueError("The input CSV must contain a 'Sequence' column.")
    if df["Sequence"].empty:
        raise ValueError("The 'Sequence' column is empty.")

    logging.debug(f"Initial number of rows: {len(df)}")

    logging.info("Removing duplicates from CSV...")
    df.drop_duplicates(subset=["Sequence"], inplace=True)
    logging.info("Duplicates removed from CSV.")

    logging.info(
        f"Removing sequences not within length range {min_length}-{max_length} "
        "from CSV..."
    )
    df = df[df["Sequence"].str.len().between(min_length, max_length)]
    logging.info(
        f"Sequences not within length range {min_length}-{max_length} removed from CSV."
    )

    logging.info("Resetting index after cleaning...")
    df.reset_index(drop=True, inplace=True)
    df.to_csv(destination_path, index=False)
    logging.debug(f"Number of rows after cleaning: {len(df)}")
    return df


if __name__ == "__main__":
    params = dvc.api.params_show(stages="data_cleanup")
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    clean_data(
        source_path=Path(params["paths"]["csv_file"]),
        destination_path=Path(params["paths"]["csv_file_clean"]),
        min_length=params["gene_lengths"]["min"],
        max_length=params["gene_lengths"]["max"],
    )
