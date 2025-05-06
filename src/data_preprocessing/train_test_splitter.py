import logging
from pathlib import Path

import dvc.api
import pandas as pd
from pandas import DataFrame
from sklearn.model_selection import train_test_split


def save_train_test_validation_csvs(
    data_splits: dict,
    destination_train_csv: Path,
    destination_validation_csv: Path,
    destination_test_csv: Path,
) -> None:
    """Saves the training, validation, and testing data splits to CSV files.

    Args:
        data_splits (dict): Dictionary containing the data splits.
        destination_train_csv (Path): Path to save the training CSV file.
        destination_validation_csv (Path): Path to save the validation CSV file.
        destination_test_csv (Path): Path to save the testing CSV file.
    """

    logging.info("Saving training, validation, and testing CSVs...")

    pd.DataFrame(
        {"Sequence": data_splits["X_train"], "IsGene": data_splits["y_train"]}
    ).to_csv(destination_train_csv, index=False)
    pd.DataFrame(
        {"Sequence": data_splits["X_val"], "IsGene": data_splits["y_val"]}
    ).to_csv(destination_validation_csv, index=False)
    pd.DataFrame(
        {"Sequence": data_splits["X_test"], "IsGene": data_splits["y_test"]}
    ).to_csv(destination_test_csv, index=False)

    logging.info("CSV files saved successfully.")


def train_test_validation_split(
    csv_file_path: Path,
    test_size: float,
    validation_size: float,
    destination_train_csv: Path,
    destination_validation_csv: Path,
    destination_test_csv: Path,
):
    """Splits the dataset into training, validation, and testing sets.

    Args:
        csv_file_path (Path): Path to the input CSV file.
        test_size (float): Proportion of the dataset to include in the test split.
        validation_size (float): Proportion of the dataset to include in
          the validation split.
        destination_train_csv (Path): Path to save the training CSV file.
        destination_validation_csv (Path): Path to save the validation CSV file.
        destination_test_csv (Path): Path to save the testing CSV file.

    Raises:
        ValueError: If the input CSV is missing required columns.
    """
    genes_data_frame: DataFrame = pd.read_csv(csv_file_path)
    if (
        "Sequence" not in genes_data_frame.columns
        or "IsGene" not in genes_data_frame.columns
    ):
        raise ValueError("Input CSV must contain 'Sequence' and 'IsGene' columns.")
    sequences = genes_data_frame["Sequence"].tolist()
    labels = genes_data_frame["IsGene"].values

    logging.info("Splitting data into training, validation, and testing sets...")
    X_train, X_temp, y_train, y_temp = train_test_split(
        sequences, labels, test_size=(test_size + validation_size), random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=(test_size / (test_size + validation_size)),
        random_state=42,
    )
    logging.info("Data split complete.")

    save_train_test_validation_csvs(
        data_splits={
            "X_train": X_train,
            "X_val": X_val,
            "X_test": X_test,
            "y_train": y_train,
            "y_val": y_val,
            "y_test": y_test,
        },
        destination_train_csv=destination_train_csv,
        destination_validation_csv=destination_validation_csv,
        destination_test_csv=destination_test_csv,
    )


if __name__ == "__main__":
    params = dvc.api.params_show(stages="train_test_split")
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    train_test_validation_split(
        csv_file_path=Path(params["paths"]["raw"]["combined_csv_clean"]),
        test_size=params["train_test_split"]["test_size"],
        validation_size=params["train_test_split"]["validation_size"],
        destination_train_csv=Path(params["paths"]["processed"]["training_csv"]),
        destination_validation_csv=Path(params["paths"]["processed"]["validation_csv"]),
        destination_test_csv=Path(params["paths"]["processed"]["testing_csv"]),
    )
