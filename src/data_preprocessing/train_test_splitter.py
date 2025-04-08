import logging
from pathlib import Path

import dvc.api
import pandas as pd
from pandas import DataFrame
from sklearn.model_selection import train_test_split


def save_train_test_validation_csvs(
    data_splits: dict,
) -> None:
    logging.info("Saving training, validation, and testing CSVs...")

    pd.DataFrame(
        {"Sequence": data_splits["X_train"], "IsGene": data_splits["y_train"]}
    ).to_csv(Path(params["paths"]["training_csv_file"]), index=False)
    pd.DataFrame(
        {"Sequence": data_splits["X_val"], "IsGene": data_splits["y_val"]}
    ).to_csv(Path(params["paths"]["validation_csv_file"]), index=False)
    pd.DataFrame(
        {"Sequence": data_splits["X_test"], "IsGene": data_splits["y_test"]}
    ).to_csv(Path(params["paths"]["testing_csv_file"]), index=False)

    logging.info("CSV files saved successfully.")


def train_test_validation_split(
    csv_file_path: Path, test_size: float = 0.20, validation_size: float = 0.20
):
    genes_data_frame: DataFrame = pd.read_csv(csv_file_path)
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
        {
            "X_train": X_train,
            "X_val": X_val,
            "X_test": X_test,
            "y_train": y_train,
            "y_val": y_val,
            "y_test": y_test,
        }
    )


params = dvc.api.params_show(stages="train_test_split")
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
if __name__ == "__main__":
    train_test_validation_split(Path(params["paths"]["csv_file_clean"]))
