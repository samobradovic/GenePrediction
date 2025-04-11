import logging
from pathlib import Path
from typing import Tuple

import dill
import dvc.api
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from xgboost import XGBClassifier

from src.helpers.filesystem_helper import save_pkl


def get_training_and_validation_data(
    paths: dict,
) -> Tuple[csr_matrix, np.ndarray, csr_matrix, np.ndarray]:
    """Loads training and validation data from the specified paths.

    Args:
        paths: Dictionary containing paths to the training and validation data files.

    Returns:
        X_train: Training feature set.
        y_train: Training labels.
        X_val: Validation feature set.
        y_val: Validation labels.
    """
    try:
        with open(paths["training_vectorized_file"], "rb") as f:
            X_train: csr_matrix = dill.load(f)
        with open(paths["validation_vectorized_file"], "rb") as f:
            X_val: csr_matrix = dill.load(f)
        y_train: np.ndarray = np.array(
            pd.read_csv(paths["training_csv_file"])["IsGene"].values
        )
        y_val: np.ndarray = np.array(
            pd.read_csv(paths["validation_csv_file"])["IsGene"].values
        )
    except FileNotFoundError as e:
        logging.error(f"File not found: {e.filename}")
        raise e

    return X_train, y_train, X_val, y_val


def train(paths: dict, xgb_params: dict):
    """Trains an XGBoost classifier using the specified parameters.

    Args:
        paths: Dictionary containing paths to the training and validation data files.
        xgb_params: Dictionary containing XGBoost parameters.

    """

    X_train, y_train, X_val, y_val = get_training_and_validation_data(paths)

    xgb_classifier = XGBClassifier(**xgb_params)

    logging.info("Training XGBoost model with validation...")
    xgb_classifier.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=True,
    )
    logging.info("Training complete")

    save_pkl(xgb_classifier, paths["model_file"])

    return xgb_classifier


if __name__ == "__main__":
    params = dvc.api.params_show(stages="xgb_training")
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    train(
        paths={
            "training_vectorized_file": Path(
                params["paths"]["training_vectorized_file"]
            ),
            "training_csv_file": Path(params["paths"]["training_csv_file"]),
            "validation_vectorized_file": Path(
                params["paths"]["validation_vectorized_file"]
            ),
            "validation_csv_file": Path(params["paths"]["validation_csv_file"]),
            "model_file": Path(params["paths"]["model_file"]),
        },
        xgb_params={
            "n_estimators": params["xgb_params"]["n_estimators"],
            "max_depth": params["xgb_params"]["max_depth"],
            "learning_rate": params["xgb_params"]["learning_rate"],
            "subsample": params["xgb_params"]["subsample"],
            "colsample_bytree": params["xgb_params"]["colsample_bytree"],
            "random_state": params["xgb_params"]["random_state"],
            "eval_metric": params["xgb_params"]["eval_metric"],
            "early_stopping_rounds": params["xgb_params"]["early_stopping_rounds"],
        },
    )
