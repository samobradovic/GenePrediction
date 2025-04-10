import logging
from pathlib import Path

import dill
import dvc.api
import pandas as pd
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

from helpers.filesystem_helper import save_pkl


def train(X_train, y_train, X_val, y_val):
    """Trains an XGBoost classifier with validation.

    Args:
        X_train: Training feature set.
        y_train: Training labels.
        X_val: Validation feature set.
        y_val: Validation labels.

    Returns:
        XGBClassifier: Trained XGBoost classifier.
    """
    best_params = {
        "n_estimators": 300,
        "max_depth": 10,
        "learning_rate": 0.1,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "random_state": 42,
        "eval_metric": "logloss",
        "early_stopping_rounds": 10,
    }

    xgb_classifier = XGBClassifier(**best_params)

    logging.info("Training XGBoost model with validation...")
    xgb_classifier.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=True,
    )
    logging.info("Training complete")

    return xgb_classifier


def predict(xgb_classifier, X_test, y_test):
    """Makes predictions using the trained XGBoost classifier.

    Args:
        xgb_classifier: Trained XGBoost classifier.
        X_test: Test feature set.
        y_test: Test labels.

    Returns:
        None
    """
    y_pred = xgb_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    logging.info(f"Accuracy: {accuracy}")


params = dvc.api.params_show(stages="xgb_training")
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
if __name__ == "__main__":
    with open(Path(params["paths"]["training_vectorized_file"]), "rb") as f:
        X_train = dill.load(f)
    y_train = pd.read_csv(Path(params["paths"]["training_csv_file"]))["IsGene"].values

    with open(Path(params["paths"]["validation_vectorized_file"]), "rb") as f:
        X_val = dill.load(f)
    y_val = pd.read_csv(Path(params["paths"]["validation_csv_file"]))["IsGene"].values

    model = train(X_train, y_train, X_val, y_val)

    save_pkl(model, Path(params["paths"]["model_file"]))
