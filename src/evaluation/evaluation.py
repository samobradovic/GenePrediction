import logging
from pathlib import Path

import dill
import dvc.api
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)


def generate_roc_curve(y_test: np.ndarray, y_probs: np.ndarray, path: Path) -> None:
    """
    Generate and save the ROC curve plot.

    Args:
        y_test (np.ndarray): True labels for the test set.
        y_probs (np.ndarray): Predicted probabilities for the positive class.
        path (Path): Path to save the ROC curve plot.

    """
    logging.info("Generating ROC curve...")
    fpr, tpr, _ = roc_curve(y_test, y_probs)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="blue", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color="gray", lw=2, linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic Curve")
    plt.legend(loc="lower right")
    plt.savefig(path)
    logging.info(f"ROC curve saved to {path}.")


def generate_precision_recall_curve(
    y_test: np.ndarray, y_probs: np.ndarray, path: Path
) -> None:
    """
    Generate and save the Precision-Recall curve plot.

    Args:
        y_test (np.ndarray): True labels for the test set.
        y_probs (np.ndarray): Predicted probabilities for the positive class.
        path (Path): Path to save the Precision-Recall curve plot.

    """
    logging.info("Generating precision-recall curve...")
    precision, recall, _ = precision_recall_curve(y_test, y_probs)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color="blue", lw=2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.savefig(path)
    logging.info(f"Precision-recall curve saved to {path}.")


def generate_confusion_matrix(
    y_test: np.ndarray, y_pred: np.ndarray, path: Path
) -> None:
    """
    Generate and save the confusion matrix plot.

    Args:
        y_test (np.ndarray): True labels for the test set.
        y_pred (np.ndarray): Predicted labels for the test set.
        path (Path): Path to save the confusion matrix plot.

    """
    logging.info("Generating confusion matrix...")
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=["Negative", "Positive"]
    )
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix")
    plt.savefig(path)
    logging.info(f"Confusion matrix saved to {path}.")


def generate_classification_report(
    y_test: np.ndarray, y_pred: np.ndarray, path: Path
) -> None:
    """
    Generate and save the classification report.

    Args:
        y_test (np.ndarray): True labels for the test set.
        y_pred (np.ndarray): Predicted labels for the test set.
        path (Path): Path to save the classification report.

    """
    logging.info("Generating classification report...")
    report = classification_report(y_test, y_pred, output_dict=True)
    df_report = pd.DataFrame(report).transpose()  # TODO: This does not work
    df_report.to_csv(path, index=True)
    logging.info(f"Classification report saved to {path}.")


def save_accuracy(y_test: np.ndarray, y_pred: np.ndarray, path: Path) -> None:
    """
    Save the accuracy of the model.

    Args:
        y_test (np.ndarray): True labels for the test set.
        y_pred (np.ndarray): Predicted labels for the test set.
        path (Path): Path to save the accuracy.

    """
    logging.info("Calculating accuracy...")
    accuracy = accuracy_score(y_test, y_pred)
    with open(path, "w") as f:
        f.write(f"Model Accuracy: {accuracy:.2%}\n")
    logging.info(f"Accuracy saved to {path}.")


def evaluate_xgb_model(
    model_file_path: Path,
    testing_vectorized_file: Path,
    testing_csv_file: Path,
    accuracy_path: Path,
    classification_report_path: Path,
    confusion_matrix_path: Path,
    roc_curve_path: Path,
    precision_recall_curve_path: Path,
) -> None:
    """
    Evaluate the XGBoost model on the testing data.
    Load the model and testing data, make predictions, and save the evaluation metrics.

    Args:
        model_file_path (Path): Path to the trained model file.
        testing_vectorized_file (Path): Path to the vectorized testing data file.
        testing_csv_file (Path): Path to the CSV file containing the true labels.
        accuracy_path (Path): Path to save the accuracy report.
        classification_report_path (Path): Path to save the classification report.
        confusion_matrix_path (Path): Path to save the confusion matrix plot.
        roc_curve_path (Path): Path to save the ROC curve plot.
        precision_recall_curve_path (Path): Path to save the Precision-Recall curve.

    """

    logging.info("Loading testing data...")
    with open(testing_vectorized_file, "rb") as f:
        X_test = dill.load(f)
    y_test = pd.read_csv(testing_csv_file)["IsGene"].to_numpy()
    logging.info("Testing data loaded.")
    logging.info("Loading model...")
    model = dill.load(open(model_file_path, "rb"))
    logging.info("Model loaded.")

    logging.info("Running predictions...")
    y_pred = model.predict(X_test)
    y_probs = model.predict_proba(X_test)[:, 1]
    logging.info("Predictions completed.")
    save_accuracy(y_test, y_pred, accuracy_path)
    generate_classification_report(y_test, y_pred, classification_report_path)
    generate_confusion_matrix(y_test, y_pred, confusion_matrix_path)
    generate_roc_curve(y_test, y_probs, roc_curve_path)
    generate_precision_recall_curve(y_test, y_probs, precision_recall_curve_path)


if __name__ == "__main__":
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    params = dvc.api.params_show(stages="evaluation")

    evaluate_xgb_model(
        model_file_path=Path(params["paths"]["model_file"]),
        testing_vectorized_file=Path(params["paths"]["testing_vectorized_file"]),
        testing_csv_file=Path(params["paths"]["testing_csv_file"]),
        accuracy_path=Path(params["paths"]["results"]["accuracy"]),
        classification_report_path=Path(
            params["paths"]["results"]["classification_report"]
        ),
        confusion_matrix_path=Path(params["paths"]["results"]["confusion_matrix"]),
        roc_curve_path=Path(params["paths"]["results"]["roc_curve"]),
        precision_recall_curve_path=Path(
            params["paths"]["results"]["precision_recall_curve"]
        ),
    )
