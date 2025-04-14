import logging
from pathlib import Path

import dill
import dvc.api
import matplotlib.pyplot as plt
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


def evaluate_xgb_model(model, X_test, y_test):
    """
    Evaluate and visualize an XGBoost classification model's performance.
    """
    # Predictions
    y_pred = model.predict(X_test)
    y_probs = model.predict_proba(X_test)[:, 1]

    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"✅ Model Accuracy: {accuracy:.2%}\n")

    # Classification report
    print("Classification Report:\n", classification_report(y_test, y_pred))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=["Negative", "Positive"]
    )
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix")
    # plt.show()
    plt.savefig("results/confusion_matrix.png")

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_probs)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="blue", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color="gray", lw=2, linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic Curve")
    plt.legend(loc="lower right")
    # plt.show()
    plt.savefig("results/receiver_operating_characteristic.png")

    # Precision-Recall curve
    precision, recall, _ = precision_recall_curve(y_test, y_probs)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color="green", lw=2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    # plt.show()
    plt.savefig("results/precision_recall.png")

    # SHAP values

    # if hasattr(X_test, "toarray"):
    #     X_test = X_test.toarray()

    # explainer = shap.Explainer(model, X_test)
    # shap_values = explainer(X_test)
    # shap.summary_plot(shap_values, X_test)


if __name__ == "__main__":
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    params = dvc.api.params_show(stages="evaluation")
    with open(Path(params["paths"]["testing_vectorized_file"]), "rb") as f:
        X_test = dill.load(f)
    y_test = pd.read_csv(Path(params["paths"]["testing_csv_file"]))["IsGene"].values

    model = dill.load(open(Path(params["paths"]["model_file"]), "rb"))

    evaluate_xgb_model(model, X_test, y_test)
