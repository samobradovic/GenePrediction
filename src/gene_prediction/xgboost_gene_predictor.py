import logging

from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier


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
        "eval_metric": "logloss",  # Specify the evaluation metric here
        "early_stopping_rounds": 10,  # Stops training if validation metric doesn't improve for 10 rounds
    }

    xgb_classifier = XGBClassifier(**best_params)

    logging.info("Training XGBoost model with validation...")
    xgb_classifier.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=True,  # Set to True to see progress during training
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
