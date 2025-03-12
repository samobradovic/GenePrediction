import logging

from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier


def train(X_train, y_train):

    best_params = {
        "n_estimators": 300,
        "max_depth": 10,
        "learning_rate": 0.1,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "random_state": 42,
    }

    xgb_classifier = XGBClassifier(**best_params)

    logging.info("Training XGBoost model...")
    xgb_classifier.fit(X_train, y_train)
    logging.info("Training complete")

    return xgb_classifier


def predict(xgb_classifier, X_test, y_test):
    y_pred = xgb_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    logging.info(f"Accuracy: {accuracy}")
