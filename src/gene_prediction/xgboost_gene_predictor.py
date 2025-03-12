import pandas as pd
from pandas import DataFrame
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from src.data_preprocessing.feature_extractor import extract_features
import logging


def train():
    genes_data_frame: DataFrame = pd.read_csv("data/processed/training.csv")
    sequences = genes_data_frame["Sequence"].values
    labels = genes_data_frame["IsGene"].values

    X = extract_features(sequences)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, labels, test_size=0.15, random_state=42
    )

    # Define the best parameters
    best_params = {
        "n_estimators": 300,
        "max_depth": 10,
        "learning_rate": 0.1,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "random_state": 42,
    }

    # Define the XGBoost classifier with the best parameters
    xgb_classifier = XGBClassifier(**best_params)

    # Train the classifier
    xgb_classifier.fit(X_train, y_train)

    return xgb_classifier, X_test, y_test


def predict(xgb_classifier, X_test, y_test):
    y_pred = xgb_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    logging.info(f"Accuracy: {accuracy}")
