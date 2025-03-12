# import pandas as pd
# from pandas import DataFrame
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.metrics import accuracy_score
# from sklearn.model_selection import GridSearchCV, train_test_split

# # Load the data
# genes_data_frame: DataFrame = pd.read_csv("data/processed/training.csv")
# sequences = genes_data_frame["Sequence"].values
# labels = genes_data_frame["IsGene"].values

# k = 3


# def get_kmers(seq, k):
#     return [seq[i : i + k] for i in range(len(seq) - k + 1)]


# # Convert sequences to k-mers
# kmers_list = [" ".join(get_kmers(seq, k)) for seq in sequences]

# # Convert k-mers to a numerical feature matrix
# vectorizer = CountVectorizer()
# X = vectorizer.fit_transform(kmers_list)

# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(
#     X, labels, test_size=0.15, random_state=42
# )

# # Define the RandomForest classifier
# random_forest_classifier = RandomForestClassifier(random_state=42)

# # Define the parameter grid
# param_grid = {
#     "n_estimators": [100, 200, 300],
#     "max_depth": [None, 10, 20, 30],
#     "min_samples_split": [2, 5, 10],
#     "min_samples_leaf": [1, 2, 4],
#     "bootstrap": [True, False],
# }

# # Create the GridSearchCV object
# grid_search = GridSearchCV(
#     estimator=random_forest_classifier,
#     param_grid=param_grid,
#     cv=3,
#     n_jobs=-1,
#     verbose=2,
# )

# # Perform grid search
# grid_search.fit(X_train, y_train)

# # Get the best parameters and best estimator
# best_params = grid_search.best_params_
# best_estimator = grid_search.best_estimator_

# print("Best Parameters:", best_params)

# # Make predictions with the best estimator
# y_pred = best_estimator.predict(X_test)

# # Evaluate accuracy
# print("Accuracy:", accuracy_score(y_test, y_pred))

# # Best Parameters: {'bootstrap': True, 'max_depth': 10, 'min_samples_leaf': 4, 'min_samples_split': 10, 'n_estimators': 300}


import pandas as pd
from pandas import DataFrame
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
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

    # Define the RandomForest classifier with the best parameters
    best_params = {
        "bootstrap": True,
        "max_depth": 10,
        "min_samples_leaf": 4,
        "min_samples_split": 10,
        "n_estimators": 300,
    }

    random_forest_classifier = RandomForestClassifier(random_state=42, **best_params)

    # Train the classifier
    random_forest_classifier.fit(X_train, y_train)

    return random_forest_classifier, X_test, y_test


def predict(random_forest_classifier, X_test, y_test):
    y_pred = random_forest_classifier.predict(X_test)
    logging.info("Accuracy:", accuracy_score(y_test, y_pred))
