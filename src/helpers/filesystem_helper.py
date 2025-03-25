import logging
import os
import time

import dill

from config import MODEL_FILE_PATH, VECTORIZER_FILE_PATH


def get_folder_size(folder_path: str) -> float:
    total_size = 0
    for dirpath, _, filenames in os.walk(folder_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    return total_size / (1024**3)


def save_pkl(pickle_object, file_path: str) -> None:
    directory = os.path.dirname(file_path)
    os.makedirs(directory, exist_ok=True)
    with open(file_path, "wb") as f:
        dill.dump(pickle_object, f)
    logging.info(f"Object saved as pkl to {file_path}")


def save_binaries(
    model, vectorizer, dataset_size: str, create_new_version: bool
) -> None:
    model_path = os.path.join(MODEL_FILE_PATH, f"{dataset_size.upper()}.pkl")
    vectorizer_path = os.path.join(VECTORIZER_FILE_PATH, f"{dataset_size.upper()}.pkl")
    if create_new_version:
        model_path = model_path.split(".")[0] + str(time.time()) + ".pkl"
        vectorizer_path = vectorizer_path.split(".")[0] + str(time.time()) + ".pkl"
    save_pkl(model, model_path)
    save_pkl(vectorizer, vectorizer_path)
