import logging
import os
import time

import dill

from config import MODEL_FOLDER_PATH


def get_folder_size(folder_path: str) -> float:
    total_size = 0
    for dirpath, _, filenames in os.walk(folder_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    return total_size / (1024**3)


def save_pkl(pickle_object, file_path: str) -> None:
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "wb") as f:
        dill.dump(pickle_object, f)
    logging.info(f"Object saved as pkl to {file_path}")


def save_binaries(model, vectorizer, dataset_size: str) -> None:
    versioned_file_path = get_versioned_file_path(MODEL_FOLDER_PATH)

    model_path = os.path.join(
        versioned_file_path, f"xgb_model_{dataset_size.upper()}.pkl"
    )
    vectorizer_path = os.path.join(
        versioned_file_path, f"count_vectorizer_{dataset_size.upper()}.pkl"
    )

    save_pkl(model, model_path)
    save_pkl(vectorizer, vectorizer_path)


def load_binaries(dataset_size: str, folder_path: str | None = None) -> tuple:
    if folder_path is None:
        folders = [
            f
            for f in os.listdir(MODEL_FOLDER_PATH)
            if os.path.isdir(os.path.join(MODEL_FOLDER_PATH, f))
        ]
        folder_path = os.path.join(MODEL_FOLDER_PATH, sorted(folders)[-1])

    model_path = os.path.join(folder_path, f"xgb_model_{dataset_size.upper()}.pkl")
    vectorizer_path = os.path.join(
        folder_path, f"count_vectorizer_{dataset_size.upper()}.pkl"
    )

    with open(model_path, "rb") as f:
        model = dill.load(f)
    with open(vectorizer_path, "rb") as f:
        vectorizer = dill.load(f)

    return model, vectorizer


def get_versioned_file_path(file_path: str) -> str:
    current_time = str(time.time())
    new_file_path = os.path.join(file_path, current_time)
    os.makedirs(new_file_path, exist_ok=True)
    return new_file_path
