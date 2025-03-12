import logging
import os

import dill


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
