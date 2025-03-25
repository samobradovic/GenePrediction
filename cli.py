import argparse
import logging

from config import DATASET_SIZE_OPTIONS
from pipeline_runner import run


def main():
    parser = argparse.ArgumentParser(
        description="Pipeline to train a gene prediction model."
    )

    parser.add_argument(
        "--dataset_size",
        type=str,
        choices=DATASET_SIZE_OPTIONS,
        default="small",
        help="Specify the dataset size to use for the pipeline. Options: 'small', "
        "'big', 'max'. Default is 'small'.",
    )

    parser.add_argument(
        "--create_new_version",
        action="store_true",
        help="Create a new version of the preprocessed data, vectorizer and model.",
    )

    parser.add_argument(
        "--regenerate_data",
        action="store_true",
        help="Regenerate the data for the pipeline.",
    )

    parser.add_argument(
        "--evaluate_only",
        action="store_true",
        help="Only evaluate the model without training.",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    run(
        args.dataset_size,
        args.regenerate_data,
        args.evaluate_only,
    )


if __name__ == "__main__":
    main()
