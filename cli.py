import logging

from pipeline_runner import run


def main():
    # parser = argparse.ArgumentParser(
    #     description="Pipeline to do translation/transcription."
    # )

    # parser.add_argument(
    #     "--verbose",
    #     action="store_true",
    #     help="Increase output verbosity",
    # )

    # args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    run()


if __name__ == "__main__":
    main()
