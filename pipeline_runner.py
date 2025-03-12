from typing import Iterator, List

from Bio.SeqRecord import SeqRecord

from src.data_loading.gene_fetcher import load_genes
from src.data_loading.genome_fetcher import load_genomes

from src.data_preprocessing.csv_converter import generate_train_csv

from src.gene_prediction.random_forest_gene_predictor import (
    train as train_rf,
    predict as predict_rf,
)
from src.gene_prediction.xgboost_gene_predictor import (
    train as train_xgb,
    predict as predict_xgb,
)


def run():

    # genomes = load_genomes(load_from_pickle=True, load_from_files=True)
    # genes = load_genes(load_from_pickle=True, load_from_files=True)

    # generate_train_csv(genes, genomes)

    trained_model, X_test, y_test = train_xgb()
    predict_xgb(trained_model, X_test, y_test)
