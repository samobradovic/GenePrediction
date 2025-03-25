from src.data_loading.gene_fetcher import load_genes
from src.data_loading.genome_fetcher import load_genomes
from src.data_preprocessing.csv_converter import generate_train_csv
from src.data_preprocessing.data_preprocessing import (
    extract_kmer_features,
    train_test_validation_split,
)
from src.evaluation.evaluation import evaluate_xgb_model
from src.gene_prediction.xgboost_gene_predictor import train
from src.helpers.filesystem_helper import save_binaries


def run(dataset_size: str, regenerate_data: bool, create_new_version: bool) -> None:
    genomes = load_genomes(dataset_size, regenerate_data)
    genes = load_genes(dataset_size, regenerate_data)

    train_csv_file_path: str = generate_train_csv(genes, genomes, create_new_version)

    X_train, X_val, X_test, y_train, y_val, y_test = train_test_validation_split(
        train_csv_file_path
    )

    X_train_vectorized, X_val_vectorized, X_test_vectorized, vectorizer = (
        extract_kmer_features(X_train, X_val, X_test)
    )

    trained_model = train(X_train_vectorized, y_train, X_val_vectorized, y_val)

    save_binaries(trained_model, vectorizer, dataset_size, create_new_version)

    evaluate_xgb_model(trained_model, X_test_vectorized, y_test)
