from src.data_loading.gene_fetcher import load_genes
from src.data_loading.genome_fetcher import load_genomes
from src.data_preprocessing.csv_converter import generate_combined_csv
from src.data_preprocessing.data_preprocessing import (
    extract_kmer_features_for_sequence,
    fit_vectorizer,
    save_train_test_validation_csvs,
    train_test_validation_split,
)
from src.evaluation.evaluation import evaluate_xgb_model
from src.gene_prediction.xgboost_gene_predictor import train
from src.helpers.filesystem_helper import (
    get_test_dfs_from_csv,
    load_binaries,
    save_binaries,
)


def run(
    dataset_size: str,
    regenerate_data: bool,
    evaluate_only: bool,
) -> None:
    if not evaluate_only:
        genomes = load_genomes(dataset_size, regenerate_data)
        genes = load_genes(dataset_size, regenerate_data)

        train_csv_file_path: str = generate_combined_csv(genes, genomes)

        X_train, X_val, X_test, y_train, y_val, y_test = train_test_validation_split(
            train_csv_file_path
        )

        save_train_test_validation_csvs(
            {
                "X_train": X_train,
                "X_val": X_val,
                "X_test": X_test,
                "y_train": y_train,
                "y_val": y_val,
                "y_test": y_test,
            }
        )

        X_train_vectorized, vectorizer = fit_vectorizer(X_train)
        X_val_vectorized = extract_kmer_features_for_sequence(
            X_val, vectorizer, "Validation"
        )
        X_test_vectorized = extract_kmer_features_for_sequence(
            X_test, vectorizer, "Testing"
        )

        trained_model = train(X_train_vectorized, y_train, X_val_vectorized, y_val)

        save_binaries(trained_model, vectorizer, dataset_size)

    else:
        trained_model, vectorizer = load_binaries(dataset_size)
        X_test, y_test = get_test_dfs_from_csv()
        X_test_vectorized = extract_kmer_features_for_sequence(
            X_test, vectorizer, "Testing"
        )

    evaluate_xgb_model(trained_model, X_test_vectorized, y_test)
