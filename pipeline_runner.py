from src.data_loading.gene_fetcher import load_genes
from src.data_loading.genome_fetcher import load_genomes
from src.data_preprocessing.csv_converter import generate_train_csv
from src.data_preprocessing.data_preprocessing import train_test_validation_split
from src.gene_prediction.xgboost_gene_predictor import predict, train


def run():
    genomes = load_genomes(load_from_files=True, subset_size=-1)
    genes = load_genes(load_from_files=True, subset_size=-1)

    generate_train_csv(genes, genomes)

    X_train, X_val, X_test, y_train, y_val, y_test = train_test_validation_split(
        "data/processed/processed.csv"
    )

    trained_model = train(X_train, y_train, X_val, y_val)

    # save_pkl(trained_model, "data/models/xgb_model.pkl")

    predict(trained_model, X_test, y_test)
