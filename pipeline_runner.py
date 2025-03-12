from src.data_preprocessing.data_splitter import train_test_split
from src.gene_prediction.xgboost_gene_predictor import predict, train
from src.helpers.filesystem_helper import save_pkl


def run():

    # genomes = load_genomes(load_from_pickle=True, load_from_files=True)
    # genes = load_genes(load_from_pickle=False, load_from_files=False)

    # generate_train_csv(genes, genomes)

    X_train, X_test, y_train, y_test = train_test_split(
        "data/processed/training.csv", test_size=0.15
    )

    trained_model = train(X_train, y_train)

    save_pkl(trained_model, "data/models/xgb_model.pkl")

    predict(trained_model, X_test, y_test)
