from src.gene_prediction.xgboost_gene_predictor import predict as predict_xgb
from src.gene_prediction.xgboost_gene_predictor import train as train_xgb


def run():

    # genomes = load_genomes(load_from_pickle=True, load_from_files=True)
    # genes = load_genes(load_from_pickle=True, load_from_files=True)

    # generate_train_csv(genes, genomes)

    trained_model, X_test, y_test = train_xgb()
    predict_xgb(trained_model, X_test, y_test)
