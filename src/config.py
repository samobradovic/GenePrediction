import os

MAX_GENOMES = 50000
MAX_GENOMES_SIZE_GB = 150
MAX_GENES = 50000000
MAX_GENES_SIZE_GB = 150

FILE_FORMAT = "fasta"

RAW_DATA_FOLDER_PATH = "data/raw"
GENOMES_RAW_DATA_FILE_PATH = os.path.join(RAW_DATA_FOLDER_PATH, "genomes")
GENES_RAW_DATA_FILE_PATH = os.path.join(RAW_DATA_FOLDER_PATH, "genes")
PROTEINS_RAW_DATA_FILE_PATH = os.path.join(RAW_DATA_FOLDER_PATH, "proteins")

CSV_FILE_PATH = "data/processed"
CSV_PROCESSED_FILE_PATH = os.path.join(CSV_FILE_PATH, "processed.csv")

ENTREZ_EMAIL = "Your.Name.Here@example.org"

KMER_SIZE = 3

MODEL_FOLDER_PATH = "data/models"
MODEL_FILE_PATH = os.path.join(MODEL_FOLDER_PATH, "xgb_model.pkl")
VECTORIZER_FILE_PATH = os.path.join(MODEL_FOLDER_PATH, "count_vectorizer.pkl")
