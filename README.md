# IntroBioProject

## Overview

IntroBioProject is a bioinformatics pipeline designed to process genomic data, extract meaningful insights, and train machine learning models for gene prediction. The project leverages tools like DVC for data versioning, Poetry for dependency management, and various Python libraries for data preprocessing, model training, and evaluation.

## Features

- **Data Downloading**: Fetches raw genomic and gene data using the Entrez API.
- **Data Preprocessing**: Includes FASTA-to-CSV conversion, data cleaning, and splitting into training, testing, and validation sets.
- **Vectorization**: Converts DNA sequences into k-mer representations using `CountVectorizer`.
- **Model Training**: Trains an XGBoost classifier for gene prediction.
- **Evaluation**: Generates performance metrics and visualizations, including confusion matrices and ROC curves.

## Project Structure

```
IntroBioProject/
├── data/
│   ├── raw/                # Raw genomic and gene data
│   ├── processed/          # Processed datasets (training, testing, validation)
│   └── models/             # Trained models and vectorizers
├── results/                # Evaluation results and plots
├── src/                    # Source code
│   ├── data_loading/       # Scripts for data downloading
│   ├── data_preprocessing/ # Scripts for data cleaning and preparation
│   ├── model_training/     # Scripts for training models
│   ├── evaluation/         # Scripts for model evaluation
│   └── helpers/            # Utility functions
├── tests/                  # Unit tests
├── dvc.yaml                # DVC pipeline stages
├── params.yaml             # Pipeline parameters
├── constants.yaml          # Project constants
└── README.md               # Project documentation
```

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd IntroBioProject
   ```
2. Install dependencies using Poetry:
   ```bash
   poetry install
   ```
3. Set up pre-commit hooks:
   ```bash
   poetry run pre-commit install
   ```

## Usage

### Running the Pipeline

The pipeline is managed using DVC. To execute all stages:

```bash
dvc repro
```

## Configuration

- Modify `params.yaml` to adjust pipeline parameters (e.g., dataset size, k-mer size, train-test split ratios).
- Update `constants.yaml` for file paths and other constants.

## Dependencies

- Python 3.10+
- Poetry
- DVC
- BioPython
- Scikit-learn
- XGBoost
- Transformers (for DNABERT)

## Results

Evaluation metrics and visualizations are saved in the `results/` directory, including:

- Confusion Matrix
- Precision-Recall Curve
- Receiver Operating Characteristic (ROC) Curve

## Testing

Run unit tests using pytest:

```bash
poetry run pytest
```
