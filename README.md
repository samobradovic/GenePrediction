# GenePrediction

## Overview

GenePrediction is a bioinformatics pipeline designed to process genomic data, extract meaningful insights, and train machine learning models for gene prediction. The project leverages tools like DVC for data versioning, Poetry for dependency management, and various Python libraries for data preprocessing, model training, and evaluation. Additionally, the project includes optimized C extensions for performance-critical tasks.

## Features

- **Data Downloading**: Fetches raw genomic and gene data using the Entrez API.
- **Data Preprocessing**:
  - FASTA-to-CSV conversion.
  - Data cleaning and filtering based on sequence length.
  - Splitting datasets into training, testing, and validation sets.
  - Generating non-gene sequences for balanced datasets.
- **Vectorization**: Converts DNA sequences into k-mer representations using `CountVectorizer` or a custom C extension for k-mer generation.
- **Model Training**: Trains an XGBoost classifier for gene prediction.
- **Evaluation**:
  - Generates performance metrics such as accuracy, precision, recall, and F1-score.
  - Visualizes results with confusion matrices, ROC curves, and precision-recall curves.
- **C Extensions**: Includes a custom C module for efficient k-mer generation, significantly improving performance for large datasets.

## Project Structure

```
GenePrediction/
├── data/
│   ├── raw/                # Raw genomic and gene data
│   ├── processed/          # Processed datasets (training, testing, validation)
│   └── models/             # Trained models and vectorizers
├── results/                # Evaluation results and plots
├── src/                    # Source code
│   ├── c_lang/             # C extensions for performance optimization
│   ├── data_loading/       # Scripts for data downloading
│   ├── data_preprocessing/ # Scripts for data cleaning and preparation
│   ├── model_training/     # Scripts for training models
│   ├── evaluation/         # Scripts for model evaluation
│   ├── exploratory_data_analasys/ # Scripts for exploratory data analysis
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
   cd GenePrediction
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
- Matplotlib
- Seaborn

## Results

Evaluation metrics and visualizations are saved in the `results/` directory, including:

- Confusion Matrix
- Precision-Recall Curve
- Receiver Operating Characteristic (ROC) Curve
- Classification Reports

## Testing

Run unit tests using pytest:

```bash
poetry run pytest
```

## Future Work

- Add support for additional machine learning models.
- Extend the pipeline to include more advanced feature extraction techniques.
- Optimize the C extension for multi-threaded execution.
- Integrate DNABERT for enhanced sequence classification.
