# GenePrediction

## Overview

GenePrediction is a bioinformatics pipeline designed to process prokaryotic genomic data, extract meaningful insights, and train machine learning models for gene prediction. The project leverages tools like DVC for data versioning, Poetry for dependency management, and various Python libraries for data preprocessing, model training, and evaluation. Additionally, the project includes optimized C extensions for performance-critical tasks and integrates transformer-based models like DNABERT for enhanced prediction capabilities.

## Features

- **Data Downloading**: Fetches raw genomic and gene data using the Entrez API with configurable taxonomic diversity.
- **Data Preprocessing**:
  - FASTA-to-CSV conversion.
  - Data cleaning and filtering based on sequence length.
  - Splitting datasets into training, testing, and validation sets.
  - Generating non-gene sequences for balanced datasets.
  - Sequence modification capabilities for data augmentation.
- **Vectorization**: Converts DNA sequences into k-mer representations using `CountVectorizer` or a custom C extension for k-mer generation.
- **Model Training**: 
  - XGBoost classifier for gene prediction with configurable hyperparameters.
  - DNABERT integration for transformer-based sequence analysis.
- **Evaluation**:
  - Generates performance metrics such as accuracy, precision, recall, and F1-score.
  - Visualizes results with confusion matrices, ROC curves, and precision-recall curves.
- **C Extensions**: Includes a custom C module for efficient k-mer generation, significantly improving performance for large datasets (benchmarked up to 100x faster than pure Python implementation).
- **Scientific Documentation**: Includes LaTeX paper template for documenting research findings.

## Project Structure

```
GenePrediction/
├── data/
│   ├── raw/                # Raw genomic and gene data
│   │   ├── genomes/        # Downloaded genome sequences
│   │   ├── genes/          # Downloaded gene sequences
│   │   ├── genome_ids/     # Entrez genome IDs
│   │   └── gene_ids/       # Entrez gene IDs
│   ├── processed/          # Processed datasets (training, testing, validation)
│   │   ├── *_vectorized.pkl # Vectorized datasets
│   │   └── *.csv           # CSV datasets
│   ├── models/             # Trained models and vectorizers
│   └── results/            # Evaluation results and plots
├── src/                    # Source code
│   ├── c_lang/             # C extensions for performance optimization
│   ├── data_loading/       # Scripts for data downloading
│   ├── data_preprocessing/ # Scripts for data cleaning and preparation
│   ├── model_training/     # Scripts for training models (XGBoost, DNABERT)
│   ├── evaluation/         # Scripts for model evaluation
│   ├── exploratory_data_analasys/ # Scripts for exploratory data analysis
│   └── helpers/            # Utility functions
├── tests/                  # Unit tests for all major components
├── paper/                  # Scientific documentation (LaTeX)
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

## Pipeline Stages

The pipeline is managed via DVC with the following stages:

1. **compile_c_modules**: Builds C extensions for performance optimization
2. **fetch_ids**: Retrieves Entrez IDs for genomes and genes with taxonomic diversity enforcement
3. **download_genomes**: Downloads genome and gene sequences from NCBI
4. **csv_conversion**: Converts FASTA files to CSV format
5. **non_gene_generation**: Generates non-gene sequences for balanced datasets
6. **data_cleanup**: Filters and cleans sequences based on length and quality parameters
7. **train_test_split**: Splits data into training, testing, and validation sets
8. **vectorization**: Converts sequences to k-mer feature vectors using either Python or C implementation
9. **xgb_training**: Trains the XGBoost model on vectorized training data
10. **evaluation**: Evaluates model performance and generates visualizations

To run a specific stage:

```bash
dvc repro <stage-name>
```

To view the dependency graph:

```bash
dvc dag
```

## Configuration

- Modify `params.yaml` to adjust pipeline parameters:
  - Dataset size (`small`, `big`, `max`)
  - K-mer size for DNA sequence vectorization
  - Gene length constraints (min/max)
  - Train-test split ratios
  - XGBoost hyperparameters
  - Taxonomic diversity settings for better biological representation
- Update `constants.yaml` for:
  - File paths and directory structure
  - Entrez API settings
  - Batch sizes for efficient data fetching
  - Maximum file sizes

## Dependencies

- Python 3.13+
- Poetry
- DVC
- BioPython (>=1.81)
- PyTorch (>=2.6.0)
- Transformers (>=4.49.0, for DNABERT)
- Scikit-learn (>=1.6.1)
- XGBoost (>=2.1.4)
- Pandas (>=2.2.3)
- Matplotlib (>=3.10.1)
- Einops (>=0.8.1)
- Tqdm (>=4.67.1)
- Dill (>=0.3.9)
- PyYAML (>=6.0.2)

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

## Scientific Documentation

The `paper/` directory contains a LaTeX template for documenting the research methodology and findings. This can be compiled into a PDF using any LaTeX compiler:

```bash
cd paper
pdflatex paper.tex
bibtex paper
pdflatex paper.tex
pdflatex paper.tex
```

## Future Work

- Enhance DNABERT integration with fine-tuning for prokaryotic gene prediction
- Implement additional sequence modification techniques for data augmentation
- Optimize the C extension for multi-threaded execution
- Add support for protein translation and functional annotation
- Implement CDS (Coding Sequence) finding capabilities
- Add support for eukaryotic gene prediction with splicing site detection
- Develop a web interface for easier interaction with the pipeline
- Expand taxonomic diversity controls for better model generalization
