# GenePrediction

## 📄 Academic Paper

This project is accompanied by a comprehensive [academic paper](paper/out/paper.pdf) documenting our methodology and findings. The paper presents our approach to prokaryotic gene prediction using k-mer features and XGBoost, achieving 97.77% accuracy. We encourage researchers and practitioners to explore the paper for detailed insights into our methodology, experiments, and results.

## Overview

GenePrediction is a bioinformatics pipeline designed to process prokaryotic genomic data, extract meaningful insights, and train machine learning models for gene prediction. The project combines efficient data processing with advanced machine learning techniques to provide a high-performance solution for gene prediction tasks.

## Key Features

- **High-Performance Gene Prediction**: Achieves 97.77% accuracy using XGBoost with k-mer features
- **Data Pipeline**: Complete workflow from data acquisition to evaluation with taxonomic diversity enforcement
- **Performance Optimization**: C language extensions for k-mer generation (100x faster than Python)
- **Advanced ML Integration**: Support for both gradient boosting and transformer-based approaches (DNABERT)
- **Reproducibility**: Integrated with DVC for experiment tracking and reproducible research
- **Comprehensive Evaluation**: Detailed performance metrics and visualizations

## Project Organization

```
GenePrediction/
├── data/                   # Data at various processing stages
│   ├── raw/                # Raw genomic and gene data
│   ├── processed/          # Processed datasets
│   └── models/             # Trained models and vectorizers
├── src/                    # Source code organized by function
│   ├── c_lang/             # C extensions for performance optimization
│   ├── data_loading/       # Data downloading from NCBI
│   ├── data_preprocessing/ # Data cleaning and preparation
│   ├── model_training/     # Model training implementations
│   └── evaluation/         # Model evaluation and visualization
├── tests/                  # Unit tests
├── results/                # Evaluation results and plots
├── paper/                  # Academic paper with methodology and results
├── dvc.yaml                # Pipeline definition
└── params.yaml             # Configurable parameters
```

## Quick Start

### Installation

```bash
git clone <repository-url>
cd GenePrediction
poetry install
```

### Running the Pipeline

```bash
# Run the complete pipeline
dvc repro

# Run a specific stage
dvc repro <stage-name>
```

### Pipeline Stages

The pipeline consists of 10 stages that transform raw genomic data into trained models and evaluation results:

1. Build C extensions  
2. Fetch sequence IDs  
3. Download sequences  
4. Convert to CSV  
5. Generate non-gene sequences  
6. Clean data  
7. Split datasets  
8. Vectorize sequences  
9. Train XGBoost model  
10. Evaluate and visualize results

## Configuration & Customization

Edit `params.yaml` to customize:
- Dataset size and k-mer length
- Model hyperparameters
- Taxonomic diversity settings
- Train-test split ratios

## Results

Our XGBoost model achieves **97.77% accuracy** on the test dataset, with balanced precision (98.56% for genes) and recall (97.06% for genes) metrics. Full evaluation results are available in the `results/` directory and detailed in the [academic paper](paper/out/paper.pdf).

## Testing

```bash
# Run tests
poetry run pytest
```

## Key Dependencies

Python 3.13+, Poetry, DVC, BioPython, XGBoost, Scikit-learn

## Future Directions

As outlined in our [paper](paper/out/paper.pdf), we plan to extend this work by:

1. Fine-tuning DNABERT models for prokaryotic gene prediction
2. Adding support for protein translation and functional annotation
3. Extending to eukaryotic gene prediction with splicing site detection
4. Optimizing performance through multi-threaded C extensions
5. Developing a web interface for broader accessibility

