# iNaturalist Bird Species Identification

This project implements a deep learning model for bird species identification using the iNaturalist dataset. The model is based on EfficientNet with contextual data integration.

## Project Structure

```
├── main.py                 # Main script to run the model pipeline
├── environment.yaml        # Conda environment configuration
├── indices/                # Indices for subsetting full iNaturalist dataset
├── src/                    # Source code directory
│   ├── transform/          # Data preprocessing and transformation
│   ├── model/             # Model architecture and training code
│   └── utils/             # Utility functions
└── data/                   # Data directory
    ├── train_mini/        # Training data
    └── validation/        # Validation data
```

## Setup

1. Clone this repository:
```bash
git clone [repository-url]
cd inat-amls2-project
```

2. Create and activate the conda environment:
```bash
conda env create -f environment.yaml
conda activate inat
```

## Usage

The main script `main.py` handles the complete pipeline from data preprocessing to model training and evaluation:

1. Data preprocessing: Loads and processes the training and validation data
2. Model initialization: Creates an EfficientNet model with contextual data integration
3. Training: Executes the training pipeline
4. Evaluation: Tests the model on the test set

To run the pipeline:
```bash
python main.py
```

The script will output the final training and test accuracy metrics.

## Model Architecture

The project uses an EfficientNet-based architecture (`EfficientNet_ContextualData`) that:
- Supports 1486 bird species classification
- Integrates contextual data (3-dimensional)
- Allows for fine-tuning of all layers
- Runs on GPU if available, otherwise falls back to CPU

## Data

The model expects data in the following structure:
- Training images: `data/train_mini/2021_train_mini/`
- Training labels: `data/train_mini/train_mini.json`
- Validation images: `data/validation/2021_valid/`
- Validation labels: `data/validation/val.json`

## Results

The model outputs:
- Training accuracy
- Test accuracy