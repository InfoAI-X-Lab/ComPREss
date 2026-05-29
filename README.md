# Protein Sequence Classification: Core Implementation

## Related Tool: AutoMCluster

This project is accompanied by a small data-processing utility named **AutoMCluster**, located in the `AutoMCluster/` directory. AutoMCluster is used before model training to process genome sequence files and antiSMASH results. It supports raw `.fna.gz` cleaning, sequence deduplication, sequence-length analysis and filtering, NZ-to-GCF mapping, final genome selection, batch antiSMASH prediction, and BGC result summarization.

For detailed usage, configuration, and script descriptions, please refer to `AutoMCluster/README.md`.

This repository contains the core implementation of a protein sequence classification framework based on masked language model pretraining, ESM-2 feature extraction, an additional Transformer encoder, supervised contrastive learning, center loss, and ensemble evaluation.

The current version is intended to present the main structure and methodological components of the project. 

## Current Project Structure

```text
|-- README.md
|-- requirements.txt
|-- .gitignore
|-- __init__.py
|-- data.py
|-- data_augmentation.py
|-- evaluate.py
|-- losses.py
|-- train.py
|-- utils.py
`-- models/
    |-- __init__.py
    |-- pretrain_model.py
    `-- classifier_model.py
```

## Module Overview
### `models/pretrain_model.py`
Defines the MLM pretraining model.

### `models/classifier_model.py`
Defines the downstream protein classifier.

### `data_augmentation.py`
Implements sequence-level data augmentation.
The augmentation logic modifies selected amino acid positions in specific sequence regions and is used to increase the number of samples in underrepresented classes.

### `data.py`
Contains data preparation and PyTorch data loading utilities.
This module includes:
- stratified 5-fold data splitting;
- fold-level train/test data organization;
- class-wise augmentation;
- sequence windowing for long protein sequences;
- balanced batch sampling for contrastive learning.

### `losses.py`
Defines the loss functions used in the fine-tuning stage.

### `train.py`
Contains the 5-fold training routine.

### `evaluate.py`
Contains evaluation utilities.

### `utils.py`
Provides utility functions such as:
- random seed initialization;
- CUDA memory cleanup.

## Expected Data Format
### Labeled Data

The labeled dataset is expected to be a CSV file with at least two columns:
```text
sequence,label
MKT...,0
GAV...,1
```

### Unlabeled Data
The unlabeled pretraining dataset is expected to contain at least:
```text
sequence
MKT...
GAV...
```

## Dependencies
Main dependencies are listed in `requirements.txt`:
```text
torch
transformers
pandas
numpy
scikit-learn
tqdm
optuna
```

Install dependencies with:
```bash
pip install -r requirements.txt
```
For GPU training, install the PyTorch version that matches the local CUDA environment.

## Notes for Reviewers
The repository preserves the central components of the proposed method while excluding private data, trained weights, and environment-specific execution scripts. The code is intended to show the technical design of the method.
