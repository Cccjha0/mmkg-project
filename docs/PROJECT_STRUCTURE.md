# Project Structure

## Top Level

- `ml/`: machine learning code and artifacts
- `backend/`: API services
- `frontend/`: web application
- `data/`: datasets and reusable caches
- `docs/`: project documentation

## ML Layout

- `ml/training/src/`: Python packages for data loading, models, training, evaluation, and utilities
- `ml/training/scripts/`: runnable scripts for training, cache building, debugging, and plotting
- `ml/configs/`: experiment YAML files
- `ml/inference/`: inference wrappers and exported runtime logic
- `ml/artifacts/`: checkpoints, metrics, and exported models

## Data Layout

- `data/datasets/<dataset>/raw/`: original dataset files
- `data/datasets/<dataset>/processed/`: generated dataset derivatives
- `data/cache/<dataset>/`: embedding caches and metadata
