# MNIST Digit Classification with Dagster

A complete MLOps pipeline for MNIST digit classification using Dagster's Software-Defined Assets.

## Features

- **Asset-centric ML Pipeline**: Data preprocessing, model training, evaluation, and deployment as connected assets
- **Configurable Model Architecture**: Customizable CNN with batch normalization, dropout, and learning rate scheduling
- **Rich Metadata**: Performance metrics, confusion matrices, and training curves built into the UI
- **Dev-Prod Parity**: Same code runs locally and in production with different resource configurations
- **Model Storage**: Local and S3-based model persistence with versioning
- **Conditional Deployment**: Only deploy models that meet quality thresholds
- **Batch and Real-time Inference**: Support for both batch processing and real-time predictions

## Quick Start

1. **Install dependencies**:
   ```bash
   pip install -e .
   ```

2. **Start the Dagster UI**:
   ```bash
   dagster dev
   ```

3. **Run the full pipeline**:
   - Navigate to http://localhost:3000
   - Go to the "Assets" tab
   - Click "Materialize all" to run the complete ML pipeline

## Project Structure

```
mnist_ml_project/
├── src/mnist_ml_project/
│   ├── defs/
│   │   ├── assets/
│   │   │   ├── data_assets.py    # Data loading and preprocessing
│   │   │   ├── model_assets.py   # Model training and evaluation
│   │   │   └── prediction_assets.py # Inference assets
│   │   ├── resources.py          # Storage and compute resources
│   │   ├── jobs.py              # Pipeline job definitions
│   │   └── schedules.py         # Automation schedules
│   └── definitions.py           # Resource configuration
├── data/                        # Data storage (gitignored)
└── models/                      # Model storage (gitignored)
```

## Configuration

The pipeline uses configurable resources for model storage and compute:

```python
# Local development
resources = {
    "model_storage": LocalModelStoreResource(models_path="./models"),
    "compute": ComputeResource(device="cpu", batch_size=32)
}

# Production (AWS)
resources = {
    "model_storage": S3ModelStoreResource(bucket_name="my-models"),
    "compute": ComputeResource(
        compute_env="sagemaker",
        instance_type="ml.p3.2xlarge"
    )
}
```

## Key Assets

- `raw_mnist_data` - Downloads and normalizes MNIST dataset
- `processed_mnist_data` - Creates train/validation split
- `digit_classifier` - Trains CNN model with configurable architecture
- `model_evaluation` - Evaluates model performance on test set
- `production_digit_classifier` - Deploys models that meet accuracy threshold
- `batch_digit_predictions` - Batch inference for multiple images
- `digit_predictions` - Real-time inference endpoint

## Model Architecture

The CNN architecture includes:
- Three convolutional layers with batch normalization
- Adaptive pooling for flexible input sizes
- Configurable dropout rates
- Multiple fully connected layers
- Support for various optimizers (Adam, SGD with momentum)
- Learning rate scheduling
- Early stopping

## Storage and Resources

- **Model Storage**: 
  - Local filesystem storage for development
  - S3-based storage for production
  - Full model state and metadata preservation
  
- **Compute Configuration**:
  - Local CPU/GPU compute
  - SageMaker integration for production
  - Configurable batch sizes and worker counts

## Development Notes

- Large files (models, datasets) are managed with Git LFS
- Model artifacts are saved with metadata for reproducibility
- Use the provided configs in `model_configs.py` for experimentation

