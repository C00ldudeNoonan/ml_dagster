# MNIST Digit Classification with Dagster

A complete MLOps pipeline for MNIST digit classification using Dagster's Software-Defined Assets.

## Features

- **Asset-centric ML Pipeline**: Data preprocessing, model training, evaluation, and deployment as connected assets
- **Declarative Automation**: Intelligent retraining and batch processing schedules  
- **Rich Metadata**: Performance metrics, confusion matrices, and training curves built into the UI
- **Dev-Prod Parity**: Same code runs locally and in production with different resource configurations
- **MLflow Integration**: Experiment tracking and model versioning
- **Conditional Deployment**: Only deploy models that meet quality thresholds
- **Monitoring & Alerts**: Slack notifications for pipeline failures

## Quick Start

1. **Install dependencies**:
   ```bash
   pip install -e ".[dev]"
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

- `assets/data_assets.py` - Data loading and preprocessing
- `assets/model_assets.py` - Model training, evaluation, and deployment  
- `assets/prediction_assets.py` - Batch and real-time inference
- `resources/` - Configurable resources for different environments
- `jobs/ml_jobs.py` - Job definitions for different pipeline stages
- `sensors/ml_sensors.py` - Monitoring and alerting sensors

## Configuration

Switch between development and production by changing the resources in `__init__.py`:

```python
# For development (local, small dataset)
resources=dev_resources

# For production (S3, full dataset, GPU)  
resources=prod_resources
```

## Key Assets

- `raw_mnist_data` - Downloads MNIST dataset
- `processed_mnist_data` - Normalizes and splits data
- `digit_classifier` - Trains CNN model (eager retraining)
- `model_evaluation` - Evaluates on test set with confusion matrix
- `production_digit_classifier` - Conditional deployment based on accuracy
- `batch_digit_predictions` - Scheduled batch inference (2 AM daily)
- `digit_predictions` - Real-time inference endpoint

## Automation Conditions

- **Eager retraining**: `digit_classifier` retrains when new data arrives
- **Scheduled inference**: Batch predictions run daily at 2 AM
- **Conditional deployment**: Models only deploy if accuracy > 95%

## Monitoring

- Slack alerts for pipeline failures
- Rich metadata in Dagster UI (accuracy, confusion matrices, training curves)
- MLflow experiment tracking integration

## Extending the Pipeline

Add new assets for:
- Data augmentation techniques
- Different model architectures (ResNet, Transformer)
- A/B testing between models
- Model drift detection
- Custom evaluation metrics
"""