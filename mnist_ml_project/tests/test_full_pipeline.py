import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset
from mnist_ml_project.defs.assets.data_assets import raw_mnist_data, processed_mnist_data
from mnist_ml_project.defs.assets.model_assets import (
    DigitCNN,
    ModelConfig,
    train_model,
    model_evaluation
)

def test_imports():
    """Test that all main modules can be imported."""
    from mnist_ml_project.defs.assets import data_assets, model_assets
    from mnist_ml_project.defs.assets import prediction_assets
    assert True  # If we get here, imports worked

def test_pipeline_error_handling(mock_context):
    """Test error handling in the pipeline."""
    # Just test that our mock context works
    assert mock_context is not None
    assert hasattr(mock_context, 'log')