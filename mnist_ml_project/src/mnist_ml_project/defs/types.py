from typing import TypedDict, List, Dict, Any
import torch
import dagster as dg

class ModelData(TypedDict):
    model: torch.nn.Module
    config: dict
    accuracy: float
    timestamp: str
    model_architecture: str

class DataBatch(TypedDict):
    train_data: torch.Tensor
    train_labels: torch.Tensor
    val_data: torch.Tensor
    val_labels: torch.Tensor
    test_data: torch.Tensor
    test_labels: torch.Tensor

class EvaluationResult(TypedDict):
    test_accuracy: float
    predictions: List[int]
    labels: List[int]
    classification_report: Dict[str, Any]
    model_info: Dict[str, str]

class BatchPredictionConfig(dg.Config):
    """Configuration for batch prediction processing."""
    
    num_test_images: int = 100  # Number of test images to process
    batch_size: int = 32  # Batch size for processing
    device: str = "cpu"  # Device to run inference on
    confidence_threshold: float = 0.8  # Minimum confidence threshold

class RealTimePredictionConfig(dg.Config):
    """Configuration for real-time prediction processing."""
    
    batch_size: int = 1  # Single image processing
    device: str = "cpu"  # Device to run inference on
    confidence_threshold: float = 0.7  # Minimum confidence threshold
    max_response_time_ms: int = 100  # Maximum response time in milliseconds