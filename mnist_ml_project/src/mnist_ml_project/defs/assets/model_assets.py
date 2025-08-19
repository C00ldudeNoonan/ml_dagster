import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import dagster as dg
from typing import Dict, Any, Optional
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import pickle
import os
from pathlib import Path


def load_saved_model(model_path: str) -> Dict[str, Any]:
    """
    Load a saved model from pickle file.

    Args:
        model_path: Path to the saved model pickle file

    Returns:
        Dictionary containing model, config, and metadata
    """
    with open(model_path, "rb") as f:
        model_data = pickle.load(f)

    return model_data


def list_saved_models(model_dir: str = "models") -> list:
    """
    List all saved model files in the models directory.

    Args:
        model_dir: Directory containing saved models

    Returns:
        List of model file paths sorted by modification time (newest first)
    """
    model_dir = Path(model_dir)
    if not model_dir.exists():
        return []

    model_files = list(model_dir.glob("*.pkl"))
    # Sort by modification time, newest first
    model_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

    return [str(f) for f in model_files]


class ModelConfig(dg.Config):
    """Configuration for model architecture and training."""

    # Architecture parameters
    conv1_channels: int = 32  # Reduced complexity
    conv2_channels: int = 64
    conv3_channels: int = 128
    dropout1_rate: float = 0.1  # Reduced dropout
    dropout2_rate: float = 0.2
    hidden_size: int = 256
    use_batch_norm: bool = True

    # Training parameters
    batch_size: int = 64  # Smaller batch size for better generalization
    learning_rate: float = 0.001  # Reduced learning rate
    epochs: int = 30  # Increased epochs
    optimizer_type: str = "adam"  # Changed to Adam
    momentum: float = 0.9
    weight_decay: float = 1e-5  # Reduced weight decay

    # Learning rate scheduling
    use_lr_scheduler: bool = True
    lr_step_size: int = 10
    lr_gamma: float = 0.1

    # Early stopping
    use_early_stopping: bool = True
    patience: int = 7
    min_delta: float = 0.001

    # Data augmentation - Research proven techniques
    use_data_augmentation: bool = True
    rotation_degrees: float = 15.0  # Increased from 10
    translation_pixels: float = 0.1  # New parameter
    scale_range_min: float = 0.9  # Split tuple into two floats
    scale_range_max: float = 1.1  # Split tuple into two floats

    # Model saving
    save_model: bool = True
    model_save_dir: str = "models"
    model_name_prefix: str = "mnist_cnn"


class DigitCNN(nn.Module):
    """Improved CNN for MNIST digit classification based on research."""

    def __init__(self, config: ModelConfig = None):
        super(DigitCNN, self).__init__()
        if config is None:
            config = ModelConfig()

        self.config = config

        # First convolutional block
        self.conv1 = nn.Conv2d(
            1, config.conv1_channels, kernel_size=5, padding=2
        )  # 5x5 kernel, maintain size
        self.bn1 = (
            nn.BatchNorm2d(config.conv1_channels)
            if config.use_batch_norm
            else nn.Identity()
        )
        self.pool1 = nn.MaxPool2d(2, 2)  # 28x28 -> 14x14

        # Second convolutional block
        self.conv2 = nn.Conv2d(
            config.conv1_channels, config.conv2_channels, kernel_size=5, padding=2
        )
        self.bn2 = (
            nn.BatchNorm2d(config.conv2_channels)
            if config.use_batch_norm
            else nn.Identity()
        )
        self.pool2 = nn.MaxPool2d(2, 2)  # 14x14 -> 7x7

        # Third convolutional block (new)
        self.conv3 = nn.Conv2d(
            config.conv2_channels, config.conv3_channels, kernel_size=3, padding=1
        )
        self.bn3 = (
            nn.BatchNorm2d(config.conv3_channels)
            if config.use_batch_norm
            else nn.Identity()
        )
        self.pool3 = nn.AdaptiveAvgPool2d((3, 3))  # Adaptive pooling to 3x3

        # Dropout layers
        self.dropout1 = nn.Dropout2d(config.dropout1_rate)
        self.dropout2 = nn.Dropout(config.dropout2_rate)

        # Calculate the flattened size: 3x3 * conv3_channels
        conv_output_size = 3 * 3 * config.conv3_channels

        # Fully connected layers
        self.fc1 = nn.Linear(conv_output_size, config.hidden_size)
        self.fc2 = nn.Linear(
            config.hidden_size, config.hidden_size // 2
        )  # Additional FC layer
        self.fc3 = nn.Linear(config.hidden_size // 2, 10)

    def forward(self, x):
        # First conv block
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)

        # Second conv block
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)
        x = self.dropout1(x)  # Spatial dropout

        # Third conv block
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool3(x)

        # Flatten and fully connected layers
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)  # Regular dropout
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)

        return x  # Return raw logits for CrossEntropyLoss


def train_model(context, model, train_loader, val_loader, config: ModelConfig):
    """Train the digit classification model with configurable parameters."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    context.log.info(f"Training on device: {device}")
    model.to(device)

    # Configure optimizer based on config with improved parameters
    if config.optimizer_type.lower() == "adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        context.log.info(
            f"Using Adam optimizer with lr={config.learning_rate}, weight_decay={config.weight_decay}"
        )
    elif config.optimizer_type.lower() == "sgd":
        optimizer = optim.SGD(
            model.parameters(),
            lr=config.learning_rate,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
        )
        context.log.info(
            f"Using SGD optimizer with lr={config.learning_rate}, momentum={config.momentum}"
        )
    else:
        optimizer = optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        context.log.info(f"Defaulting to Adam optimizer with lr={config.learning_rate}")

    criterion = nn.CrossEntropyLoss()

    # Add learning rate scheduler
    scheduler = None
    if config.use_lr_scheduler:
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=config.lr_step_size, gamma=config.lr_gamma
        )
        context.log.info(
            f"Using StepLR scheduler with step_size={config.lr_step_size}, gamma={config.lr_gamma}"
        )

    train_losses = []
    val_accuracies = []

    # Early stopping variables
    best_val_accuracy = 0.0
    patience_counter = 0

    # Log training configuration
    context.log.info(f"Starting training with:")
    context.log.info(f"- Batch size: {config.batch_size}")
    context.log.info(f"- Max epochs: {config.epochs}")
    context.log.info(f"- Early stopping patience: {config.patience}")
    context.log.info(f"- Model architecture:\n{str(model)}")

    for epoch in range(config.epochs):
        # Training phase
        model.train()
        train_loss = 0
        batch_count = 0
        correct_train = 0
        total_train = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            batch_count += 1

            # Calculate training accuracy
            _, predicted = torch.max(output.data, 1)
            total_train += target.size(0)
            correct_train += (predicted == target).sum().item()

            # Log progress every 100 batches
            if (batch_idx + 1) % 100 == 0:
                context.log.debug(
                    f"Epoch {epoch + 1}/{config.epochs} "
                    f"[{batch_idx + 1}/{len(train_loader)}] "
                    f"Loss: {loss.item():.4f}"
                )

        avg_train_loss = train_loss / batch_count
        train_accuracy = 100 * correct_train / total_train
        train_losses.append(avg_train_loss)

        # Validation phase
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0

        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                outputs = model(data)
                val_loss += criterion(outputs, target).item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += target.size(0)
                val_correct += (predicted == target).sum().item()

        val_accuracy = 100 * val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)
        val_accuracies.append(val_accuracy)

        # Log epoch results
        context.log.info(
            f"Epoch {epoch + 1}/{config.epochs} - "
            f"Train Loss: {avg_train_loss:.4f} - "
            f"Train Acc: {train_accuracy:.2f}% - "
            f"Val Loss: {avg_val_loss:.4f} - "
            f"Val Acc: {val_accuracy:.2f}% - "
            f"LR: {optimizer.param_groups[0]['lr']:.6f}"
        )

        # Early stopping logic
        if config.use_early_stopping:
            if val_accuracy > best_val_accuracy + config.min_delta:
                best_val_accuracy = val_accuracy
                patience_counter = 0
                context.log.info(
                    f"New best validation accuracy: {best_val_accuracy:.2f}%"
                )
            else:
                patience_counter += 1
                context.log.info(
                    f"Validation accuracy did not improve. "
                    f"Best: {best_val_accuracy:.2f}% "
                    f"Current: {val_accuracy:.2f}% "
                    f"Patience: {patience_counter}/{config.patience}"
                )

            if patience_counter >= config.patience:
                context.log.warning(
                    f"Early stopping triggered after {epoch + 1} epochs. "
                    f"Best validation accuracy: {best_val_accuracy:.2f}%"
                )
                break

        # Step the learning rate scheduler
        if scheduler is not None:
            scheduler.step()
            context.log.debug(
                f"Learning rate adjusted to: {optimizer.param_groups[0]['lr']:.6f}"
            )

    # Final training summary
    context.log.info("Training completed:")
    context.log.info(f"- Best validation accuracy: {best_val_accuracy:.2f}%")
    context.log.info(f"- Final learning rate: {optimizer.param_groups[0]['lr']:.6f}")
    context.log.info(f"- Total epochs run: {epoch + 1}")

    return model, train_losses, val_accuracies


@dg.asset(
    description="Train CNN digit classifier with configurable parameters",
    compute_kind="model_training",
    group_name="model_pipeline",
    required_resource_keys={"model_storage"},
)
def digit_classifier(
    context,
    processed_mnist_data: Dict[str, torch.Tensor],
    config: ModelConfig,
) -> DigitCNN:
    """Train a CNN to classify handwritten digits 0-9 with flexible configuration."""

    context.log.info(f"Training with config: {config.model_dump()}")

    train_data = processed_mnist_data["train_data"]
    val_data = processed_mnist_data["val_data"]
    train_labels = processed_mnist_data["train_labels"]
    val_labels = processed_mnist_data["val_labels"]

    # Create data loaders with configurable batch size
    train_dataset = TensorDataset(train_data, train_labels)
    val_dataset = TensorDataset(val_data, val_labels)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    # Initialize model with configuration
    model = DigitCNN(config)

    # Train the model - pass context to train_model
    trained_model, train_losses, val_accuracies = train_model(
        context, model, train_loader, val_loader, config
    )

    final_val_accuracy = val_accuracies[-1]

    # Add metadata
    context.add_output_metadata(
        {
            "final_val_accuracy": final_val_accuracy,
            "training_epochs": len(train_losses),
            "configured_epochs": config.epochs,
            "model_parameters": sum(p.numel() for p in trained_model.parameters()),
            "final_train_loss": train_losses[-1],
            "learning_rate": config.learning_rate,
            "batch_size": config.batch_size,
            "optimizer": config.optimizer_type,
            "early_stopping_used": config.use_early_stopping,
        }
    )

    context.log.info(
        f"Model training completed. Final validation accuracy: {final_val_accuracy:.2f}%"
    )

    # Save model as pickle file if requested
    if config.save_model:
        # Create models directory if it doesn't exist
        model_dir = Path(config.model_save_dir)
        model_dir.mkdir(exist_ok=True)

        # Create filename with timestamp and accuracy
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        accuracy_str = f"{final_val_accuracy:.2f}".replace(".", "p")
        filename = f"{config.model_name_prefix}_{timestamp}_acc{accuracy_str}.pkl"
        model_path = model_dir / filename

        # Save the trained model
        context.log.info(f"Saving model as {filename}")
        model_store = context.resources.model_storage
        model_store.save_model(trained_model, filename)

        context.add_output_metadata(
            {"model_name": filename, "final_accuracy": final_val_accuracy},
            output_name="result",
        )

    return trained_model


class ModelEvaluationConfig(dg.Config):
    """Configuration for model evaluation."""

    batch_size: int = 64
    accuracy_threshold: float = 0.90


@dg.asset(
    description="Evaluate model performance on test set",
    compute_kind="model_evaluation",
    group_name="model_pipeline",
    required_resource_keys={"model_storage"},
)
def model_evaluation(
    context,
    raw_mnist_data: Dict[str, Any],
    config: ModelEvaluationConfig,
) -> Dict[str, Any]:
    """Evaluate the trained model on the test set."""

    # Get the model store resource
    model_store = context.resources.model_storage

    # Get the latest trained model
    try:
        saved_models = list_saved_models(model_store.models_path)
        if not saved_models:
            context.log.error("No saved models found for evaluation")
            return {
                "test_accuracy": 0.0,
                "predictions": [],
                "labels": [],
                "classification_report": {},
            }

        latest_model_path = saved_models[0]  # First one is newest due to sorting
        context.log.info(f"Loading model for evaluation from {latest_model_path}")

        # Load the model directly since it's the full model object
        with open(latest_model_path, "rb") as f:
            model_to_evaluate = pickle.load(f)

        # Log model metadata
        context.log.info(f"Model loaded successfully")
        context.log.info(f"Model architecture:\n{str(model_to_evaluate)}")

    except Exception as e:
        context.log.error(f"Failed to load model for evaluation: {str(e)}")
        context.log.error(f"Exception details: {str(e.__class__.__name__)}")
        import traceback

        context.log.error(f"Traceback: {traceback.format_exc()}")
        return {
            "test_accuracy": 0.0,
            "predictions": [],
            "labels": [],
            "classification_report": {},
        }

    test_data = raw_mnist_data["test_data"]
    test_labels = raw_mnist_data["test_labels"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_to_evaluate.to(device)
    model_to_evaluate.eval()

    # Make predictions
    all_predictions = []
    all_labels = []

    test_dataset = TensorDataset(test_data, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model_to_evaluate(data)
            _, predicted = torch.max(outputs.data, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(target.cpu().numpy())

    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)

    # Calculate metrics
    test_accuracy = float(np.mean(all_predictions == all_labels))

    # Create confusion matrix plot
    cm = confusion_matrix(all_labels, all_predictions)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title("Confusion Matrix - Test Set")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    # Generate classification report
    class_report = classification_report(all_labels, all_predictions, output_dict=True)

    context.add_output_metadata(
        {
            "test_accuracy": test_accuracy,
            "test_samples": len(all_labels),
            "precision_macro": float(class_report["macro avg"]["precision"]),
            "recall_macro": float(class_report["macro avg"]["recall"]),
            "f1_macro": float(class_report["macro avg"]["f1-score"]),
            "evaluated_model_path": latest_model_path,
        },
        output_name="result",
    )

    context.log.info(f"Model evaluation completed. Test accuracy: {test_accuracy:.4f}")

    plt.close(fig)
    return {
        "test_accuracy": test_accuracy,
        "predictions": all_predictions.tolist(),
        "labels": all_labels.tolist(),
        "classification_report": class_report,
        "model_info": {"path": latest_model_path},
    }


class DeploymentConfig(dg.Config):
    """Configuration for model deployment."""

    accuracy_threshold: float = 0.90


@dg.asset(
    description="Deploy model to production if it meets quality threshold",
    compute_kind="model_deployment",
    group_name="model_pipeline",
    required_resource_keys={"model_storage"},
)
def production_digit_classifier(
    context,
    digit_classifier: DigitCNN,
    model_evaluation: Dict[str, Any],
    config: DeploymentConfig,
) -> Optional[DigitCNN]:
    """Deploy model to production only if it meets quality threshold."""

    test_accuracy = model_evaluation["test_accuracy"]

    context.log.info(
        f"Candidate model accuracy: {test_accuracy:.4f}, Threshold: {config.accuracy_threshold}"
    )

    if test_accuracy >= config.accuracy_threshold:
        context.log.info("Model meets quality threshold - deploying to production")

        # Get the model store resource
        model_store = context.resources.model_storage

        # Create model name with accuracy
        accuracy_str = f"{test_accuracy:.2f}".replace(".", "p")
        model_name = f"production_model_{accuracy_str}"

        # Save the model using the model store
        context.log.info(f"Saving production model as {model_name}")
        model_store.save_model(digit_classifier, model_name)

        context.add_output_metadata(
            {
                "deployment_status": "deployed",
                "deployed_accuracy": test_accuracy,
                "deployment_threshold": config.accuracy_threshold,
                "model_name": model_name,
            },
            output_name="result",
        )

        return digit_classifier
    else:
        context.log.warning(
            f"Model accuracy {test_accuracy:.4f} below threshold {config.accuracy_threshold} - skipping deployment"
        )
        context.add_output_metadata(
            {
                "deployment_status": "skipped",
                "candidate_accuracy": test_accuracy,
                "deployment_threshold": config.accuracy_threshold,
            },
            output_name="result",
        )
        return None
