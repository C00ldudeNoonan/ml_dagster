from torchvision import datasets, transforms
import torch
from typing import Dict, Any
import dagster as dg
from sklearn.model_selection import train_test_split


@dg.asset(
    description="Download and load raw MNIST dataset",
    compute_kind="data_ingestion",
    group_name="data_processing",
)
def raw_mnist_data(context) -> Dict[str, Any]:
    """Download the raw MNIST dataset."""

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),  # MNIST mean and std
        ]
    )

    # Download training data
    train_dataset = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )

    # Download test data
    test_dataset = datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )

    # Convert to tensors
    train_data = torch.stack([train_dataset[i][0] for i in range(len(train_dataset))])
    train_labels = torch.tensor(
        [train_dataset[i][1] for i in range(len(train_dataset))]
    )

    test_data = torch.stack([test_dataset[i][0] for i in range(len(test_dataset))])
    test_labels = torch.tensor([test_dataset[i][1] for i in range(len(test_dataset))])

    context.log.info(
        f"Loaded {len(train_data)} training samples and {len(test_data)} test samples"
    )

    return {
        "train_data": train_data,
        "train_labels": train_labels,
        "test_data": test_data,
        "test_labels": test_labels,
    }


@dg.asset(
    description="Preprocess MNIST images for training",
    compute_kind="data_preprocessing",
    group_name="data_processing",
)
def processed_mnist_data(
    context, raw_mnist_data: Dict[str, Any]
) -> Dict[str, torch.Tensor]:
    """Process MNIST data and create train/validation split."""

    train_data = raw_mnist_data["train_data"]
    train_labels = raw_mnist_data["train_labels"]
    test_data = raw_mnist_data["test_data"]
    test_labels = raw_mnist_data["test_labels"]

    # Create validation split from training data
    train_data, val_data, train_labels, val_labels = train_test_split(
        train_data, train_labels, test_size=0.2, random_state=42, stratify=train_labels
    )

    # Convert back to tensors
    train_data = torch.tensor(train_data)
    val_data = torch.tensor(val_data)
    train_labels = torch.tensor(train_labels)
    val_labels = torch.tensor(val_labels)

    context.add_output_metadata(
        {
            "train_samples": len(train_data),
            "val_samples": len(val_data),
            "test_samples": len(test_data),
            "image_shape": str(train_data.shape[1:]),
            "num_classes": len(torch.unique(train_labels)),
        }
    )

    context.log.info(
        f"Processed data - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}"
    )

    return {
        "train_data": train_data,
        "val_data": val_data,
        "train_labels": train_labels,
        "val_labels": val_labels,
        "test_data": test_data,
        "test_labels": test_labels,
    }
