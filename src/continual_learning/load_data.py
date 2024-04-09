from pathlib import Path
from typing import List, Tuple

from torch.utils.data import DataLoader, Dataset, random_split, Subset
from torchvision import datasets, transforms

PROJECT_DIR = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_DIR / "data"
MODELS_DIR = PROJECT_DIR / "models"


def _load_data(
    dataset_name: str, norm_mean: tuple, norm_std: tuple, val_size: float = 0.1
) -> Tuple[Dataset, Dataset, Dataset]:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std),
        ]
    )

    if dataset_name == "CIFAR10":
        ds = datasets.CIFAR10
    elif dataset_name == "CIFAR100":
        ds = datasets.CIFAR100
    elif dataset_name == "MNIST":
        ds = datasets.MNIST

    full_train_dataset = ds(
        root=str(DATA_DIR / f"{dataset_name}"),
        train=True,
        download=True,
        transform=transform,
    )
    total_size = len(full_train_dataset)
    val_size = int(total_size * val_size)
    train_size = total_size - val_size

    train_dataset, val_dataset = random_split(
        full_train_dataset, [train_size, val_size]
    )

    test_dataset = Subset(
        ds(
            root=str(DATA_DIR / f"{dataset_name}"),
            train=False,
            download=True,
            transform=transform,
        ),
        list(range(10000)),
    )

    return train_dataset, val_dataset, test_dataset


def load_cifar10_datasets() -> Tuple[Dataset, Dataset, Dataset]:
    return _load_data("CIFAR10", (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))


def load_cifar100_datasets() -> Tuple[Dataset, Dataset, Dataset]:
    return _load_data("CIFAR100", (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))


def load_mnist_datasets() -> Tuple[Dataset, Dataset, Dataset]:
    return _load_data("MNIST", (0.1307,), (0.3081,))


def get_task_dataloaders(
    train_dataset: Dataset,
    val_dataset: Dataset,
    test_dataset: Dataset,
    batch_size: int,
    tasks: List[int],
    model_with_single_head_model: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:

    if model_with_single_head_model:
        train_task_dataset = [data for data in train_dataset if data[1] in tasks]
        val_task_dataset = [data for data in val_dataset if data[1] in tasks]
        test_task_dataset = [data for data in test_dataset if data[1] in tasks]
    else:
        train_task_dataset = [
            (img, tasks.index(label)) for img, label in train_dataset if label in tasks
        ]
        val_task_dataset = [
            (img, tasks.index(label)) for img, label in val_dataset if label in tasks
        ]
        test_task_dataset = [
            (img, tasks.index(label)) for img, label in test_dataset if label in tasks
        ]

    print(f"len(train_task_dataset) = {len(train_task_dataset)}")
    print(f"len(val_task_dataset)   = {len(val_task_dataset)}")
    print(f"len(test_task_dataset)  = {len(test_task_dataset)}")

    # Creating data loaders
    train_dl = DataLoader(
        dataset=train_task_dataset, batch_size=batch_size, shuffle=True
    )
    val_dl = DataLoader(dataset=val_task_dataset, batch_size=batch_size, shuffle=False)
    test_dl = DataLoader(
        dataset=test_task_dataset, batch_size=batch_size, shuffle=False
    )

    return train_dl, val_dl, test_dl
