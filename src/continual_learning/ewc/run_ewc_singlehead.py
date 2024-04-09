import argparse
from datetime import datetime
from typing import List

from torch.utils.data import Dataset

from continual_learning.evaluation import plot_test_accuracies
from continual_learning.ewc.ewc_loss import EWCLoss
from continual_learning.ewc.ewc_models import EWCSingleHeadNN
from continual_learning.load_data import (
    get_task_dataloaders,
    load_cifar100_datasets,
    load_mnist_datasets,
    load_cifar10_datasets,
)
from continual_learning.utils import set_random_seeds


def run_ewc_single_head(
    train_dataset: Dataset,
    val_dataset: Dataset,
    test_dataset: Dataset,
    tasks: List[List[int]],
    n_epochs: int,
    n_classes: int,
    input_size: int,
    lr: float = 1e-03,
    weight_decay: float = 1e-05,
    lambda_ewc: float = 0.1,
) -> List[float]:
    model = EWCSingleHeadNN(input_size, n_classes=n_classes)
    ewc_objects = []

    data_loaders = [
        get_task_dataloaders(
            train_dataset,
            val_dataset,
            test_dataset,
            batch_size=64,
            tasks=task,
            model_with_single_head_model=True,
        )
        for task in tasks
    ]

    avg_test_accs = []
    for task_idx, task in enumerate(tasks):
        print(f"\nTraining on task: {task_idx + 1} - Digits: {task}")
        train_task_dl, val_task_dl, _ = data_loaders[task_idx]

        model = model.train_model(
            train_task_dl,
            val_task_dl,
            ewc_objects,
            lambda_ewc,
            n_epochs,
            None,
            lr,
            weight_decay,
        )

        if task_idx < len(tasks) - 1:
            new_ewc = EWCLoss(model, train_task_dl)
            ewc_objects.append(new_ewc)

        test_accs = []
        for _, _, test_dl in data_loaders[: task_idx + 1]:
            test_accs.append(model.test_model(test_dl))

        test_acc = sum(test_accs) / len(test_accs)
        avg_test_accs.append(test_acc)

    return avg_test_accs


if __name__ == "__main__":
    set_random_seeds(seed=42)

    parser = argparse.ArgumentParser(
        description="Train a model on a specified dataset."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["MNIST", "CIFAR10", "CIFAR100"],
        default="MNIST",
        help="Dataset to use (MNIST, CIFAR10, CIFAR100)",
    )

    args = parser.parse_args()
    ds = args.dataset
    print(f"dataset = {ds}")

    # batch_size = 64
    n_epochs = 40
    tasks = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]
    n_classes = 10

    if ds == "MNIST":
        train_ds, val_ds, test_ds = load_mnist_datasets()
        input_size = 784
    elif ds == "CIFAR10":
        train_ds, val_ds, test_ds = load_cifar10_datasets()
        input_size = 3072
    elif ds == "CIFAR100":
        train_ds, val_ds, test_ds = load_cifar100_datasets()
        input_size = 3072
        tasks = [list(range(i, i + 10)) for i in range(0, 100, 10)]
        n_classes = 100
    else:
        raise ValueError(f"Invalid dataset name: {ds}")

    model2accs = {}

    for lr in [1e-3, 1e-4, 1e-5]:
        for weight_decay in [0.0, 1e-5, 1e-4]:
            print(f"lr = {lr}")
            print(f"weight_decay = {weight_decay}")

            model_name = f"EWC Singlehead {ds} lr={lr} wd={weight_decay}"
            accs = run_ewc_single_head(
                train_ds,
                val_ds,
                test_ds,
                tasks,
                n_epochs,
                n_classes,
                input_size,
                lr=lr,
                weight_decay=weight_decay,
            )
            model2accs[model_name] = accs

    plot_test_accuracies(
        model2accs,
        title=f"Hyperparameter search: EWC Singlehead models on the {ds} test set",
        save_to=f"EWC_Singlehead_on_{ds}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png",
    )
