import argparse
from datetime import datetime
from typing import List

from torch.utils.data import Dataset

from continual_learning.evaluation import plot_test_accuracies
from continual_learning.ewc.ewc_loss import EWCLoss
from continual_learning.ewc.ewc_models import EWCMultiHeadNN
from continual_learning.load_data import (
    get_task_dataloaders,
    load_mnist_datasets,
    load_cifar10_datasets,
    load_cifar100_datasets,
)
from continual_learning.utils import set_random_seeds


def run_ewc_multi_head(
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
    patience: int = 5,
) -> List[float]:

    output_size = int(n_classes / len(tasks))
    n_tasks = len(tasks)
    model = EWCMultiHeadNN(
        input_size=input_size, output_size=output_size, n_tasks=n_tasks
    )

    ewc_objects = []

    data_loaders = [
        get_task_dataloaders(
            train_dataset,
            val_dataset,
            test_dataset,
            batch_size=64,
            tasks=task,
            model_with_single_head_model=False,
        )
        for task in tasks
    ]

    avg_test_accs = []
    for task_id, task in enumerate(tasks):
        print(f"Training on task: {task_id + 1} - Digits: {task}")
        train_dl, val_dl, _ = data_loaders[task_id]

        model = model.train_model(
            train_dl,
            val_dl,
            ewc_objects,
            lambda_ewc,
            n_epochs,
            task_id,
            lr,
            weight_decay,
            patience,
        )

        if task_id < len(tasks) - 1:
            new_ewc = EWCLoss(model, train_dl, task_id)
            ewc_objects.append(new_ewc)

        test_accs = []
        for test_task_id in range(task_id + 1):
            _, _, test_dl = data_loaders[test_task_id]
            test_accs.append(model.test_model(test_dl, test_task_id))

        test_acc = sum(test_accs) / len(test_accs)
        avg_test_accs.append(test_acc)
        print(f"For task {task_id}: test_acc = {test_acc}")

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
    n_epochs = 20
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

            model_name = f"EWC Multihead {ds} lr={lr} wd={weight_decay}"
            accs = run_ewc_multi_head(
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
        title=f"Hyperparameter search: EWC Multihead models on the {ds} test set",
        save_to=f"EWC_Multihead_on_{ds}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png",
    )
