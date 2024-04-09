import argparse
from datetime import datetime
from typing import List, Literal

from torch.utils.data import Dataset

from continual_learning.evaluation import plot_test_accuracies
from continual_learning.load_data import (
    get_task_dataloaders,
    load_mnist_datasets,
    load_cifar10_datasets,
    load_cifar100_datasets,
)
from continual_learning.utils import set_random_seeds
from continual_learning.vcl.vcl_models import VCLMultiHeadNN


def run_vcl_multi_head(
    train_dataset: Dataset,
    val_dataset: Dataset,
    test_dataset: Dataset,
    tasks: List[List[int]],
    n_epochs: int,
    n_classes: int,
    input_size: int,
    lr: float = 1e-03,
    weight_decay: float = 1e-05,
    divergence: Literal["KL", "JS"] = "KL"
) -> List[float]:

    output_size = int(n_classes / len(tasks))

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

    model = VCLMultiHeadNN(
        input_size=input_size,
        hidden_sizes=(256, 256),
        output_size=output_size,
        n_tasks=len(tasks),
        divergence=divergence
    )

    avg_test_accs = []
    for task_idx in range(len(tasks)):
        train_dl, val_dl, _ = data_loaders[task_idx]
        model = model.train_model(
            train_dl, val_dl, n_epochs, task_idx, lr=lr, weight_decay=weight_decay
        )

        test_accs = []
        for test_task_id in range(task_idx + 1):
            _, _, test_dl = data_loaders[test_task_id]
            test_acc = model.test_model(test_dl, test_task_id)
            print(f"For task {test_task_id}: test_acc = {test_acc}")

            test_accs.append(test_acc)

        avg_test_acc = sum(test_accs) / len(test_accs)
        avg_test_accs.append(avg_test_acc)
        print(f"For task {task_idx}: avg_test_acc = {avg_test_acc}")
        model.update_priors()

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

    for div in ["JS", "KL"]:
        for lr in [1e-4, 1e-5]:
            for weight_decay in [0.0, 1e-5]:
                print(f"div = {div}")
                print(f"lr = {lr}")
                print(f"weight_decay = {weight_decay}")

                model_name = f"VCL {div} Multihead {ds} lr={lr} wd={weight_decay}"
                accs = run_vcl_multi_head(
                    train_ds,
                    val_ds,
                    test_ds,
                    tasks,
                    n_epochs,
                    n_classes,
                    input_size,
                    lr=lr,
                    weight_decay=weight_decay,
                    divergence=div
                )
                model2accs[model_name] = accs

    plot_test_accuracies(
        model2accs,
        title=f"Hyperparameter search: VCL {div} Multihead models on the {ds} test set",
        save_to=f"VCL_{div}_Multihead_on_{ds}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png",
    )
