import argparse
import json
from datetime import datetime

from continual_learning.evaluation import plot_test_accuracies
from continual_learning.ewc.run_ewc_multihead import (
    run_ewc_multi_head,
)
from continual_learning.ewc.run_ewc_singlehead import run_ewc_single_head
from continual_learning.load_data import (
    load_mnist_datasets,
    load_cifar10_datasets,
    load_cifar100_datasets,
    PROJECT_DIR,
)
from continual_learning.utils import set_random_seeds
from continual_learning.vcl.run_vcl_multihead import run_vcl_multi_head
from continual_learning.vcl.run_vcl_singlehead import run_vcl_single_head

if __name__ == "__main__":
    print("Starting experiments ...")
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

    # batch_size = 64
    n_epochs = 40
    tasks = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]
    n_classes = 10

    print(f"Loading {ds} dataset ...")
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

    with open(PROJECT_DIR / "hyper_params.json", "r") as f:
        params = json.load(f)[f"{ds}"]

    model_name = "EWC Singlehead"
    print(f"\nTraining {model_name} model ...")
    accs = run_ewc_single_head(
        train_ds,
        val_ds,
        test_ds,
        tasks,
        n_epochs,
        n_classes,
        input_size,
        lr=params[model_name]["lr"],
        weight_decay=params[model_name]["weight_decay"],
    )
    model2accs[model_name] = accs

    model_name = "EWC Multihead"
    print(f"\nTraining {model_name} model ...")
    accs = run_ewc_multi_head(
        train_ds,
        val_ds,
        test_ds,
        tasks,
        n_epochs,
        n_classes,
        input_size,
        lr=params[model_name]["lr"],
        weight_decay=params[model_name]["weight_decay"],
    )
    model2accs[model_name] = accs

    model_name = "VCL Singlehead"
    print(f"\nTraining {model_name} model ...")
    accs = run_vcl_single_head(
        train_ds,
        val_ds,
        test_ds,
        tasks,
        n_epochs,
        n_classes,
        input_size,
        lr=params[model_name]["lr"],
        weight_decay=params[model_name]["weight_decay"],
    )
    model2accs[model_name] = accs

    model_name = "VCL Multihead"
    print(f"\nTraining {model_name} model ...")
    accs = run_vcl_multi_head(
        train_ds,
        val_ds,
        test_ds,
        tasks,
        n_epochs,
        n_classes,
        input_size,
        lr=params[model_name]["lr"],
        weight_decay=params[model_name]["weight_decay"],
    )
    model2accs[model_name] = accs

    plot_test_accuracies(
        model2accs,
        title=f"Comparison of multi-head and single-head models on the {ds} test set",
        save_to=f"test_accs_on_{ds}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png",
        random_colors=False,
    )
