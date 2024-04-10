from pathlib import Path
from typing import Dict, List, Optional

from matplotlib import pyplot as plt

PLOTS_DIR = Path(__file__).parent.parent.parent / "plots"


def get_color(model_name: str) -> str:
    if "EWC" in model_name.upper():
        return "#000080"  # navy blue
    elif "VCL KL" in model_name.upper():
        return "#FF1493"  # pink
    elif "VCL JS" in model_name.upper():
        return "#FFD300"  # yellow
    else:
        raise ValueError("Invalid model: Must be one of: 'EWC', 'VCL KL', 'VCL JS'.")


def get_linestyle(model_name: str) -> str:
    if "single" in model_name.lower():
        return "dashed"
    elif "multi" in model_name.lower():
        return "solid"
    else:
        raise ValueError(
            "Invalid model name: Must contain 'single' or 'multi' corresponding to the number of heads."
        )


def plot_test_accuracies(
    model2accuracies: Dict[str, List[float]],
    title: str,
    save_to: Optional[str] = None,
    random_colors: bool = True,
) -> None:
    """
    Plots the test accuracies of different models.
    """
    plt.figure(figsize=(10, 6))
    n_models = len(model2accuracies)

    for i, (model_name, accuracies) in enumerate(model2accuracies.items()):
        tasks = list(range(1, len(accuracies) + 1))
        if random_colors:
            plt.plot(
                tasks,
                accuracies,
                label=model_name,
                marker="x",
            )
        else:
            color = get_color(model_name)
            line_style = get_linestyle(model_name)
            linewidth = 1.5 + (n_models - i) * 0.6
            plt.plot(
                tasks,
                accuracies,
                label=model_name,
                marker="x",
                color=color,
                linestyle=line_style,
                linewidth=linewidth,
            )

    plt.title(title)
    plt.xlabel("Task")
    plt.ylabel("Test Accuracy")
    plt.ylim(0, 1.04)

    max_tasks = max(len(accuracies) for accuracies in model2accuracies.values())
    plt.xticks(range(1, max_tasks + 1))

    plt.legend()
    plt.grid(True)

    if save_to is not None:
        plt.savefig(PLOTS_DIR / save_to)

    plt.show()
