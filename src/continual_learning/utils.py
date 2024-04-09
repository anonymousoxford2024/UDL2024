import random

import numpy as np
import torch


def set_random_seeds(seed: int = 42) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Ensuring CUDA operations are deterministic
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)  # for multi-GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
