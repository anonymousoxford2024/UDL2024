from typing import Optional, Union, Tuple, Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from continual_learning.vcl.kl_divergence import kl_divergence, js_divergence


def variable_normal(shape: Union[int, tuple]) -> nn.Parameter:
    return nn.Parameter(torch.Tensor(*shape).normal_(mean=0.0, std=0.1))


def variable_fill(shape: Union[int, tuple]) -> nn.Parameter:
    return nn.Parameter(torch.Tensor(*shape).fill_(-3.0))


def compute_acc(outputs: torch.Tensor, targets: torch.Tensor) -> np.ndarray:
    correct = outputs.argmax(dim=-1).cpu().numpy() == targets.cpu().numpy()
    acc = np.mean(correct)
    return acc


def _compute_stable_var(rho: nn.Parameter) -> torch.Tensor:
    return torch.log1p(torch.exp(rho))


class VariationalLinear(nn.Module):
    def __init__(
        self, in_feats: int, out_feats: int, divergence: Literal["KL", "JS"]
    ) -> None:
        super(VariationalLinear, self).__init__()
        self.divergence = divergence

        self.in_features = in_feats
        self.out_features = out_feats

        # set priors
        self.prior_weights_mu = torch.tensor(0)
        self.prior_weights_sigma = torch.tensor(0.01)

        # set weights
        self.weights_mu = variable_normal((out_feats, in_feats + 1))
        self.weights_rho = variable_fill((out_feats, in_feats + 1))
        self.weights_sigma = torch.Tensor()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Augment input with ones for bias integration
        ones = torch.ones(x.size(0), 1, device=x.device)
        x_augmented = torch.cat([x, ones], dim=1)

        self.weights_sigma = _compute_stable_var(self.weights_rho)

        # Compute activations using augmented weights and inputs
        mean_activation = F.linear(x_augmented, self.weights_mu)
        var_activation = F.linear(x_augmented**2, self.weights_sigma**2) + 1e-16
        std_activation = torch.sqrt(var_activation)

        eps = (
            torch.empty(mean_activation.size()).normal_(0, 1) if self.training else 0.0
        )
        return mean_activation + std_activation * eps

    def compute_divergence(self) -> torch.Tensor:
        if self.divergence == "KL":
            div = kl_divergence(
                self.prior_weights_mu,
                self.prior_weights_sigma,
                self.weights_mu,
                self.weights_sigma,
            )
        elif self.divergence == "JS":
            div = js_divergence(
                self.prior_weights_mu,
                self.prior_weights_sigma,
                self.weights_mu,
                self.weights_sigma,
            )
        return div


class EvidenceLowerBoundLoss(nn.Module):

    def __init__(self, model):
        super().__init__()
        self.num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    def forward(self, outputs, targets, kl):
        return (
            F.nll_loss(outputs, targets, reduction="mean")
            + 0.1 * kl / self.num_params
        )


class VCLBase(nn.Module):
    def __init__(self):
        super(VCLBase, self).__init__()

    def train_model(
        self,
        train_loader,
        val_loader,
        n_epochs: int = 10,
        task_id: Optional[int] = None,
        lr: float = 1e-03,
        weight_decay: float = 1e-05,
        patience: int = 5,
    ):
        self.train()
        optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        loss_fn = EvidenceLowerBoundLoss(self)

        highest_acc = 0.0
        patience_counter = 0
        best_model_state = self.state_dict()

        for _ in tqdm(range(n_epochs)):
            for data, target in train_loader:
                optimizer.zero_grad()
                if task_id is not None:
                    output = F.log_softmax(self(data, task_id), dim=-1)
                    kl = self.get_divergence(task_id)
                else:
                    output = F.log_softmax(self(data), dim=-1)
                    kl = self.get_divergence()
                loss = loss_fn(output, target, kl)
                loss.backward()
                optimizer.step()

            val_acc = self.test_model(val_loader, task_id)
            if val_acc > highest_acc:
                highest_acc = val_acc
                best_model_state = self.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                break

        self.load_state_dict(best_model_state)
        return self

    def test_model(
        self,
        dl: DataLoader,
        task_id: Optional[int] = None,
    ):
        self.eval()
        accs = []

        for data, target in dl:
            if task_id is not None:
                output = F.log_softmax(self(data, task_id), dim=-1)
            else:
                output = F.log_softmax(self(data), dim=-1)
            accs.append(compute_acc(output, target))

        return np.mean(accs)

    def update_priors(self):
        for layer in self.layers:
            layer.prior_weights_mu = layer.weights_mu.data
            layer.prior_weights_sigma = layer.weights_sigma.data

    def get_divergence(self, task_id: Optional[int] = None):
        div = 0.0
        for layer in self.layers:
            div += layer.compute_divergence()
        if task_id is not None:
            div += self.task_specific_heads[task_id].compute_divergence()
        return div


class VCLSingeHeadNN(VCLBase):
    def __init__(
        self,
        input_size: int,
        hidden_sizes: Tuple[int, int],
        output_size: int,
        divergence: Literal["KL", "JS"] = "KL",
    ):
        super(VCLSingeHeadNN, self).__init__()
        self.flatten = nn.Flatten()
        self.layers = nn.ModuleList()
        self.layers.append(VariationalLinear(input_size, hidden_sizes[0], divergence))

        for i in range(len(hidden_sizes) - 1):
            self.layers.append(
                VariationalLinear(hidden_sizes[i], hidden_sizes[i + 1], divergence)
            )

        self.layers.append(VariationalLinear(hidden_sizes[-1], output_size, divergence))

    def forward(self, x):
        x = self.flatten(x)
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            x = F.relu(x)
        x = self.layers[-1](x)
        return x


class VCLMultiHeadNN(VCLBase):
    def __init__(
        self,
        input_size: int,
        hidden_sizes: Tuple[int, int],
        output_size: int,
        n_tasks: int,
        divergence: Literal["KL", "JS"] = "KL",
    ):
        super(VCLMultiHeadNN, self).__init__()
        self.flatten = nn.Flatten()
        self.layers = nn.ModuleList()

        # Shared layers
        self.layers.append(VariationalLinear(input_size, hidden_sizes[0], divergence))
        for i in range(len(hidden_sizes) - 1):
            self.layers.append(
                VariationalLinear(hidden_sizes[i], hidden_sizes[i + 1], divergence)
            )

        # Task-specific output layers
        self.task_specific_heads = nn.ModuleList(
            [
                VariationalLinear(hidden_sizes[-1], output_size, divergence)
                for _ in range(n_tasks)
            ]
        )

    def forward(self, x, task_id):
        x = self.flatten(x)
        for layer in self.layers:
            x = layer(x)
            x = F.relu(x)
        x = self.task_specific_heads[task_id](x)
        return x
