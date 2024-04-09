from typing import Tuple, Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm


class EWCBase(nn.Module):
    def __init__(self) -> None:
        super(EWCBase, self).__init__()

    def train_model(
        self,
        train_loader,
        val_loader,
        ewc_objects: list,
        lambda_ewc: float,
        n_epochs: int = 10,
        task_id: Optional[int] = None,
        lr: float = 1e-03,
        weight_decay: float = 1e-05,
        patience: int = 5,
    ):
        optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)

        highest_acc = 0.0
        patience_counter = 0
        best_model_state = self.state_dict()

        for _ in tqdm(range(n_epochs)):
            if task_id == 0:
                self.train_without_ewc(train_loader, optimizer, task_id)
            else:
                self.train_with_ewc(
                    train_loader,
                    optimizer,
                    ewc_objects,
                    lambda_ewc,
                    task_id,
                )
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

    def train_without_ewc(
        self,
        data_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        task_id: Optional[int] = None,
    ) -> None:
        self.train()
        for data, target in data_loader:
            optimizer.zero_grad()
            if task_id is not None:
                output = self(data, task_id)
            else:
                output = self(data)

            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()

    def train_with_ewc(
        self,
        data_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        ewc_objects: list,
        lambda_ewc: float,
        task_id: Optional[int] = None,
    ) -> None:
        self.train()
        for data, target in data_loader:
            optimizer.zero_grad()
            if task_id is not None:
                output = self(data, task_id)
            else:
                output = self(data)

            task_loss = F.cross_entropy(output, target)

            ewc_penalty = torch.tensor(0.0)
            for ewc in ewc_objects:
                ewc_penalty += ewc.penalty(self)

            total_loss = task_loss + (lambda_ewc * ewc_penalty)
            total_loss.backward()
            optimizer.step()

    def test_model(
        self,
        data_loader: DataLoader,
        task_id: Optional[int] = None,
    ) -> float:
        self.eval()
        correct = 0
        with torch.no_grad():
            for data, target in data_loader:
                if task_id is not None:
                    output = self(data, task_id)
                else:
                    output = self(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        accuracy = correct / len(data_loader.dataset)
        return accuracy


class EWCSingleHeadNN(EWCBase):
    def __init__(
        self,
        input_size: int = 784,
        hidden_sizes: Tuple[int, int] = (256, 256),
        n_classes: int = 10,
    ) -> None:
        super(EWCSingleHeadNN, self).__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class EWCMultiHeadNN(EWCBase):
    def __init__(
        self,
        n_tasks: int = 5,
        input_size: int = 784,
        hidden_sizes: Tuple[int, int] = (256, 256),
        output_size: int = 2,
    ) -> None:
        super(EWCMultiHeadNN, self).__init__()
        self.n_tasks = n_tasks

        # Shared layers
        self.shared_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
        )

        # Task-specific output layers
        self.task_heads = nn.ModuleList(
            [nn.Linear(hidden_sizes[1], output_size) for _ in range(n_tasks)]
        )

    def forward(self, x: torch.Tensor, task_id: int) -> torch.Tensor:
        x = self.shared_layers(x)
        x = self.task_heads[task_id](x)
        return x
