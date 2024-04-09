from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader


class EWCLoss:
    def __init__(
        self,
        model: nn.Module,
        dataset: DataLoader,
        task_id: Optional[int] = None,
    ) -> None:
        self.model = model
        self.dataset = dataset
        self.task_id = task_id
        self.importance = {
            n: torch.zeros(p.shape)
            for n, p in model.named_parameters()
            if p.requires_grad
        }
        self._compute_importance()

    def _compute_importance(self) -> None:
        self.model.eval()
        fisher_matrix = {
            n: torch.zeros(p.shape)
            for n, p in self.model.named_parameters()
            if p.requires_grad
        }

        for data, target in self.dataset:
            self.model.zero_grad()
            if self.task_id is not None:
                output = self.model(data, self.task_id)
            else:
                output = self.model(data)
            loss = F.nll_loss(F.log_softmax(output, dim=1), target)
            loss.backward()

            for n, p in self.model.named_parameters():
                if p.grad is not None:
                    fisher_matrix[n] += p.grad.pow(2)

        for n in fisher_matrix.keys():
            self.importance[n] = fisher_matrix[n] / len(self.dataset.dataset)

    def penalty(self, model: nn.Module) -> torch.Tensor:
        loss = 0
        for n, p in model.named_parameters():
            if n in self.importance:
                loss += (
                    self.importance[n] * (p - self.model.state_dict()[n]) ** 2
                ).sum()
        return loss
