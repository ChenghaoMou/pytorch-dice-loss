from typing import Optional, Union

import torch.nn as nn
import torch.nn.functional as F
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked

patch_typeguard()


class TverskyLoss(nn.Module):
    def __init__(
        self,
        alpha: float = 0.3,
        beta: Optional[float] = 0.7,
        gamma: float = 1,
        epsilon: float = 1e-8,
        reduction: Optional[str] = "mean",
        ignore_index: Optional[int] = None,
        multi_label: bool = False,
        asymmetric: bool = False,
    ):
        """
        Parameters
        ----------
        alpha : TensorType[None]
            The alpha parameter for the tversky loss, which is the weight of the false positives.
        beta : float
            The beta parameter for the tversky loss, which is the weight of the false negatives.
        gamma : float
            The gamma parameter for the focal tversky loss, where gamma < 1 increases the degree of focusing on harder examples.
        epsilon : float
            The epsilon parameter for the tversky loss, which is the epsilon value for numerical stability.
        ignore_index : Optional[int]
            The index to ignore in the target.
        reduction : Optional[str]
            The reduction to apply to the loss.
        multi_label : bool
            If true, the loss is computed for each label separately.
        asymmetric : bool
            If true, then alpha plus beta must be 1.
        """
        super().__init__()
        assert reduction in [
            "none",
            "mean",
            "sum",
        ], "reduction must be one of ['none', 'mean', 'sum']"
        self.alpha = alpha
        if asymmetric:
            self.beta = 1 - alpha
        else:
            self.beta = beta
        self.gamma = gamma
        self.epsilon = epsilon
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.multi_label = multi_label
        self.probability = nn.Softmax(dim=-1) if not multi_label else nn.Sigmoid()

    @typechecked
    def forward(
        self,
        logits: TensorType["batch", "label"],
        target: Union[TensorType["batch"], TensorType["batch", "label"]],
    ):
        C = logits.size(-1)
        proba: TensorType["batch", "label"] = self.probability(logits)

        if not self.multi_label:
            target: TensorType["batch", "label"] = F.one_hot(
                target, num_classes=proba.shape[1]
            )

        if self.ignore_index is not None and 0 <= self.ignore_index < C:
            target[:, self.ignore_index] = 0
            proba[:, self.ignore_index] = 0

        TP: TensorType["batch", "label"] = proba * target
        FP: TensorType["batch", "label"] = proba * (1 - target)
        FN: TensorType["batch", "label"] = (1 - proba) * target

        TI: TensorType["batch", "label"] = (
            (TP + self.epsilon)
            / (TP + self.alpha * FP + self.beta * FN + self.epsilon)
        )
        loss: TensorType["batch"] = ((1 - TI) ** (1 / self.gamma)).sum(dim=-1)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss