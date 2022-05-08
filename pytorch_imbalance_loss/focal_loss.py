from typing import List, Optional, Union

import torch.nn as nn
import torch.nn.functional as F
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked

from pytorch_imbalance_loss.tversky_loss import TverskyLoss

patch_typeguard()

class FocalLoss(nn.Module):
    
    """Focal loss from https://arxiv.org/abs/1708.02002."""
    
    def __init__(
        self,
        alpha: Optional[TensorType[None]] = None,
        gamma: float = 2,
        reduction: Optional[str] = "mean",
        ignore_index: Optional[int] = None,
        multi_label: bool = False,
    ):
        """
        Parameters
        ----------
        alpha : TensorType[None]
            The alpha parameter for the focal loss, which is the class weights.
        gamma : float
            The gamma parameter for the focal loss, which is the downweighting factor.
        ignore_index : Optional[int]
            The index to ignore in the target.
        reduction : Optional[str]
            The reduction to apply to the loss.
        """
        super().__init__()
        assert reduction in ["none", "mean", "sum"], "reduction must be one of ['none', 'mean', 'sum']"
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.multi_label = multi_label

    @typechecked
    def forward(
        self, logits: TensorType["batch", "label"], target: Union[TensorType["batch"], TensorType["batch", "label"]]
    ):
        """
        Calculates the focal loss for a batch of logits and targets.

        Parameters
        ----------
        logits : torch.Tensor
            The logits of the model.
        target : torch.Tensor
            The target labels.
        
        Returns
        -------
        torch.Tensor
            The loss.
        
        Examples
        --------
        >>> import torch
        >>> x = torch.randn(10, 2)
        >>> loss = FocalLoss(torch.tensor([0.5, 0.5]))
        >>> perfectly_wrong = loss(x, 1 - torch.argmax(x, dim=-1).long())
        >>> perfectly_correct = loss(x, torch.argmax(x, dim=-1).long())
        >>> assert perfectly_wrong > perfectly_correct, "Focal loss is not working correctly"
        """
        if self.multi_label:
            proba: TensorType["batch", "label"] = F.sigmoid(logits)
            ce: TensorType["batch", "label"] = F.binary_cross_entropy(proba, target, reduction="none")
            truth_prob: TensorType["batch", "label"] = proba * target
        else:
            log_proba: TensorType["batch", "label"] = F.log_softmax(logits, dim=-1)
            ce:  TensorType["batch"] = F.nll_loss(log_proba, target, weight=self.alpha, reduction="none", ignore_index=self.ignore_index)
            truth_prob: TensorType["batch"] = (log_proba.gather(dim=-1, index=target.unsqueeze(-1)).squeeze(-1)).exp()
        
        focal_term: Union[TensorType["batch", "label"], TensorType["batch"]] = (1 - truth_prob) ** self.gamma
        loss: Union[TensorType["batch", "label"], TensorType["batch"]] = focal_term * ce

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss

class UnifiedFocalLoss(nn.Module):

    def __init__(
        self,
        class_weights: TensorType[None],
        gamma: float,
        lambda_: float,
        reduction: Optional[str] = "mean",
        ignore_index: Optional[int] = None,
        multi_label: bool = False,
        epsilon: float = 1e-8,
    ):
        super().__init__()
        self.sigma = class_weights
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.multi_label = multi_label
        self.lambda_ = lambda_
        self.epsilon = epsilon
        self.focal_loss = FocalLoss(alpha=self.sigma, gamma=1 - self.gamma, reduction="none", ignore_index=self.ignore_index, multi_label=self.multi_label)
        self.tversky_loss = TverskyLoss(alpha=self.sigma, beta=1 - self.sigma, gamma=1/self.gamma, reduction="none", epsilon=self.epsilon, ignore_index=self.ignore_index, multi_label=self.multi_label)
    
    @typechecked
    def forward(self, logits: TensorType["batch", "label"], target: Union[TensorType["batch"], TensorType["batch", "label"]]):
        
        modifier_focal_loss = self.focal_loss(logits, target)
        modifier_tversky_loss = self.tversky_loss(logits, target)
        loss = self.lambda_ * modifier_tversky_loss + (1 - self.lambda_) * modifier_focal_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss