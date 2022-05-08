import logging
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked

patch_typeguard()
logger = logging.getLogger(__name__)


class DiceLoss(nn.Module):
    def __init__(
        self,
        gamma: float = 1.0,
        squared: bool = True,
        reduction: Optional[str] = "mean",
        ignore_index: Optional[int] = None,
        multi_label: bool = False,
    ) -> None:
        """
        Dice loss function.

        Parameters
        ----------
        gamma : float
            Gamma value for the dice loss.
        squared : bool
            If true, the loss is computed as using squared denominator terms.
        reduction : str
            Specifies the reduction to apply to the output.
        ignore_index : int
            The index to ignore in the target.
        multi_label : bool
            If true, the loss is computed for each label separately.
        """
        super().__init__()
        self.reduction = reduction
        self.gamma = gamma
        self.squared = squared
        self.ignore_index = ignore_index
        self.multi_label = multi_label

    @typechecked
    def forward(
        self, logits: TensorType["batch", "label"], target: Union[TensorType["batch"], TensorType["batch", "label"]]
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        logits : Tensor
            The logits tensor.
        target : Tensor
            The target tensor.

        Returns
        -------
        Tensor
            The loss tensor.

        Examples
        --------
        >>> import torch
        >>> x = torch.rand(16, 10)
        >>> random_y = torch.randint(0, 10, (16,))
        >>> perfect_y = torch.argmax(x, dim=1)
        >>> ignored_y = torch.ones_like(random_y)
        >>> ignored_x = torch.zeros_like(x)
        >>> ignored_x[:, 1] = 100
        >>> loss = DiceLoss(ignore_index=1, gamma=1e-8)(ignored_x, ignored_y)
        >>> assert torch.isclose(loss, torch.zeros_like(loss)), f"{loss} should be zero"
        >>> perfect_loss = DiceLoss(gamma=1e-8)(x, perfect_y)
        >>> large_loss = DiceLoss(gamma=1e-8)(x, random_y)
        >>> assert large_loss >= perfect_loss, f"{large_loss} > {perfect_loss} should be true"
        """
        if self.multi_label:
            logits: TensorType["batch", "label"] = F.logsigmoid(logits).exp()
            target_one_hot: TensorType["batch", "label"] = target.float()
        else:
            logits: TensorType["batch", "label"] = F.log_softmax(logits, dim=-1).exp()
            target_one_hot: TensorType["batch", "label"] = F.one_hot(
                target, num_classes=logits.shape[1]
            ).float()
        
        mask: TensorType["batch", "label"] = torch.ones_like(logits).float()
        if self.ignore_index is not None and 0 <= self.ignore_index < logits.size(1):
            mask[:, self.ignore_index] = 0

        if self.squared:
            denominator: TensorType["batch", "label"] = (logits**2 + target_one_hot**2)
        else:
            denominator: TensorType["batch", "label"] = (logits + target_one_hot)
        
        denominator: TensorType["batch"] = (denominator * mask).sum(dim=-1)
        numerator: TensorType["batch"] = ((logits * target_one_hot) * mask).sum(dim=-1)

        loss: TensorType["batch"] = 1.0 - (2.0 * numerator + self.gamma) / (denominator + self.gamma)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class SelfAdjustingDiceLoss(nn.Module):
    def __init__(
        self,
        gamma: Optional[float] = 1,
        squared: Optional[bool] = False,
        overwhelming_ratio: float = 0.0,
        alpha: float = 0.0,
        reduction: Optional[str] = "mean",
        ignore_index: Optional[int] = None,
        multi_label: bool = False,
    ) -> None:
        """A modified Dice Loss for imbalanced data in NLP. Reference: https://arxiv.org/abs/1911.02855;

        Parameters
        ----------
        gamma : Optional[float], optional
            Smoothing parameter (𝛾) for the [Sørensen–Dice coefficient](https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient), by default 1.
        squared : Optional[bool], optional
            To square the denominator (p and y) or not, by default False
        overwhelming_ratio : float, optional
            Overwhelming ratio (#negative/#positive), by default 0.0. I don't seem to find this in the paper and why this is not 1 by default is beyond me.
            The only thing I can find close to this is in Table 1, which (#negative/#positive) is actually different from what is in their orginal implementation (#positive/#negative).
        alpha : float, optional
            Hyperparameter for controling the decaying factor (1-p), leading to a (1-p)^⍺ decaying rate, by default 0.0 (no decay)
        reduction : Optional[str], optional
            Common parameter for loss computation, one of {"mean", "sum", "none"}, by default "mean"
        ignore_index : Optional[int], optional
            The index to ignore in the target.
        multi_label : bool, optional
            If true, the loss is computed for each label separately.
        """
        super(SelfAdjustingDiceLoss, self).__init__()

        self.reduction = reduction
        self.gamma = gamma
        self.squared = squared
        self.overwhelming_ratio = overwhelming_ratio
        self.alpha = alpha
        self.ignore_index = ignore_index
        self.multi_label = multi_label

    def forward(
        self, logits: TensorType["batch", "label"], target: TensorType["batch", "label"]
    ) -> Tensor:
        """
        Calculate the self-adjusting Dice Loss for the given inputs and target.

        Parameters
        ----------
        inputs : Tensor
            [Batch size, class size]
        target : Tensor
            [Batch size]

        Returns
        -------
        Tensor
            The final loss

        Examples
        --------
        >>> import torch
        >>> x = torch.rand(16, 10)
        >>> random_y = torch.randint(0, 10, (16,))
        >>> perfect_y = torch.argmax(x, dim=1)
        >>> perfect_loss = SelfAdjustingDiceLoss(gamma=1e-8)(x, perfect_y)
        >>> large_loss = SelfAdjustingDiceLoss(gamma=1e-8)(x, random_y)
        >>> assert large_loss >= perfect_loss, f"{large_loss} > {perfect_loss} should be true"
        """

        C = logits.shape[-1]

        if self.multi_label:
            logits: TensorType["batch", "label"] = F.logsigmoid(logits).exp()
            target_one_hot: TensorType["batch", "label"] = target.float()
        else:
            logits: TensorType["batch", "label"] = F.log_softmax(logits, dim=-1).exp()
            target_one_hot: TensorType["batch", "label"] = F.one_hot(
                target, num_classes=logits.shape[1]
            ).float()
        mask: TensorType["batch", "label"] = torch.ones_like(logits).float()
        if self.ignore_index is not None and 0 <= self.ignore_index < logits.size(1):
            mask[:, self.ignore_index] = 0
        loss = 0

        for label_class in range(C):

            unary_probas: TensorType["batch"] = logits[..., label_class]
            unary_target: TensorType["batch"] = target_one_hot[..., label_class]
            unary_mask: TensorType["batch"] = mask[..., label_class]

            if self.overwhelming_ratio > 0:
                pos_mask: TensorType["batch"] = unary_target.bool()
                neg_mask: TensorType["batch"] = ~pos_mask

                pos_num = pos_mask.sum()
                neg_num = len(target) - pos_num

                # Only learn from this many negative examples (difficult ones) for each class
                keep_num = min(int(pos_num * self.overwhelming_ratio / C), neg_num)

                if keep_num > 0:

                    neg_scores_class: TensorType["batch"] = torch.masked_select(
                        unary_probas, neg_mask.view(-1, 1)
                    )
                    neg_scores_sort, _ = torch.sort(
                        neg_scores_class,
                    )
                    threshold = neg_scores_sort[-keep_num + 1]

                    pred_positives = torch.argmax(logits, dim=-1) == label_class  # [-1]
                    true_positives = pos_mask.view(-1)  # [-1]
                    thresholded = unary_probas >= threshold  # [-1]
                    # predicted positives (not necessarily have enough score)
                    # thresholded -> positives and negatives with enough score
                    # pred_positives & thresholded -> examples have enough score to be predicted as positives, including difficult negatives
                    cond = (pred_positives & thresholded) | true_positives
                    ohem_mask_class = torch.where(cond, 1, 0)
                    unary_probas = unary_probas * ohem_mask_class
                    unary_target = unary_target * ohem_mask_class

            unary_probas: TensorType["batch"] = ((1 - unary_probas) ** self.alpha) * unary_probas

            if self.squared:
                denominator: TensorType["batch"] = (unary_probas**2 + unary_target**2)
            else:
                denominator: TensorType["batch"] = (unary_probas + unary_target)

            denominator: TensorType["batch"] = (denominator * unary_mask).view(-1)
            numerator: TensorType["batch"] = ((unary_probas * unary_target) * unary_mask).view(-1)

            loss += 1.0 - (2.0 * numerator + self.gamma) / (denominator + self.gamma)

        loss: TensorType["batch"] = loss / C

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss
