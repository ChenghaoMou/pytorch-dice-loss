from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from einops import rearrange

class DiceLoss(nn.Module):
    def __init__(
        self,
        smooth: Optional[float] = 1,
        square_denominator: Optional[bool] = False,
        with_logits: Optional[bool] = True,
        ohem_ratio: float = 0.0,
        alpha: float = 0.0,
        reduction: Optional[str] = "mean",
    ) -> None:
        """
        A modified Dice Loss for imbalanced data in NLP. Reference: https://arxiv.org/abs/1911.02855;

        Parameters
        ----------
        smooth : Optional[float], optional
            Smoothing parameter (𝛾) for the [Sørensen–Dice coefficient](https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient), by default 1.
        square_denominator : Optional[bool], optional
            To square the denominator (p and y) or not, by default False
        with_logits : Optional[bool], optional
            Whether the inputs is raw logits or softmaxed probas, by default True
        ohem_ratio : float, optional
            Overwhelming ratio (#negative/#positive), by default 0.0. I don't seem to find this in the paper and why this is not 1 by default is beyond me.
            The only thing I can find close to this is in Table 1, which (#negative/#positive) is actually different from what is in the implementation (#positive/#negative).
        alpha : float, optional
            Hyperparameter for controling the decaying factor (1-p), leading to a (1-p)^⍺ decaying rate, by default 0.0 (no decay)
        reduction : Optional[str], optional
            Common parameter for loss computation, one of {"mean", "sum", None}, by default "mean"
        """
        super(DiceLoss, self).__init__()
        
        self.reduction = reduction
        self.with_logits = with_logits
        self.smooth = smooth
        self.square_denominator = square_denominator
        self.ohem_ratio = ohem_ratio
        self.alpha = alpha

    def forward(
        self, inputs: Tensor, target: Tensor, mask: Optional[Tensor] = None
    ) -> Tensor:
        """
        Calculate the self-adjusting Dice Loss for the given inputs and target.

        Parameters
        ----------
        inputs : Tensor
            [Batch size, sequence size, class size]
        target : Tensor
            [Batch size, sequence size]
        mask : Optional[Tensor], optional
            [Batch size, sequence size], by default None

        Returns
        -------
        Tensor
            The final loss

        Examples
        --------
        >>> loss = DiceLoss(with_logits=False, reduction='mean')
        >>> inputs = torch.FloatTensor(
        ...     [[[1, 0, 0], [0, 0, 1], [0, 0, 1], [0, 1, 0], [1, 0, 0]]]
        ... )
        >>> inputs.requires_grad = True
        >>> target = torch.LongTensor([[0, 2, 2, 1, 0]])
        >>> output = loss(
        ...     inputs, target, mask=torch.BoolTensor([[True, True, True, True, True]])
        ... )
        >>> assert output.item()==1.81333327293396, output.item()
        >>> loss = DiceLoss(with_logits=False, reduction='mean')
        >>> inputs = torch.FloatTensor(
        ...     [[[0, 0.5, 0.5], [0, 0.5, 0.5], [0, 0.5, 0.5], [0, 0.5, 0.5], [0, 0.5, 0.5]]]
        ... )
        >>> inputs.requires_grad = True
        >>> target = torch.LongTensor([[0, 2, 2, 1, 0]])
        >>> output = loss(
        ...     inputs, target, mask=torch.BoolTensor([[True, True, True, True, True]])
        ... )
        >>> assert output.item()==2.1454546451568604, output.item()
        >>> loss = DiceLoss(with_logits=True, reduction='mean')
        >>> inputs = torch.FloatTensor(
        ...     [[[0, 0.5, 0], [0, 0.5, 0], [0, 0.5, 0], [0, 0.5, 0], [0.5, 0.5, 0]]]
        ... )
        >>> inputs.requires_grad = True
        >>> target = torch.LongTensor([[0, 2, 2, 1, 0]])
        >>> output = loss(
        ...     inputs, target, mask=torch.BoolTensor([[True, True, True, True, True]])
        ... )
        >>> assert output.item()==2.154679536819458, output.item()
        """

        logits_size = inputs.shape[-1]
        
        # Both unary and binary can be mapped to binary
        if logits_size > 2:
            loss = self._multi_class(inputs, target, logits_size, mask=mask)
        else:
            loss = self._binary_class(inputs, target, logits_size, mask=mask)

        return {'mean': torch.mean, 'sum': torch.sum}.get(self.reduction, lambda x: x)(loss)

    def _compute(self, flat_input, flat_target):

        flat_input = ((1 - flat_input) ** self.alpha) * flat_input
        interection = torch.sum(flat_input * flat_target, -1)

        if self.square_denominator:
            denominator = torch.sum(torch.square(flat_input)) + torch.sum(torch.square(flat_target), -1) + self.smooth
        else:
            denominator = flat_input.sum() + flat_target.sum() + self.smooth

        nominator = 2 * interection + self.smooth

        return 1 - nominator / denominator

    def _multi_class(self, inputs, target, logits_size, mask=None):
        # One-VS-Rest multiclass loss function
        # Assume the target is label indices

        if self.with_logits:
            inputs = F.softmax(inputs, dim=-1)

        inputs = rearrange(inputs, "B S C -> (B S) C")
        target = rearrange(target, "B S -> (B S)")

        flat_input = inputs  # [-1, C]
        flat_target = F.one_hot(target, num_classes=logits_size).float()  # [-1, C]

        if mask is None:
            mask = torch.ones_like(
                target, dtype=target.dtype, device=target.device
            ).unsqueeze(dim=-1)
        else:
            mask = rearrange(mask, "B S -> (B S) 1").float()

        flat_input = flat_input * mask
        flat_target = flat_target * mask

        loss = 0
        total_size = mask.sum()

        for label_class in range(logits_size):

            flat_input_class = flat_input[..., label_class]
            flat_target_class = flat_target[..., label_class]

            if self.ohem_ratio > 0:
                pos_example = flat_target[
                    ..., label_class
                ].bool()  # or target == label_class [-1]
                neg_example = ~pos_example  # or target != label_class [-1]

                pos_num = pos_example.sum()
                neg_num = total_size - pos_num
                keep_num = min(int(pos_num * self.ohem_ratio / logits_size), neg_num)

                if keep_num > 0:

                    neg_scores = torch.masked_select(
                        flat_input, neg_example.view(-1, 1)
                    ).reshape(-1, logits_size)

                    neg_scores_class = neg_scores[:, label_class]
                    neg_scores_sort, _ = torch.sort(
                        neg_scores_class,
                    )
                    threshold = neg_scores_sort[-keep_num + 1]

                    pred_positives = (
                        torch.argmax(flat_input, dim=-1) == label_class
                    )  # [-1]
                    true_positives = pos_example.view(-1)  # [-1]
                    thresholded = flat_input[..., label_class] >= threshold  # [-1]
                    # predicted positives (not necessarily have enough score)
                    # thresholded -> positives and negatives with enough score
                    # pred_positives & thresholded -> examples have enough score to be predicted as positives, including difficult negatives
                    cond = (pred_positives & thresholded) | true_positives
                    ohem_mask_class = torch.where(cond, 1, 0)
                    flat_input_class = flat_input_class * ohem_mask_class
                    flat_target_class = flat_target_class * ohem_mask_class

            # print(flat_input_class, flat_target_class)
            loss_class = self._compute(
                flat_input_class.view(-1, 1), flat_target_class.view(-1, 1)
            )
            loss += loss_class

        return loss

    def _binary_class(self, inputs, target, logits_size, mask=None):

        if self.with_logits:
            if logits_size == 2:
                inputs = F.softmax(inputs, dim=-1)
            else:
                inputs = F.sigmoid(inputs)

        if logits_size == 2:
            # Only take the probabilities for the positive class
            inputs = inputs[:, -1]

        flat_input = inputs.view(-1)
        flat_target = target.view(-1).float()

        if mask is None:
            mask = torch.ones_like(
                flat_target, dtype=flat_target.dtype, device=flat_target.device
            )
        else:
            mask = mask.view(-1).float()

        assert (
            mask.size() == flat_target.size()
        ), f"Incompatible mask shape: {mask.size()} and {flat_target.size()}"

        flat_input = flat_input * mask
        flat_target = flat_target * mask

        if self.ohem_ratio > 0:
            pos_example = flat_target > 0.5
            neg_example = flat_target <= 0.5

            pos_num = pos_example.sum()
            neg_num = neg_example.sum()
            # It only makes sense to make ohem_ratio = #neg/#pos, so here the keep_num is the number of negative examples to keep for learning,
            keep_num = min(int(pos_num * self.ohem_ratio), neg_num)

            neg_scores = torch.masked_select(flat_input, neg_example.bool())
            neg_scores_sort, _ = torch.sort(neg_scores)
            threshold = neg_scores_sort[-keep_num + 1]
            # Only learn from positive examples and negatives examples with score > threshold (difficult ones I assume, because their probabilities are too close to positives)
            cond = (flat_input > threshold) | pos_example.view(-1)
            ohem_mask = torch.where(cond, 1, 0)
            flat_input = flat_input * ohem_mask
            flat_target = flat_target * ohem_mask

        return self._compute(flat_input, flat_target)

    def __str__(self):
        """
        Returns a readable string representation of the instance.
        """
        return f"Dice Loss smooth:{self.smooth}, ohem: {self.ohem_ratio}, alpha: {self.alpha}"

    def __repr__(self):
        """
        Returns a complete string representation of the instance, reconstructable with eval.
        """
        return f"DiceLoss({self.smooth}, {self.square_denominator}, {self.with_logits}, {self.ohem_ratio}, {self.alpha}, '{self.reduction}')"
