"""PyTorch version of self-balanced Dice Loss for data-imbalanced NLP tasks."""

from .dice_loss import SelfAdjustingDiceLoss

__all__ = ['SelfAdjustingDiceLoss']