import pytest
import torch
import torch.nn as nn

from pytorch_imbalance_loss.focal_loss import FocalLoss


@pytest.mark.parametrize("idx", range(1000))
def test_focal_loss(idx):
    x = torch.rand(64, 2)
    l = torch.rand(64).ge(0.1).long()

    output0 = FocalLoss(gamma=0, reduction="none")(x, l)
    output1 = nn.CrossEntropyLoss(reduction="none")(x, l)
    assert torch.isclose(output0, output1).all()


@pytest.mark.parametrize("idx", range(1000))
def test_focal_loss_multilabel(idx):
    x = torch.rand(64, 10)
    l = torch.rand(64, 10).ge(0.5).float()

    output0 = FocalLoss(gamma=0, reduction="none", multi_label=True)(x, l)
    output1 = nn.BCEWithLogitsLoss(reduction="none")(x, l)
    assert torch.isclose(output0, output1).all()
