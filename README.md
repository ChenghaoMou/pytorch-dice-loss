# pytorch-dice-loss
[![PyPI version](https://badge.fury.io/py/pytorch-dice-loss.svg)](https://badge.fury.io/py/pytorch-dice-loss) [![Codacy Badge](https://app.codacy.com/project/badge/Grade/5f80c041543f424ebfe7967a677879d8)](https://www.codacy.com/gh/ChenghaoMou/pytorch-dice-loss/dashboard?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=ChenghaoMou/pytorch-dice-loss&amp;utm_campaign=Badge_Grade)

Re-implementation of [Dice Loss for NLP Tasks](https://github.com/ShannonAI/dice_loss_for_NLP):

-  Added more comments
-  Simplified some computation

## Installation

```shell
pip install pytorch-dice-loss
```

## Usage
```python
from pytorch_dice_loss import DiceLoss

loss = DiceLoss(with_logits=False, reduction='mean')
# [B, S, C]
input = torch.FloatTensor(
    [[[1, 0, 0], [0, 0, 1], [0, 0, 1], [0, 1, 0], [1, 0, 0]]]
)
input.requires_grad = True
# [B, S]
target = torch.LongTensor([[0, 2, 2, 1, 0]])
mask=torch.BoolTensor([[True, True, True, True, True]]
output = loss(
    input, target, mask=mask)
)
```