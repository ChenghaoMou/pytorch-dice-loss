# pytorch-dice-loss
Re-implementation of [Dice Loss for NLP Tasks](https://github.com/ShannonAI/dice_loss_for_NLP):

- Added more comments
- Simplified some computation

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
