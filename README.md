# pytorch-imbalance-loss
[![PyPI version](https://badge.fury.io/py/pytorch-dice-loss.svg)](https://badge.fury.io/py/pytorch-dice-loss) [![Codacy Badge](https://app.codacy.com/project/badge/Grade/5f80c041543f424ebfe7967a677879d8)](https://www.codacy.com/gh/ChenghaoMou/pytorch-dice-loss/dashboard?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=ChenghaoMou/pytorch-dice-loss&amp;utm_campaign=Badge_Grade)

Implementations of loss functions for imbalanced NLP data.

- [x] [Dice Loss](https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient)
- [x] [Self-adjusting Dice Loss](https://arxiv.org/pdf/1911.02855.pdf)
- [x] [Focal Loss](http://arxiv.org/abs/1708.02002)
- [x] [Tversky Loss](https://doi.org/10.1007/978-3-319-67389-9_44)
- [x] [Focal Tversky Loss](https://ieeexplore.ieee.org/document/8759329)
- [x] [Unified Focal Loss](https://doi.org/10.1016/j.compmedimag.2021.102026)

## From Binary Cross Entropy to Focal Loss

### Cross Entropy

$$
BCE(\underset{n\times 1}{\hat{y}}, \underset{n\times 1}{y}) = - (\underset{n\times 1}{y}\log(\underset{n\times 1}{\hat{y}}) + \underset{n\times 1}{(1 - y)}\log(\underset{n\times 1}{1 - \hat{y}}))
$$

It is easier to understand if we express it in binary values
$$

BCE(\underset{n\times 1}{\hat{y}}, \underset{n\times 1}{y}) =
\begin{cases}
- (0 + 0) = 0 & \text{if $y_i=1$ and $\hat{y}_i=1$} \\
- (-\infty + 0) = \infty & \text{if $y_i=1$ and $\hat{y}_i=0$} \\
- (0 + 0) = 0 & \text{if $y_i=0$ and $\hat{y}_i=0$} \\
- (0 + -\infty) = \infty & \text{if $y_i=0$ and $\hat{y}_i=1$} \\
\end{cases}
$$

Or in terms of probablity

$$
BCE(\underset{n\times 1}{\hat{y}}, \underset{n\times 1}{y}) = - (\underset{\text{prob. being 1}}{y}\log(\underset{\text{pred. being 1}}{\hat{y}}) + \underset{\text{prob. being 0}}{(1 - y)}\log(\underset{\text{pred. being 0}}{1 - \hat{y}}))
$$

Extending it to multi-class:

$$
CE(\underset{n\times c}{\hat{y}}, \underset{n\times c}{y}) = - \frac{1}{N}\sum_{i=1}^{N}\sum_{c=1}^{C}\underset{\text{prob. i being c}}{y_{i, c}}\log(\underset{\text{pred. i being c}}{\hat{y}_{i, c}})
$$

The larger the gap between $y_i$ and $\hat{y}_i$ is, the larger the loss will be.

The problem of class imbalance is that the model will be more frequently updated to accommodate certain classes and not the rest.

### Focal Loss

The essence of Focal Loss is to down-weight **easy** examples that are contributed by the imbalanced classes.

The additional factor introduced by Focal Loss is 

$$
\alpha (1 - \hat{y}_{i,\text{correct c}})^\gamma
$$

and the whole loss, in multi-class scenario is

$$
\begin{aligned}
    FL(\underset{n\times c}{\hat{y}}, \underset{n\times c}{y}) &= - \frac{1}{N}\sum_{i=1}^{N}\sum_{c=1}^{C}\alpha (1 - \hat{y}_{i,\text{correct c}})^\gamma\underset{\text{prob. i being c}}{y_{i, c}}\log(\underset{\text{pred. i being c}}{\hat{y}_{i, c}})\\
    &= - \frac{1}{N}\sum_{i=1}^{N}\alpha (1 - \hat{y}_{i,\text{correct c}})^\gamma\sum_{c=1}^{C}\underset{\text{prob. i being c}}{y_{i, c}}\log(\underset{\text{pred. i being c}}{\hat{y}_{i, c}})
\end{aligned}
$$

If the probability assigned to the true class is close to 1, the loss would be small(er), even though it might be wrong. If the prediction is far off, the loss is amplied as a result.

### Dice Loss

#### Sørensen–Dice coefficient

Assuming binary classification, the Sørensen–Dice coefficient for one example $x_i$ is given as:

$$
DSC(x_i) = \frac{2p_{i1}y_{i1} + \gamma}{p_{i1} + y_{i1} + \gamma}
$$

$$
DSC(x_i) = 
\begin{cases}
\frac{2 * 1 * 1 + 1}{1 + 1 + 1} = 1 & \text{if $y_{i1}=1$ and $p_{i1}=1$} \\
\frac{2 * 0 * 1 + 1}{0 + 1 + 1} = 0.5 & \text{if $y_{i1}=1$ and $p_{i1}=0$} \\
\frac{2 * 1 * 0 + 1}{0 + 0 + 1} = 1 & \text{if $y_{i1}=0$ and $p_{i1}=0$} \\
\frac{2 * 1 * 0 + 1}{1 + 0 + 1} = 0.5 & \text{if $y_{i1}=0$ and $p_{i1}=1$} \\
\end{cases}
$$

The Dice Loss is then defined as:

$$
DL = \frac{1}{N}\sum_i[1 - \frac{2p_{i1}y_{i1} + \gamma}{p_{i1}^2 + y_{i1}^2 + \gamma}]
$$

The denominator terms are squared for faster convergence.

Or in another form:

$$
DL = 1 - \frac{2\sum_i\sum_cp_{ic}y_{ic} + \gamma}{\sum_i\sum_c(p_{ic}^2 + y_{ic}^2) + \gamma}
$$

#### Self-adjusting Dice Loss

The self-adjusting introduces a parameter called overwhelming ratio to control how many negative examples to learn per class, thus encourages the model to have a balanced perspection of each class.

The set up is as follows:

1. Given a overwhelming ratio $r$: how many negatives to learn as *a percentage of number of positives*. For example, for one class $c_i$, the number of positives are $P_i$ and the number of positives are therefore $N - P_i$, but the number of negatives you will learn is $min(N - P_i, P_i \times r)$.
2. How to choose those negatives to learn: (within one class), sort your **predictions** on negative examples and choose the  $min(N - P_i, P_i \times r)$-th largest probability as the threshold.
3. For this particular class, now you have a set of examples to learn from:
   1. all positive examples (TP + FN)
   2. all predicted positive examples (TP + FP)
   3. all examples with a probability >= threshold in **2**: This only extend the previous two sets by including difficult TNs, if any

### Tversky Loss

The Tversky Loss is a more generalized loss that looks very similar to Dice Loss. It introduces two more parameters for False Positives and False Negatives:


$$
TL = \frac{1}{N\times C}\sum_i\sum^C[1 - \frac{2\overset{\text{TP}}{p_{i1}y_{i1}} + \gamma}{\underset{\text{TP}}{p_{i1}y_{i1}} + \alpha \underset{\text{FP}}{p_{i1}y_{i0}} + \beta \underset{\text{FN}}{p_{i0}y_{y1}}+ \gamma}]
$$

In the asymmetric variant: $\alpha + \beta = 1$.




### Unified Focal Loss

Only the symmetric version is included here since the asymmetric was only designed for binary segmentation problems. I might extend that in the near future.

By replacing $\alpha$ and $\gamma$ with $\delta$ and $1-\gamma$ in Focal Loss and adding a focal term in Tversky Loss, the symmetric Unfied Focal Loss is defined as:

$$
UFL(\underset{n\times c}{\hat{y}}, \underset{n\times c}{y}) = - \frac{\underline{\lambda}}{N}\sum_{i=1}^{N}\underline{\delta} (1 - \hat{y}_{i,\text{correct c}})^{\underline{1 - \gamma}}\sum_{c=1}^{C}\underset{\text{prob. i being c}}{y_{i, c}}\log(\underset{\text{pred. i being c}}{\hat{y}_{i, c}}) + \underline{(1 - \lambda)} (\frac{1}{N\times C}\sum_i\sum^C(1 - \frac{2\overset{\text{TP}}{p_{i1}y_{i1}} + \gamma}{\underset{\text{TP}}{p_{i1}y_{i1}} + \alpha \underset{\text{FP}}{p_{i1}y_{i0}} + \beta \underset{\text{FN}}{p_{i0}y_{y1}}+ \gamma}) ^ {\underline{\gamma}})
$$


