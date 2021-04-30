import torch
from torch import nn
from torch.nn.modules.loss import _Loss

__all__ = ["JointLoss2", "JointLoss3", "WeightedLoss"]


class WeightedLoss(_Loss):
    """Wrapper class around loss function that applies weighted with fixed factor.
    This class helps to balance multiple losses if they have different scales
    """

    def __init__(self, loss, weight=1.0):
        super().__init__()
        self.loss = loss
        self.weight = weight

    def forward(self, *input):
        return self.loss(*input) * self.weight


class JointLoss2(_Loss):
    """
    Wrap two loss functions into one. This class computes a weighted sum of two losses.
    """

    def __init__(self, first: nn.Module, second: nn.Module, first_weight=1.0, second_weight=1.0):
        super().__init__()
        self.first = WeightedLoss(first, first_weight)
        self.second = WeightedLoss(second, second_weight)

    def forward(self, *input):
        return self.first(*input) + self.second(*input)


class JointLoss3(_Loss):
    """
    Wrap three loss functions into one. This class computes a weighted sum of three losses.
    """

    def __init__(self, first: nn.Module, second: nn.Module, third: nn.Module, first_weight=1.0, second_weight=1.0,
                 third_weight=1.0):
        super().__init__()
        self.first = WeightedLoss(first, first_weight)
        self.second = WeightedLoss(second, second_weight)
        self.third = WeightedLoss(third, third_weight)

    def forward(self, *input):
        first = self.first(*input)
        second = self.second(*input)
        # third = self.third(*input[0].argmax(dim=1),*input[1])
        return first + second


"""
使用例子：
IoULoss_fn = IoULoss()
lovasz_fn = losses.LovaszLoss(mode='multiclass')
DiceLoss_fn = losses.DiceLoss(mode='multiclass')
SoftCrossEntropy_fn = losses.SoftCrossEntropyLoss(smooth_factor=0.1)
# CrossEntropy_fn = nn.CrossEntropyLoss()
# focal_loss_fn = losses.FocalLoss(mode='multiclass')

# criterion = JointLoss3(first=DiceLoss_fn, second=SoftCrossEntropy_fn, third=IoULoss_fn,
#                         first_weight=1.0, second_weight=1.0, third_weight=1.0).cuda()
criterion = JointLoss2(first=lovasz_fn, second=SoftCrossEntropy_fn,
                       first_weight=1.0, second_weight=1.0).cuda()
"""
