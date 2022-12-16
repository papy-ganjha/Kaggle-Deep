from typing import List
from torch import Tensor
from torch import nn
import torch.nn.functional as F
import torch

ALPHA = 0.5
BETA = 0.5
GAMMA = 1


class FocalTverskyLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(FocalTverskyLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1, alpha=ALPHA, beta=BETA, gamma=GAMMA):

        # comment out if your model contains a sigmoid or equivalent activation layer
        # inputs = F.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()
        FP = ((1-targets) * inputs).sum()
        FN = (targets * (1-inputs)).sum()

        Tversky = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)
        FocalTversky = (1 - Tversky)**gamma

        return FocalTversky + F.binary_cross_entropy(inputs, targets)


def compute_per_channel_dice(input, target, epsilon=1e-5, ignore_index=None, weight=None):
    # assumes that input is a normalized probability

    # input and target shapes must match
    assert input.size() == target.size(), "'input' and 'target' must have the same shape"

    # mask ignore_index if present
    if ignore_index is not None:
        mask = target.clone().ne_(ignore_index)
        mask.requires_grad = False

        input = input * mask
        target = target * mask

    input = flatten(input)
    target = flatten(target)

    target = target.float()
    # Compute per channel Dice Coefficient
    intersect = (input * target).sum(-1)
    if weight is not None:
        intersect = weight * intersect

    denominator = (input + target).sum(-1)
    return 2. * intersect / denominator.clamp(min=epsilon)


class DiceLoss(nn.Module):
    """Computes Dice Loss, which just 1 - DiceCoefficient described above.
    Additionally allows per-class weights to be provided.
    """

    def __init__(self, epsilon=1e-5, weight=None, ignore_index=None, sigmoid_normalization=True,
                 skip_last_target=False):
        super(DiceLoss, self).__init__()
        self.epsilon = epsilon
        self.register_buffer('weight', weight)
        self.ignore_index = ignore_index
        # The output from the network during training is assumed to be un-normalized probabilities and we would
        # like to normalize the logits. Since Dice (or soft Dice in this case) is usually used for binary data,
        # normalizing the channels with Sigmoid is the default choice even for multi-class segmentation problems.
        # However if one would like to apply Softmax in order to get the proper probability distribution from the
        # output, just specify sigmoid_normalization=False.
        if sigmoid_normalization:
            self.normalization = nn.Sigmoid()
        else:
            self.normalization = nn.Softmax(dim=1)
        # if True skip the last channel in the target
        self.skip_last_target = skip_last_target

    def forward(self, input, target):
        # get probabilities from logits
        # input = self.normalization(input)
        if self.weight is not None:
            weight = torch.autograd.Variable(self.weight, requires_grad=False)
        else:
            weight = None

        if self.skip_last_target:
            target = target[:, :-1, ...]

        per_channel_dice = compute_per_channel_dice(input, target, epsilon=self.epsilon, ignore_index=self.ignore_index,
                                                    weight=weight)
        # Average the Dice score across all channels/classes
        return torch.mean(1. - per_channel_dice) + F.binary_cross_entropy(input, target)


class DiceLossMulticlass(nn.Module):
    def __init__(self, weights=None, size_average=False):
        super(DiceLossMulticlass, self).__init__()
        self.weights = weights
        self.size_average = size_average

    def forward(self, inputs, targets, smooth=1):
        if self.weights is not None:
            assert self.weights.shape == (targets.shape[1], )

        # make a copy not to change the default weights in the instance of DiceLossMulticlass
        weights = self.weights

        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)

        # flatten label and prediction images, leave BATCH and NUM_CLASSES
        # (BATCH, NUM_CLASSES, H, W) -> (BATCH, NUM_CLASSES, H * W)
        inputs = inputs.view(inputs.shape[0], inputs.shape[1], -1)
        targets = targets.view(targets.shape[0], targets.shape[1], -1)

        #intersection = (inputs * targets).sum()
        intersection = (inputs * targets).sum(0).sum(1)
        #dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)
        dice = (2.*intersection + smooth) / \
            (inputs.sum(0).sum(1) + targets.sum(0).sum(1) + smooth)

        if (weights is None) and self.size_average == True:
            weights = (targets == 1).sum(0).sum(1)
            weights /= weights.sum()  # so they sum up to 1

        if weights is not None:
            return 1 - (dice*weights).mean()
        else:
            return 1 - dice.mean()


def iou_compute(pred, true, from_logits: bool):
    if from_logits:
        pred = F.softmax(pred, dim=1)
    intersection = (pred * true).sum()
    total = (pred + true).sum()
    return (2 * intersection) / (total - intersection)


class MulticlassJaccardLoss(nn.Module):
    """Implementation of Jaccard loss for multiclass (semantic) image segmentation task
    """
    __name__ = 'mc_jaccard_loss'

    def __init__(self, classes: List[int] = None, from_logits=True, weight=None, reduction='elementwise_mean'):
        super(MulticlassJaccardLoss, self).__init__()
        self.classes = classes
        self.from_logits = from_logits
        self.weight = weight
        self.reduction = reduction

    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        """
        :param y_pred: NxCxHxW
        :param y_true: NxHxW
        :return: scalar
        """
        if self.from_logits:
            y_pred = y_pred.softmax(dim=1)

        n_classes = y_pred.size(1)
        smooth = 1e-3

        if self.classes is None:
            classes = range(n_classes)
        else:
            classes = self.classes
            n_classes = len(classes)

        loss = torch.zeros(n_classes, dtype=torch.float, device=y_pred.device)

        if self.weight is None:
            weights = [1] * n_classes
        else:
            weights = self.weight

        for class_index, weight in zip(classes, weights):

            jaccard_target = y_true[:, class_index, ...]
            jaccard_output = y_pred[:, class_index, ...]

            num_preds = jaccard_target.long().sum()

            if num_preds == 0:
                loss[class_index-1] = 0  # custom
            else:
                iou = iou_compute(
                    jaccard_output, jaccard_target, from_logits=False)
                loss[class_index] = (1.0 - iou) * weight  # custom

        if self.reduction == 'elementwise_mean':
            return loss.mean()

        if self.reduction == 'sum':
            return loss.sum()

        return loss


class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):

        # comment out if your model contains a sigmoid or equivalent activation layer
        # inputs = F.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2.*intersection + smooth) / \
            (inputs.sum() + targets.sum() + smooth)
        BCE = F.cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss

        return Dice_BCE


def flatten(tensor):
    """Flattens a given tensor such that the channel axis is first.
    The shapes are transformed as follows:
       (N, C, D, H, W) -> (C, N * D * H * W)
    """
    C = tensor.size(1)
    # new axis order
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))
    # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
    transposed = tensor.permute(axis_order)
    # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
    return transposed.contiguous().view(C, -1)


class GeneralizedDiceLoss(nn.Module):
    """Computes Generalized Dice Loss (GDL) as described in https://arxiv.org/pdf/1707.03237.pdf
    """

    def __init__(self, epsilon=1e-5, weight=None, ignore_index=None, sigmoid_normalization=False):
        super(GeneralizedDiceLoss, self).__init__()
        self.epsilon = epsilon
        self.weight = weight
        self.ignore_index = ignore_index
        if sigmoid_normalization:
            self.normalization = nn.Sigmoid()
        else:
            self.normalization = nn.Softmax(dim=1)

    def forward(self, input, target):
        # get probabilities from logits
        # input = self.normalization(input)

        assert input.size() == target.size(), "'input' and 'target' must have the same shape"

        # mask ignore_index if present
        if self.ignore_index is not None:
            mask = target.clone().ne_(self.ignore_index)
            mask.requires_grad = False

            input = input * mask
            target = target * mask

        input = flatten(input)
        target = flatten(target)

        target = target.float()
        target_sum = target.sum(-1)
        class_weights = torch.autograd.Variable(
            1. / (target_sum * target_sum).clamp(min=self.epsilon), requires_grad=True)

        intersect = (input * target).sum(-1) * (1/25)
        if self.weight is not None:
            weight = torch.autograd.Variable(self.weight, requires_grad=True)
            intersect = weight * intersect
        intersect = intersect.sum()

        denominator = ((input + target).sum(-1) * class_weights).sum()

        return 1. - 2. * intersect / denominator.clamp(min=self.epsilon)
