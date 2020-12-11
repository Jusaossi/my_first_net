import torch
import torch.nn as nn
import torch.nn.functional as F


class MyDiceLoss2(nn.Module):
    def __init__(self, reduction='mean'):
        super(MyDiceLoss2, self).__init__()
        self.reduction = reduction

    def forward(self, inputs, targets, smooth=1e-5):
        intersection = torch.sum(inputs * targets)
        union = torch.sum(inputs**2) + torch.sum(targets**2)
        dice = 2 * (intersection + smooth) / (union + smooth)
        own_loss = torch.add(1, - dice)
        if self.reduction == 'sum':
            return own_loss.sum()
        else:
            return own_loss.mean()


class MyDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(MyDiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        # inputs = F.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        return 1 - dice


class MyLogDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(MyLogDiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        # inputs = F.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        return - dice.log()

class MyLogDiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(MyLogDiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        # inputs = F.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        dice_loss = - dice.log()
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss

        return Dice_BCE

class MyDiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(MyDiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        # inputs = F.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss

        return Dice_BCE


class MyIoULoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(MyIoULoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        # inputs = F.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # intersection is equivalent to True Positive count
        # union is the mutually inclusive area of all labels & predictions
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection

        IoU = (intersection + smooth) / (union + smooth)

        return 1 - IoU


class MyTverskyBceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(MyTverskyBceLoss, self).__init__()

    def forward(self, inputs, targets, alpha, beta=1, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        # inputs = F.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()
        FP = ((1 - targets) * inputs).sum()
        FN = (targets * (1 - inputs)).sum()

        Tversky = 1 - (TP + smooth) / (TP + alpha * FP + beta * FN + smooth)
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        return Tversky + BCE


class MyTverskyLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(MyTverskyLoss, self).__init__()

    def forward(self, inputs, targets, alpha, beta=1, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        # inputs = F.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()
        FP = ((1 - targets) * inputs).sum()
        FN = (targets * (1 - inputs)).sum()

        Tversky = 1 - (TP + smooth) / (TP + alpha * FP + beta * FN + smooth)

        return Tversky
class MyFocalTverskyLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(MyFocalTverskyLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1, alpha=0.5, beta=1, gamma=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        # inputs = F.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()
        FP = ((1 - targets) * inputs).sum()
        FN = (targets * (1 - inputs)).sum()

        Tversky = (TP + smooth) / (TP + alpha * FP + beta * FN + smooth)
        FocalTversky = (1 - Tversky) ** gamma

        return FocalTversky


class MyBceLOSS(nn.Module):
    def __init__(self, reduction='mean'):
        super(MyBceLOSS, self).__init__()
        self.reduction = reduction

    def forward(self, inputs, targets):
        eps = 1e-12
        own_loss = -(targets * inputs.clamp(min=eps).log() + (1 - targets) * (1 - inputs).clamp(min=eps).log())
        if self.reduction == 'sum':
            return own_loss.sum()
        else:
            return own_loss.mean()


class MyFocalLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(MyFocalLoss, self).__init__()
        self.reduction = reduction

    def forward(self, inputs, targets, beta=0.6, gamma=1):
        eps = 1e-12
        own_loss = -(beta * targets * (1-inputs).pow(gamma) * inputs.clamp(min=eps).log() + (1-beta) * (1 - targets) * inputs.pow(gamma) * (1 - inputs).clamp(min=eps).log())
        if self.reduction == 'sum':
            return own_loss.sum()
        else:
            return own_loss.mean()


class MyWeightedBceLOSS(nn.Module):
    def __init__(self, reduction='mean'):
        super(MyWeightedBceLOSS, self).__init__()
        self.reduction = reduction

    def forward(self, inputs, targets, beta, ypsilon):
        eps = 1e-12
        own_loss = -(beta * targets * inputs.clamp(min=eps).log() + (1-beta) * (1 - targets) * (1 - inputs).clamp(min=eps).log())
        if self.reduction == 'sum':
            return own_loss.sum()
        else:
            return own_loss.mean()

class MyWeighted2BceLOSS(nn.Module):
    def __init__(self, reduction='mean'):
        super(MyWeighted2BceLOSS, self).__init__()
        self.reduction = reduction

    def forward(self, inputs, targets, weight=1.2):
        mod_inputs = torch.add(1, - inputs)
        mod_targets = torch.add(1, - targets)
        my_bce = - torch.mul(weight, torch.mul(targets, torch.log(inputs))) - torch.mul(mod_targets,
                                                                                        torch.log(mod_inputs))
        if self.reduction == 'sum':
            return torch.sum(my_bce)
        else:
            return torch.mean(my_bce)


class MyBalancedWeightedBceLOSS(nn.Module):
    def __init__(self, reduction='mean'):
        super(MyBalancedWeightedBceLOSS, self).__init__()
        self.reduction = reduction

    def forward(self, inputs, targets, weight=0.8):
        mod_inputs = torch.add(1, - inputs)
        mod_targets = torch.add(1, - targets)
        mod_weight = torch.add(1, - weight)
        my_bce = - torch.mul(weight, torch.mul(targets, torch.log(inputs))) - torch.mul(mod_weight,
                                                                                        torch.mul(mod_targets,
                                                                                                  torch.log(
                                                                                                      mod_inputs)))
        if self.reduction == 'sum':
            return torch.sum(my_bce)
        else:
            return torch.mean(my_bce)


class MyAlphaBalancedFocalLOSS(nn.Module):
    def __init__(self, reduction='mean'):
        super(MyAlphaBalancedFocalLOSS, self).__init__()
        self.reduction = reduction

    def forward(self, inputs, targets, alpha=0.5, gamma=0.5):
        mod_inputs = torch.add(1, - inputs)
        mod_targets = torch.add(1, - targets)
        mod_alpha = torch.add(1, - alpha)
        pow_input_1 = torch.pow(inputs, gamma)
        pow_input_2 = torch.pow(mod_inputs, gamma)
        my_focal_1 = - torch.mul(pow_input_2, torch.mul(alpha, torch.mul(targets, torch.log(inputs))))
        my_focal_2 = - torch.mul(pow_input_1, torch.mul(mod_alpha, torch.mul(mod_targets, torch.log(mod_inputs))))
        focal = my_focal_1 + my_focal_2
        if self.reduction == 'sum':
            return torch.sum(focal)
        else:
            return torch.mean(focal)

class MyMixedLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super(MyMixedLoss, self).__init__()
        self.alpha = alpha
        self.complement_alpha = 1 - alpha
        self.bce = MyFocalLoss()
        self.dice = MyLogDiceLoss()

    def forward(self, inputs, targets):
        loss = self.alpha * self.bce(inputs, targets) + self.complement_alpha * self.dice(inputs, targets)
        return loss
