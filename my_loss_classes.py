import torch
import torch.nn as nn
import torch.nn.functional as F


class MyDiceLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(MyDiceLoss, self).__init__()
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


class MyFocalLOSS(nn.Module):
    def __init__(self, reduction='mean'):
        super(MyFocalLOSS, self).__init__()
        self.reduction = reduction

    def forward(self, inputs, targets, beta, gamma):
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

    def forward(self, inputs, targets, alpha=0.5, gamma=2):
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
        self.bce = MyBceLOSS()
        self.dice = MyDiceLoss()

    def forward(self, inputs, targets):
        loss = self.alpha * self.bce(inputs, targets) + self.complement_alpha * self.dice(inputs, targets)
        return loss
