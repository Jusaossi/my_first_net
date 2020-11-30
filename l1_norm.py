
import torch

def calculate_l1(predicts, targets):
    subtract_inputs = torch.add(targets, - predicts)
    subtract_inputs_abs = torch.abs(subtract_inputs)
    return torch.mean(subtract_inputs_abs).item()

def calculate_teeth_pixels(predicts, targets):
    elements_count = targets.numel()
    sum_of_teeth_target = torch.sum(targets).item().__int__()
    sum_of_no_teeth_target = elements_count - sum_of_teeth_target
    teeth_prediction = (predicts > 0.5)
    correct_teeth_quess = teeth_prediction * targets

    no_teeth_in_target = torch.ones_like(targets)
    no_teeth_in_target = no_teeth_in_target * (targets <= 0.5)
    no_teeth_prediction = (predicts <= 0.5)
    correct_no_teeth_quess = no_teeth_prediction * no_teeth_in_target

    sum_of_correct_teeth_predictions = torch.sum(correct_teeth_quess).item().__int__()
    sum_of_correct_no_teeth_predictions = torch.sum(correct_no_teeth_quess).item().__int__()

    return sum_of_correct_teeth_predictions, sum_of_teeth_target, sum_of_correct_no_teeth_predictions, sum_of_no_teeth_target


def calculate_my_metrics(inputs, targets):
    # flatten label and prediction tensors
    inputs = inputs.view(-1)
    targets = targets.view(-1)

    # True Positives, False Positives & False Negatives
    TP = (inputs * targets).sum()
    FP = ((1 - targets) * inputs).sum()
    FN = (targets * (1 - inputs)).sum()
    TN = ((1 - targets) * (1 - inputs)).sum()

    recall = TP / (TP + FN)
    true_negative_rate = TN / (TN + FP)
    precision = TP / (TP + FP)
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    f1_score = (2 * precision * recall) / (precision + recall)
    return recall.item(), true_negative_rate.item(), precision.item(), accuracy.item(), f1_score.item()

