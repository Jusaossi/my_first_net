
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

