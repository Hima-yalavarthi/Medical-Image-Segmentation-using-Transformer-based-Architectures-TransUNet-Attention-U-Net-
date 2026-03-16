import torch
import numpy as np

def get_metrics(predict, target, threshold=0.5):
    predict = torch.sigmoid(predict)
    predict = (predict > threshold).float()
    target = (target > threshold).float()
    
    predict = predict.view(-1)
    target = target.view(-1)
    
    tp = (predict * target).sum().item()
    fp = (predict * (1 - target)).sum().item()
    fn = ((1 - predict) * target).sum().item()
    tn = ((1 - predict) * (1 - target)).sum().item()
    
    dice = (2 * tp) / (2 * tp + fp + fn + 1e-6)
    iou = tp / (tp + fp + fn + 1e-6)
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    
    return {
        "dice": dice,
        "iou": iou,
        "precision": precision,
        "recall": recall
    }
