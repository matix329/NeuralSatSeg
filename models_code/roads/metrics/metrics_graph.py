import torch
import numpy as np
from scipy.stats import spearmanr

def edge_precision(y_pred, y_true, threshold=0.5):
    y_pred_bin = (y_pred > threshold).float()
    tp = ((y_pred_bin == 1) & (y_true == 1)).sum().item()
    fp = ((y_pred_bin == 1) & (y_true == 0)).sum().item()
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0

def edge_recall(y_pred, y_true, threshold=0.5):
    y_pred_bin = (y_pred > threshold).float()
    tp = ((y_pred_bin == 1) & (y_true == 1)).sum().item()
    fn = ((y_pred_bin == 0) & (y_true == 1)).sum().item()
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0

def edge_f1(y_pred, y_true, threshold=0.5):
    p = edge_precision(y_pred, y_true, threshold)
    r = edge_recall(y_pred, y_true, threshold)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

def edge_iou(y_pred, y_true, threshold=0.5):
    y_pred_bin = (y_pred > threshold).float()
    intersection = ((y_pred_bin == 1) & (y_true == 1)).sum().item()
    union = ((y_pred_bin == 1) | (y_true == 1)).sum().item()
    return intersection / union if union > 0 else 0.0

def edge_mae(y_pred, y_true):
    return torch.abs(y_pred - y_true).mean().item()

def edge_rmse(y_pred, y_true):
    return torch.sqrt(torch.mean((y_pred - y_true) ** 2)).item()

def edge_spearman(y_pred, y_true):
    y_pred_np = y_pred.detach().cpu().numpy().flatten()
    y_true_np = y_true.detach().cpu().numpy().flatten()
    if np.std(y_true_np) == 0 or np.std(y_pred_np) == 0:
        return 0.0
    corr, _ = spearmanr(y_pred_np, y_true_np)
    return float(corr) if not np.isnan(corr) else 0.0 