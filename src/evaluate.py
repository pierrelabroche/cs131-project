import numpy as np
from scipy.ndimage import distance_transform_edt

def compute_metrics(pred_binary, gt_binary, fov_mask):
    # Restrict to FOV pixels only
    pred = pred_binary[fov_mask > 0].flatten()
    gt   = (gt_binary[fov_mask > 0] > 0).flatten()

    TP = np.sum((pred == 1) & (gt == 1))
    TN = np.sum((pred == 0) & (gt == 0))
    FP = np.sum((pred == 1) & (gt == 0))
    FN = np.sum((pred == 0) & (gt == 1))

    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0
    accuracy    = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0.0
    f1          = 2*TP / (2*TP + FP + FN) if (2*TP + FP + FN) > 0 else 0.0

    return {
        'sensitivity': sensitivity,
        'specificity': specificity,
        'accuracy':    accuracy,
        'f1':          f1,
        'TP': TP, 'TN': TN, 'FP': FP, 'FN': FN
    }

def roc_curve(response_map, gt_binary, fov_mask, n_thresholds=200):
    """
    Sweep thresholds over response_map and return TPR and FPR arrays.

    Returns:
        tpr: sensitivity at each threshold (descending threshold order)
        fpr: 1-specificity at each threshold
    """
    vals = response_map[fov_mask > 0].flatten()
    thresholds = np.unique(np.percentile(vals, np.linspace(100, 0, n_thresholds)))[::-1]
    tpr_list, fpr_list = [], []
    for t in thresholds:
        pred = (response_map >= t).astype(int)
        m = compute_metrics(pred, gt_binary, fov_mask)
        tpr_list.append(m['sensitivity'])
        fpr_list.append(1 - m['specificity'])
    return np.array(tpr_list), np.array(fpr_list)


def auc(fpr, tpr):
    """Compute area under the ROC curve using the trapezoidal rule."""
    order = np.argsort(fpr)
    return np.trapezoid(tpr[order], fpr[order])


def split_by_caliber(gt_binary, radius_threshold=1.5):
    """
    Split ground truth vessel mask into thin and thick subsets using the
    distance transform. Each vessel pixel's distance to the nearest background
    pixel approximates the local vessel radius at that point.

    Inputs:
        gt_binary:        (H, W) binary ground truth mask (0/1 or 0/255)
        radius_threshold: pixels with distance <= this are 'thin' (default 3px)
    Outputs:
        thin_mask:  (H, W) uint8 — vessel pixels with radius <= radius_threshold
        thick_mask: (H, W) uint8 — vessel pixels with radius >  radius_threshold
    """
    dist = distance_transform_edt(gt_binary > 0)
    thin_mask  = ((gt_binary > 0) & (dist <= radius_threshold)).astype(np.uint8)
    thick_mask = ((gt_binary > 0) & (dist >  radius_threshold)).astype(np.uint8)
    return thin_mask, thick_mask