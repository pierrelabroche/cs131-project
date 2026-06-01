import numpy as np

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

def roc_curve(response_map, gt_binary, fov_mask, n_thresholds=100):
    """
    Sweep thresholds over response_map and return TPR and FPR arrays.

    Returns:
        tpr: sensitivity at each threshold (descending threshold order)
        fpr: 1-specificity at each threshold
    """
    thresholds = np.linspace(1, 0, n_thresholds)
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
    return np.trapz(tpr[order], fpr[order])