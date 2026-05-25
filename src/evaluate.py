import numpy as np

def compute_metrics(pred_binary, gt_binary, fov_mask):
    # Restrict to FOV pixels only
    pred = pred_binary[fov_mask > 0].flatten()
    gt   = (gt_binary[fov_mask > 0] > 0).flatten()

    TP = np.sum((pred == 1) & (gt == 1))
    TN = np.sum((pred == 0) & (gt == 0))
    FP = np.sum((pred == 1) & (gt == 0))
    FN = np.sum((pred == 0) & (gt == 1))

    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)
    accuracy    = (TP + TN) / (TP + TN + FP + FN)
    f1          = 2*TP / (2*TP + FP + FN)

    return {
        'sensitivity': sensitivity,
        'specificity': specificity,
        'accuracy':    accuracy,
        'f1':          f1,
        'TP': TP, 'TN': TN, 'FP': FP, 'FN': FN
    }

def roc_curve(response_map, gt_binary, fov_mask):
    # Sweep threshold from 0 to 1
    # Return arrays of sensitivity and (1-specificity)
    thresholds = np.linspace(0, 1, 100)
    sensitivities, one_minus_specs = [], []
    for t in thresholds:
        pred = (response_map > t).astype(int)
        m = compute_metrics(pred, gt_binary, fov_mask)
        sensitivities.append(m['sensitivity'])
        one_minus_specs.append(1 - m['specificity'])
    return np.array(sensitivities), np.array(one_minus_specs)