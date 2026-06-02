"""
Read per-image metrics CSVs and AUC CSV, then print and save a summary table
with mean ± std for each method.

Output:
    - Printed table to terminal
    - results/metrics_summary.csv
"""

import os
import numpy as np
import pandas as pd
from tabulate import tabulate

METHODS = {
    "Canny":            "outputs/canny/training/canny_training_metrics.csv",
    "Gabor":            "outputs/gabor/training/gabor_training_metrics.csv",
    "Color Threshold":  "outputs/color_threshold/training/color_threshold_training_metrics.csv",
    "Fusion (weighted)":"outputs/fusion_weighted/training/fusion_weighted_training_metrics.csv",
}
AUC_CSV  = "outputs/roc_auc_per_image.csv"
OUT_CSV  = "results/metrics_summary.csv"
METRICS  = ["sensitivity", "specificity", "accuracy", "f1"]


def main():
    auc_df = pd.read_csv(AUC_CSV)

    rows = []
    for display_name, csv_path in METHODS.items():
        if not os.path.exists(csv_path):
            print(f"Missing: {csv_path} — run run_drive.py --method first")
            continue

        df = pd.read_csv(csv_path)

        row = {"Method": display_name}
        for m in METRICS:
            row[m.capitalize()] = f"{df[m].mean():.4f}"

        method_aucs = auc_df[auc_df["method"] == display_name]["auc"]
        if len(method_aucs) > 0:
            row["AUC"] = f"{method_aucs.mean():.4f} ± {method_aucs.std():.4f}"
        else:
            row["AUC"] = "N/A"

        rows.append(row)

    print("\n=== Results Summary — DRIVE Training Set (mean over 20 images) ===\n")
    print(tabulate(rows, headers="keys", tablefmt="outline"))

    os.makedirs("results", exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    print(f"\nSaved {OUT_CSV}")


if __name__ == "__main__":
    main()
