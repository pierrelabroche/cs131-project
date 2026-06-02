"""
Read per-image metrics CSVs and print/save a summary table with mean values.
AUC is reported as mean ± std; all other metrics as mean over 20 images.

Usage:
    python src/summarize_results.py --dataset drive
    python src/summarize_results.py --dataset stare

Output:
    - Printed table to terminal
    - results/metrics_summary_<dataset>.csv
"""

import os
import argparse
import pandas as pd
from tabulate import tabulate

METRICS = ["sensitivity", "specificity", "accuracy", "f1"]

DRIVE_METHODS = {
    "Canny":             "outputs/canny/training/canny_training_metrics.csv",
    "Gabor":             "outputs/gabor/training/gabor_training_metrics.csv",
    "Color Threshold":   "outputs/color_threshold/training/color_threshold_training_metrics.csv",
    "Fusion (weighted)": "outputs/fusion_weighted/training/fusion_weighted_training_metrics.csv",
}

STARE_METHODS = {
    "Canny":             "outputs/stare/canny/canny_stare_metrics.csv",
    "Gabor":             "outputs/stare/gabor/gabor_stare_metrics.csv",
    "Color Threshold":   "outputs/stare/color_threshold/color_threshold_stare_metrics.csv",
    "Fusion (weighted)": "outputs/stare/fusion_weighted/fusion_weighted_stare_metrics.csv",
}

AUC_CSV = "outputs/roc_auc_per_image_{dataset}.csv"


def build_rows(methods, auc_df):
    rows = []
    for display_name, csv_path in methods.items():
        if not os.path.exists(csv_path):
            print(f"Missing: {csv_path}")
            continue

        df = pd.read_csv(csv_path)
        row = {"Method": display_name}
        for m in METRICS:
            row[m.capitalize()] = f"{df[m].mean():.4f}"

        if auc_df is not None:
            method_aucs = auc_df[auc_df["method"] == display_name]["auc"]
            row["AUC"] = f"{method_aucs.mean():.4f} ± {method_aucs.std():.4f}" if len(method_aucs) > 0 else "N/A"
        else:
            row["AUC"] = "N/A"

        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["drive", "stare"], default="drive")
    args = parser.parse_args()

    is_drive = args.dataset == "drive"
    methods  = DRIVE_METHODS if is_drive else STARE_METHODS
    label    = "DRIVE Training Set" if is_drive else "STARE Dataset"

    auc_path = AUC_CSV.format(dataset=args.dataset)
    auc_df   = pd.read_csv(auc_path) if os.path.exists(auc_path) else None

    rows = build_rows(methods, auc_df)

    print(f"\n=== Results Summary — {label} (mean over 20 images) ===\n")
    print(tabulate(rows, headers="keys", tablefmt="outline"))

    os.makedirs("results", exist_ok=True)
    out_csv = f"results/metrics_summary_{args.dataset}.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"\nSaved {out_csv}")


if __name__ == "__main__":
    main()
