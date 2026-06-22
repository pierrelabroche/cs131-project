# CS131 Final Project — Retinal Vessel Segmentation

**Hnin Yupar Mon & Pierre Labroche**

Classical computer vision methods for segmenting blood vessels in retinal fundus images, evaluated on the DRIVE and STARE datasets.

## Setup

```bash
conda env create -f environment.yml
conda activate cs131_project
```

`environment.yml` pins the full environment. Key dependencies: Python 3.11, numpy, opencv-python, scikit-image, scipy, matplotlib, pandas.

## Data

Download and place in `data/` folder:

- **DRIVE**: https://www.kaggle.com/datasets/andrewmvd/drive-digital-retinal-images-for-vessel-extraction
- **STARE**: https://cecas.clemson.edu/~ahoover/stare/probing/index.html
  - https://cecas.clemson.edu/~ahoover/stare/probing/stare-images.tar
  - https://cecas.clemson.edu/~ahoover/stare/probing/labels-ah.tar
  - https://cecas.clemson.edu/~ahoover/stare/probing/labels-vk.tar

Expected structure:
```text
data/
├── DRIVE/
│   └── training/
│       ├── images/          # .tif files, e.g. 21_training.tif
│       ├── 1st_manual/      # ground truth .gif, e.g. 21_manual1.gif
│       └── mask/            # FOV masks .gif, e.g. 21_training_mask.gif
└── STARE/
    ├── stare-images/        # .ppm.gz files
    ├── labels-ah/           # annotator AH labels .ah.ppm.gz
    └── labels-vk/           # annotator VK labels .vk.ppm.gz
```

## Project Structure

```
src/
  preprocessing.py           # Green channel extraction + CLAHE
  canny.py                   # Full Canny edge detector (from scratch, no OpenCV)
  gabor.py                   # Gabor filter bank (Soares et al. 2006, simplified)
  color_threshold.py         # Color thresholding in Lab space
  fusion.py                  # Fusion strategies: avg, max, min, weighted
  evaluate.py                # compute_metrics(), roc_curve(), auc()
  run_drive.py               # Run one method on all 20 DRIVE training images
  run_stare.py               # Run one method on all STARE images
  plot_roc.py                # ROC curves for all methods (DRIVE or STARE)
  summarize_results.py       # Print/save mean metrics table per dataset
  tune_threshold.py          # Grid-search optimal fusion threshold on DRIVE
  caliber_analysis.py        # Thin vs. thick vessel sensitivity analysis
  caliber_distribution.py    # Vessel radius distribution across DRIVE GT masks
  generate_milestone_figures.py  # Produces report figures
outputs/
  <method>/training/         # DRIVE: per-image mask PNGs + metrics CSV
  stare/<method>/            # STARE: per-image mask PNGs + metrics CSV
  roc_auc_per_image_drive.csv
  roc_auc_per_image_stare.csv
results/
  figures/                   # All saved figures (see table below)
  metrics_summary_drive.csv
  metrics_summary_stare.csv
notebooks/explore.ipynb      # Exploration notebook
```

## Running

All scripts must be run from the project root.

### Segmentation

```bash
# DRIVE (20 training images)
python src/run_drive.py --method canny
python src/run_drive.py --method gabor
python src/run_drive.py --method color_threshold
python src/run_drive.py --method fusion_avg        # not used in final project
python src/run_drive.py --method fusion_weighted   # AUC-weighted fusion (best)
python src/run_drive.py --method fusion_max        # not used in final project
python src/run_drive.py --method fusion_min        # not used in final project

# STARE
python src/run_stare.py --method canny
python src/run_stare.py --method fusion_weighted
# ... same method choices as above; add --annotator vk to use VK labels (default: ah)
```

Outputs go to `outputs/<method>/training/` (DRIVE) or `outputs/stare/<method>/` (STARE): PNG masks + metrics CSV per image.

### Analysis & Figures

```bash
# ROC curves (mean ± 1 std band, AUC in legend)
python src/plot_roc.py --dataset drive   # fig in results/figures/roc_curves_drive.png
python src/plot_roc.py --dataset stare   # fig in results/figures/roc_curves_stare.png

# Summary metrics table (Sensitivity, Specificity, Accuracy, F1, AUC)
python src/summarize_results.py --dataset drive
python src/summarize_results.py --dataset stare

# Tune fusion binarization threshold (sweeps 100 thresholds, maximizes mean F1)
python src/tune_threshold.py --method fusion_weighted

# Thin vs. thick vessel sensitivity analysis
python src/caliber_analysis.py --dataset drive   # fig in results/figures/caliber_analysis_drive.png
python src/caliber_analysis.py --dataset stare   # fig in results/figures/caliber_analysis_stare.png

# Vessel radius distribution across DRIVE GT masks
python src/caliber_distribution.py              # fig in results/figures/caliber_distribution.png

# All report figures (figure2, figure3, figure4 variants)
python src/generate_milestone_figures.py
```

## Methods

### Preprocessing (`src/preprocessing.py`)
Extracts the **green channel** (highest vessel contrast in retinal images) and applies **CLAHE** (clipLimit=2.0, tileGridSize=8×8) for local contrast enhancement. Returns `(green, enhanced)`.

### Canny (`src/canny.py`)
Implemented from scratch:
1. Gaussian blur (kernel size = 6σ+1, default σ=1.0)
2. Sobel gradient (magnitude + direction)
3. Non-maximum suppression
4. Hysteresis thresholding (default low=0.05, high=0.15 as fractions of max)

Post-processed with morphological closing (`disk(2)`) + `binary_fill_holes` to convert thin edges into filled vessel regions.

### Gabor (`src/gabor.py`)
Simplified version of [Soares et al. 2006](https://ieeexplore.ieee.org/document/1677727):
- 4 scales: λ ∈ {2, 3, 4, 6} px; σ = 0.56λ; γ = 0.5
- 18 orientations: 0°–170° in 10° steps
- Max response across all (scale, orientation) pairs, normalized to [0, 1]
- Fixed threshold (default=0.1)

### Color Threshold (`src/color_threshold.py`)
Converts RGB -> Lab, inverts the L channel (vessels are dark -> become bright peaks), applies CLAHE, and thresholds the normalized L at 0.5.

### Fusion (`src/fusion.py`)
Combines the normalized [0, 1] response maps of all three base methods:
- `average_fusion_segment` — element-wise mean (threshold=0.253)
- `weighted_fusion_segment` — AUC-weighted average
- `max_fusion_segment` — element-wise max
- `min_fusion_segment` — element-wise min

### Evaluation (`src/evaluate.py`)
- `compute_metrics(pred, gt, fov_mask)` -> sensitivity, specificity, accuracy, F1, TP/TN/FP/FN (within FOV only)
- `roc_curve(response_map, gt, fov_mask)` -> sweeps 100 thresholds [0, 1]
- `auc(fpr, tpr)` -> trapezoidal rule

## Figures

| File | Description |
|------|-------------|
| `figure2_canny.png` | Original / Canny edges / yellow vessel overlay |
| `figure3_mask_vs_gt.png` | Original / ground truth / error map (Canny) |
| `figure4_error_1row.png` | Single-row error maps for all 4 methods (report figure) |
| `figure4_error_2panel.png` | 2-panel: Canny vs. Fusion (weighted) error maps |
| `figure4_error_4panel.png` | 2×2 grid error maps for all 4 methods |
| `roc_curves_drive.png` | Averaged ROC curves on DRIVE (mean ± 1 std, with AUC) |
| `roc_curves_stare.png` | Averaged ROC curves on STARE (mean ± 1 std, with AUC) |
| `caliber_analysis_drive.png` | Thin vs. thick vessel sensitivity per method (DRIVE) |
| `caliber_analysis_stare.png` | Thin vs. thick vessel sensitivity per method (STARE) |
| `caliber_distribution.png` | Vessel radius histogram + cumulative distribution (DRIVE) |
| `generalization_comparison.png` | DRIVE vs. STARE performance comparison |
| `threshold_tuning_fusion_weighted.png` | F1/Sensitivity/Specificity vs. threshold sweep |

Error map colors: green = TP, red = FN, blue = FP, dark gray = TN.
