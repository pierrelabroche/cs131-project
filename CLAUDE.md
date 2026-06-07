# CS131 Final Project — Retinal Vessel Segmentation

**Authors:** Hnin Yupar Mon & Pierre Labroche  
**Course:** CS131 (Computer Vision)

## Goal

Segment blood vessels in retinal fundus images using classical (non-deep-learning) computer vision methods. Three methods are implemented and compared on the DRIVE dataset.

## Environment

```bash
conda env create -f environment.yml
conda activate cs131_project
```

Conda env name: `cs131_project`  
Python: 3.11 | Key libs: numpy, opencv-python, scikit-image, scipy, matplotlib, pandas

## Data

Not committed. Must be downloaded and placed in `data/`:

```
data/
├── DRIVE/
│   └── training/
│       ├── images/          # .tif files, named like 21_training.tif
│       ├── 1st_manual/      # ground truth .gif, named like 21_manual1.gif
│       └── mask/            # FOV masks .gif, named like 21_training_mask.gif
└── STARE/
    ├── stare-images/
    ├── labels-ah/
    └── labels-vk/
```

DRIVE: https://www.kaggle.com/datasets/andrewmvd/drive-digital-retinal-images-for-vessel-extraction  
STARE: https://cecas.clemson.edu/~ahoover/stare/probing/index.html

## Project Structure

```
src/
  preprocessing.py          # Green channel extraction + CLAHE
  canny.py                  # Full Canny edge detector (from scratch, no OpenCV)
  gabor.py                  # Gabor filter bank (Soares et al. 2006, simplified)
  color_threshold.py        # Color thresholding in Lab space
  evaluate.py               # compute_metrics(), roc_curve(), auc()
  fusion.py                 # Average fusion of Canny + Gabor + Color response maps
  run_drive.py              # Main runner: one method, all 20 DRIVE training images
  run_canny_drive.py        # (legacy canny-only runner)
  generate_milestone_figures.py  # Produces results/figures/figure{2,3,4}*.png
  plot_roc.py               # ROC curves for all 4 methods; outputs roc_curves.png
outputs/
  <method>/training/        # Per-image mask PNGs + metrics CSV
results/figures/            # Saved comparison figures
notebooks/explore.ipynb     # Exploration notebook
```

## How to Run

All scripts must be run from the project root:

```bash
# Run a method on all 20 DRIVE training images
python src/run_drive.py --method canny
python src/run_drive.py --method gabor
python src/run_drive.py --method color_threshold
python src/run_drive.py --method fusion_avg    # average fusion
python src/run_drive.py --method fusion_max   # max fusion (OR)
python src/run_drive.py --method fusion_min   # min fusion (AND)

# Generate milestone figures (figures 2, 3, 4)
python src/generate_milestone_figures.py

# Generate ROC curves for all 4 methods
python src/plot_roc.py
```

Outputs go to `outputs/<method>/training/` (PNG masks + `<method>_training_metrics.csv`).  
Figures go to `results/figures/`.

## Methods

### Preprocessing (`src/preprocessing.py`)
- Extracts the **green channel** (vessels are most visible here in retinal images)
- Applies **CLAHE** (clipLimit=2.0, tileGridSize=8×8) for local contrast enhancement
- Returns `(green, enhanced)` tuple

### Canny (`src/canny.py`)
Fully implemented from scratch (no `cv2.Canny`):
1. Gaussian blur (kernel size = 6σ+1, default σ=1.0)
2. Sobel gradient (magnitude + direction)
3. Non-maximum suppression
4. Hysteresis thresholding (default low=0.05, high=0.15 as ratios of max)

In `run_drive.py`, Canny edges are post-processed with morphological closing (`disk(2)`) + `binary_fill_holes` to convert thin edges into filled vessel regions.

### Gabor (`src/gabor.py`)
Simplified version of Soares et al. 2006:
- 4 scales: λ ∈ {2, 3, 4, 6} pixels; σ = 0.56λ; γ = 0.5
- 18 orientations: 0°–170° in 10° steps
- Max response across all (scale, orientation) combinations
- Normalized to [0,1]; binary mask via fixed threshold (default=0.1)
  - Note: Otsu over-thresholds because vessels are ~12% of pixels

### Color Threshold (`src/color_threshold.py`)
- Convert RGB → Lab color space
- Invert L channel (vessels are dark → become bright peaks)
- Apply CLAHE
- Threshold normalized L at 0.5 (default)

### Evaluation (`src/evaluate.py`)
- `compute_metrics(pred, gt, fov_mask)` → sensitivity, specificity, accuracy, F1, TP/TN/FP/FN
- Metrics are computed only within the FOV mask
- `roc_curve(response_map, gt, fov_mask)` → sweeps 100 thresholds [0,1]
- `auc(fpr, tpr)` → area under the ROC curve via the trapezoidal rule

### Fusion (`src/fusion.py`)
Three fusion strategies over the normalized [0,1] response maps of all base methods:
- `_get_response_maps(img, enhanced)` → (canny_response, gabor_response, color_response)
- `average_fusion_segment(img, enhanced, threshold=0.5)` — element-wise mean; balanced recall/precision
- `max_fusion_segment(img, enhanced, threshold=0.5)` — element-wise max (logical OR); maximizes recall
- `min_fusion_segment(img, enhanced, threshold=0.5)` — element-wise min (logical AND); maximizes precision

### ROC Curves (`src/plot_roc.py`)
Generates averaged ROC curves across all 20 DRIVE training images for all four methods:
- 200 thresholds per image; curves interpolated onto a 500-point shared FPR grid
- Plots mean ± 1 std band per method; reports AUC in the legend
- Output: `results/figures/roc_curves.png`

## Figures Generated

| File | Description |
|------|-------------|
| `figure2_canny.png` | Original / Canny edges / Yellow overlay |
| `figure3_mask_vs_gt.png` | Original / Ground truth / Error map (Canny) |
| `figure4_method_comparison.png` | Error maps for all 3 methods side-by-side |
| `roc_curves.png` | Averaged ROC curves (mean ± 1 std) for all 4 methods with AUC |

Error map colors: green=TP, red=FN, blue=FP, dark gray=TN

## Notes

- `run_canny_drive.py` is a legacy script superseded by `run_drive.py --method canny`
- The Gabor implementation intentionally does NOT use OpenCV's `getGaborKernel`; kernels are built manually with `np.meshgrid` and `cv2.filter2D` is used only for the convolution step
- Scripts use relative paths so they must be run from the project root
