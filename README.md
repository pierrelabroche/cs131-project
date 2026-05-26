# CS131 Final Project — Retinal Vessel Segmentation

Hnin Yupar Mon & Pierre Labroche

## Setup

```bash
conda env create -f environment.yml
conda activate cs131_project
```

`environment.yml` pins the full environment. Key dependencies: Python 3.11, numpy, opencv-python, scikit-image, scipy, matplotlib, pandas.

## Data

Download and place in `data/` folder:
- DRIVE: https://www.kaggle.com/datasets/andrewmvd/drive-digital-retinal-images-for-vessel-extraction
- STARE: https://cecas.clemson.edu/~ahoover/stare/probing/index.html
- - https://cecas.clemson.edu/~ahoover/stare/probing/stare-images.tar
- - https://cecas.clemson.edu/~ahoover/stare/probing/labels-ah.tar
- - https://cecas.clemson.edu/~ahoover/stare/probing/labels-vk.tar

Expected structure:
```text
data/
├── DRIVE/
│   └── training/
│       ├── images/
│       ├── 1st_manual/
│       └── mask/
└── STARE/
    ├── stare-images/
    ├── labels-ah/
    └── labels-vk/
```

## Running

Run all scripts from the project root:

```bash
# Run a segmentation method on all 20 DRIVE training images
# Choices: canny, gabor, color_threshold
python src/run_drive.py --method canny
python src/run_drive.py --method gabor
python src/run_drive.py --method color_threshold

# Generate milestone figures (preprocessing + per-method error map comparison)
python src/generate_milestone_figures.py
```

Outputs are saved to `outputs/<method>/training/` (masks + metrics CSV) and `results/figures/`.
