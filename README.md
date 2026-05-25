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
# Canny — evaluates all 20 DRIVE training images, saves masks + metrics CSV
python src/run_canny_drive.py

# Milestone figures (Figures 1–3)
python src/generate_milestone_figures.py
```
