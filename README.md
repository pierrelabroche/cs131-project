# CS131 Final Project — Retinal Vessel Segmentation

Hnin Yupar Mon & Pierre Labroche

## Setup

```bash
conda env create -f environment.yml
conda activate cs131_project
```

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

```bash
python test_preprocessing.py
```
