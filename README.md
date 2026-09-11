# CCTV Shoplifting Detection

A computer vision portfolio project comparing two approaches to detecting
shoplifting behavior in CCTV footage:

1. **Classical baseline** — a non-deep-learning motion/SSIM-based approach
   that flags frames with anomalous motion patterns, with no training
   required.
2. **YOLOv8 detector** — fine-tuned on a synthetic CCTV shoplifting dataset
   with YOLO-format bounding box annotations.

The final deliverable is a single self-contained Google Colab notebook
(run via Google Drive mount) that reproduces both pipelines end-to-end.

## Project status

Working locally in VS Code with this GitHub repo through the development
steps; the project will be consolidated into one Colab notebook as the
final deliverable.

- [x] Step 1: Repo scaffolding, dataset download, EDA
- [ ] Step 2: Classical motion/SSIM baseline
- [ ] Step 3: YOLOv8 fine-tuning
- [ ] Step 4: Final Colab notebook combining both approaches

## Repository structure

```
.
├── data/                  # Dataset (gitignored — populated by download script)
├── notebooks/             # Jupyter notebooks (EDA, experiments, final Colab notebook)
├── src/
│   ├── data/              # Dataset download + EDA scripts
│   ├── classical_cv/      # Motion/SSIM baseline
│   ├── detection/         # YOLOv8 training/inference
│   └── utils/             # Shared helpers
├── outputs/               # Generated plots, weights, results (gitignored)
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

pip install -r requirements.txt
```

### Kaggle API credentials

The dataset download uses `kagglehub`, which needs a Kaggle API token.

1. Log in to Kaggle → click your profile picture → **Settings**.
2. Under **API**, click **Create New Token**. This downloads `kaggle.json`.
3. Place it at `~/.kaggle/kaggle.json` (on Windows: `C:\Users\<you>\.kaggle\kaggle.json`),
   *or* set the `KAGGLE_USERNAME` and `KAGGLE_KEY` environment variables to
   the `username` / `key` values from that file.

### Download the dataset

```bash
python src/data/download_dataset.py
```

This downloads the dataset via `kagglehub`, mirrors it into `data/`
(gitignored — never committed), prints the full folder tree, and prints
the contents of any `data.yaml` / `classes.txt` / caption-annotation JSON
it finds. Nothing about the dataset's structure or class list is
hardcoded elsewhere in this project — everything downstream reads it from
whatever config actually exists in the download.

### Run the EDA

```bash
python src/data/eda.py
```

This discovers the class list and train/val/test split directories from
the config found above, then writes to `outputs/eda/` (gitignored):
`class_distribution.png`, a 6-9 image `sample_grid.png` with YOLO boxes
drawn on, and `resolution_histogram.png`. If a VLM caption/annotation file
is present, it prints a few example image/caption pairs to the console.

## Dataset

**Source:** [simuletic/cctv-shoplifting-detection-dataset-yolo-and-vlm](https://www.kaggle.com/datasets/simuletic/cctv-shoplifting-detection-dataset-yolo-and-vlm)
on Kaggle, created by **Simuletic**.

> **Status:** the download and EDA scripts above are implemented and have
> been validated against a synthetic mock dataset with the same YOLO +
> VLM-caption structure, but have not yet been run against the real
> Kaggle dataset in this environment (no Kaggle API token was available
> when this repo was set up). Run the two commands above locally, then
> replace this note with the real per-split image counts, class
> distribution, resolution histogram, and 3-5 example VLM caption pairs
> that `eda.py` prints/plots.

**Known limitation — sim-to-real gap:** this dataset is **synthetic**
(simulated CCTV footage), not real store surveillance recordings. A
detector trained purely on it may not transfer cleanly to real CCTV
footage, which has different camera noise, compression artifacts,
lighting, and behavior distributions. This gap should be treated as a
limitation of the project, not something the classical or YOLO baseline
is expected to fully overcome.

## License / Attribution

Dataset attribution: Simuletic, via Kaggle (see link above). This project
uses the dataset for non-commercial research/portfolio purposes.
