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
Either:

- place your token at `~/.kaggle/kaggle.json`, or
- set the `KAGGLE_USERNAME` and `KAGGLE_KEY` environment variables.

Then download the dataset:

```bash
python src/data/download_dataset.py
```

## Dataset

<!-- Filled in after running src/data/download_dataset.py and src/data/eda.py -->

Placeholder — will be replaced with real numbers, class distribution, and
sample visualizations once the download and EDA scripts have been run.

**Source:** [simuletic/cctv-shoplifting-detection-dataset-yolo-and-vlm](https://www.kaggle.com/datasets/simuletic/cctv-shoplifting-detection-dataset-yolo-and-vlm)
on Kaggle, created by **Simuletic**.

## License / Attribution

Dataset attribution: Simuletic, via Kaggle (see link above). This project
uses the dataset for non-commercial research/portfolio purposes.
