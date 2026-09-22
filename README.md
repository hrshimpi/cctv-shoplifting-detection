# CCTV Shoplifting Detection

A computer vision portfolio project comparing two approaches to detecting
shoplifting behavior in CCTV footage:

1. **Classical baseline** — a non-deep-learning motion/SSIM-based approach
   that flags frames with anomalous motion patterns, with no training
   required.
2. **YOLOv8 detector** — fine-tuned on a synthetic CCTV shoplifting dataset
   with YOLO-format pose/keypoint annotations (single "person" class).

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

This dataset ships no `data.yaml`/`classes.txt` and no train/val/test
split, so the script discovers a class signal from whatever it actually
finds instead (see "Real structure" below), then writes to `outputs/eda/`
(gitignored): `class_distribution.png`, a 6-9 image `sample_grid.png` with
YOLO boxes drawn on, and `resolution_histogram.png`. It also prints
example image/caption pairs pulled from the per-video VLM metadata.

## Dataset

**Source:** [simuletic/cctv-shoplifting-detection-dataset-yolo-and-vlm](https://www.kaggle.com/datasets/simuletic/cctv-shoplifting-detection-dataset-yolo-and-vlm)
on Kaggle, created by **Simuletic**.

**Real structure**, as discovered by actually running `download_dataset.py`
and `eda.py` against the download (there is no `data.yaml` or
`classes.txt` anywhere in it — nothing about this layout is hardcoded
elsewhere in the project; every script reads it at runtime):

```
CCTV_Shoplifting_Dataset/
├── images/            456 flat PNG frames, "{video}_f{frame:04d}.png"
├── labels/             matching YOLO-pose .txt labels (see below)
├── VLM-labels/         one JSON per source video (8 total) — not a
│                       single shared caption file
├── annotated_images/   pre-rendered visualizations (not used here)
└── annotated_videos/, videos/   source + annotated MP4s (not used here)
```

- **456 total sampled frames**, all **544×544 px** (uniform across every
  single image), sampled every 3rd frame from **8 short synthetic
  surveillance videos** (4 "shoplifting", 4 "not_shoplifting"). There is
  **no train/val/test split** — it ships as one flat pool.
- **Class distribution** — per-frame sequence label, read from each
  video's `is_anomaly_sequence` flag in `VLM-labels/*.json`:
  **260 shoplifting frames / 196 not_shoplifting frames**.
- **Labels are YOLO *pose* format, not plain bounding boxes**: each line
  is `class_id xc yc w h` followed by **17 COCO-keypoint `(x, y,
  visibility)` triplets** for that detected person. `class_id` is always
  `0` — a single detection class (person), not a multi-class problem.
  Across the dataset there are **771 person detections** in 456 frames
  (some frames have up to 4 people).
- **VLM captions are not one shared file** — each of the 8 source videos
  has its own `VLM-labels/<video>_vlm_meta.json`, with a global scene
  description plus 3 chronological segments (setup / transition /
  resolution), each carrying its own `frame_range` and
  `scene_description`. Real example pairs (frame → caption):

  | Frame | Label | Caption |
  |---|---|---|
  | `not_shoplifting1_f0072.png` | not_shoplifting | "The subject leans forward slightly, lowering the box with both hands, and places it flat onto the display table next to the other devices." |
  | `shoplifting1_f0072.png` | shoplifting | "The subject swiftly moves the box upward toward their chest, using both hands to push the box directly inside the front opening of their plaid jacket." |
  | `shoplifting2_f0072.png` | shoplifting | "The subject swiftly lowers their right arm, pushing the square box directly down into the front kangaroo pouch of their gray hoodie." |

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
