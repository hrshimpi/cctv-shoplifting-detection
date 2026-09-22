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
- [x] Step 2: Classical motion/SSIM baseline
- [ ] Step 3: YOLOv8 fine-tuning
- [ ] Step 4: Final Colab notebook combining both approaches

## Repository structure

```
.
├── data/                  # Raw dataset, exactly as downloaded (gitignored)
├── notebooks/             # Jupyter notebooks (EDA, experiments, final Colab notebook)
├── src/
│   ├── data/              # download_dataset.py, eda.py
│   ├── classical_cv/      # motion_baseline.py (headless), select_roi_interactive.py
│   │                       # (local-only GUI), run_baseline_demo.py
│   ├── detection/         # prepare_yolo_dataset.py, (later) training/inference
│   └── utils/             # dataset_discovery.py — shared discovery logic, used by
│                           # both src/data/ and src/detection/ so nothing re-guesses
│                           # the dataset's structure independently
├── outputs/               # Generated plots, prepared datasets, weights (gitignored)
│   ├── eda/               # plots from eda.py
│   ├── classical_baseline/ # annotated videos + alert logs from run_baseline_demo.py
│   ├── alerts/            # flagged-frame JPEGs from the classical baseline
│   └── yolo_dataset/      # 2-class YOLO dataset built by prepare_yolo_dataset.py
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

### Run the classical baseline demo

```bash
python src/classical_cv/run_baseline_demo.py
```

Runs the SSIM/ROI motion baseline (step 2 — see "Baseline: classical
motion detection" below) on real clips from `data/CCTV_Shoplifting_Dataset/videos/`,
prints a frame-count / alerts / average-similarity summary for each, and
writes an annotated video + JSONL alert log per clip to
`outputs/classical_baseline/` (gitignored) plus flagged-frame JPEGs to
`outputs/alerts/`. This only needs the raw video download, not the YOLO
dataset prep below.

`src/classical_cv/select_roi_interactive.py` is a separate, local-only
tool (uses `cv2.selectROI`, needs a real display) for picking your own
ROI coordinates by hand on a new camera/clip — it is never imported by
the headless `motion_baseline.py` module.

### Prepare the YOLO training dataset

```bash
python src/detection/prepare_yolo_dataset.py
```

The raw labels are YOLO-*pose* format with a single class (`0` = person,
see "Real structure" below) — they don't encode shoplifting vs. not at
all. This script re-derives that class from the same VLM-labels metadata
`eda.py` uses, strips the pose keypoints, and writes a plain 2-class YOLO
*detection* dataset to `outputs/yolo_dataset/` (gitignored, regenerated
by re-running this script): `images/{train,val}/`, `labels/{train,val}/`,
and a `data.yaml` ready for `ultralytics` training. See "YOLO training
dataset" below for what it actually produced and why the split works the
way it does.

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

## Baseline: classical motion detection

A non-deep-learning baseline (`src/classical_cv/motion_baseline.py`):
watch one fixed Region of Interest (ROI) on a shelf/display case, and
flag any frame whose ROI has visibly changed from a reference, using the
Structural Similarity Index (SSIM) on denoised grayscale crops
(`compute_ssim_score` = grayscale + median blur + `skimage`'s
`structural_similarity`). No training, no labels, no GPU.

**Headless by design.** `motion_baseline.py` never calls `cv2.imshow` or
`cv2.selectROI` — it has to run inside Google Colab later, which has no
display. ROI coordinates come in as a plain `(x, y, w, h)` tuple or a
small JSON config. Picking that ROI is the one step that genuinely needs
a human eye on the footage, so it's split into its own local-only script
(`select_roi_interactive.py`) that *is* allowed to use `cv2.selectROI` -
it is never imported by the headless module.

**Two reference strategies**, both implemented in `MotionROIDetector`:
- `"fixed"` — the first frame's ROI crop, frozen for the whole run. The
  textbook version of this technique. It drifts under lighting changes:
  if ambient light shifts later on, the whole ROI reads as "different"
  from a reference lit under earlier conditions, and every later frame
  false-alarms on nothing but brightness.
- `"rolling"` — the reference is an EMA of the ROI crop, updated after
  every frame that does *not* alert (alert frames are excluded so a real
  change never gets quietly absorbed into the new "normal"). This is the
  deliberate upgrade over the naive fixed version.

**Alerts are cross-platform, not a beep.** Each alert appends
`{timestamp, frame_index, similarity_score, roi}` as one JSON line to an
alert log, and optionally saves the flagged frame as a JPEG to
`outputs/alerts/` - both configurable, both off the moment you don't
pass a path.

### Real run on real footage

`run_baseline_demo.py` picked ROI coordinates by hand off actual frames
(not guessed) — e.g. for the electronics-table scene, cropping candidate
regions and looking at them directly showed the smartphone box sitting
at roughly `x:[178,245], y:[345,405]` in frame 0. Threshold `0.8`, as
specified, run against real clips from the raw download:

| Video | Mode | Frames | Alerts | Avg. similarity | First alert | Recovers before end? |
|---|---|---|---|---|---|---|
| `not_shoplifting1` | fixed | 145 | 138 | 0.552 | frame 7 | No |
| `shoplifting1` | fixed | 145 | 130 | 0.588 | frame 15 | No |
| `shoplifting1` | rolling | 145 | 129 | 0.611 | frame 16 | No |
| `shoplifting2` | fixed | 145 | 137 | 0.416 | frame 8 | No |

This confirms the mechanism works exactly as designed: the detector
stays quiet while the ROI is genuinely unchanged (SSIM 0.92–1.0 for the
first couple of frames) and correctly flags it the moment the tracked
item visibly leaves the ROI a few frames later. The annotated output
video draws the ROI in green while quiet and red the instant it flags -
confirmed visually frame-by-frame (frame 0: green box around the
box-on-table, `SSIM=1.000`; frame 60: red box around the now-empty
tabletop spot, `SSIM=0.555`).

**The most interesting finding wasn't planned - it fell out of the real
numbers above.** `not_shoplifting1` and `shoplifting1` are the *same*
camera, same table, same actor: in `not_shoplifting1` the box is visibly
back on the table by the last frame; in `shoplifting1` it never returns
(concealed). A human watching either clip can tell them apart instantly.
This detector cannot: both alert within the first ~15 frames and *stay*
alerted through the end, "recovers before end?" = No for both. Once the
person moves at all, their exact pose never matches the frozen frame-0
reference closely enough again - whether the item comes back or not.
That is not a bug in this implementation, it is the ceiling of comparing
raw pixels to one fixed reference. See limitations below.

### Real limitations (not hypothetical)

- **A single static ROI has no idea *what* changed, only *that* it did.**
  The concrete evidence: `not_shoplifting1` (item returned) and
  `shoplifting1` (item concealed) produce statistically indistinguishable
  alert patterns - restocking and theft look identical to this detector.
  Telling them apart needs either a smarter reference (per-slot object
  presence, not whole-crop pixel similarity) or a downstream model - which
  is exactly the gap step 3's YOLO detector is meant to help close.
- **Sensitive to anything in the ROI, not just the tracked item.** On the
  `hardware_aisle` ROI (`shoplifting2`), a large share of the alerts are
  driven by the person's own body/arm passing in front of the shelf, not
  the item itself - occlusion by a shopper looks the same as an item
  disappearing.
- **The reference frame has to already show the "normal" state.** These
  clips start mid-interaction, so there's no long quiet lead-in to learn
  from - the "fixed" reference is only ever as good as whatever frame 0
  happened to catch.
- **The rolling/EMA reference can get stuck, too.** It only updates on
  *quiet* frames by design - once a real change makes the detector alert
  continuously (as it does above), the rolling reference stops updating
  entirely and behaves almost identically to the fixed one (129 vs. 130
  alerts on the same clip). It adapts to slow drift, not to a sustained
  changed state.
- **Every camera needs its own manual ROI.** There is no automatic
  discovery here - `select_roi_interactive.py` has to be re-run by a
  person for each new camera angle, and a moved/re-angled camera silently
  invalidates the old ROI coordinates.

## YOLO training dataset

`prepare_yolo_dataset.py` turns the raw download above into the dataset
step 3 (YOLOv8 fine-tuning) will actually train on. What it uses, where
it's stored, and how it works:

- **Input:** the raw `data/CCTV_Shoplifting_Dataset/images/` +
  `labels/` + `VLM-labels/*.json` described above — read-only, never
  modified in place.
- **Output location:** `outputs/yolo_dataset/` — **gitignored**, just
  like `data/` and `outputs/eda/`. It's a derived artifact, fully
  regenerated by re-running the script, so it's never committed; nothing
  downstream should assume it exists without running this script first.
  On this machine it's currently **~233 MB** (456 images duplicated
  across `images/train/` + `images/val/`, plus small `.txt` label files).
- **Class remap:** the raw label's `class_id` is always `0` (it's a
  pose-format "person" box, not a shoplifting/not-shoplifting label).
  This script looks up each frame's source video in `VLM-labels/*.json`
  and rewrites every box in that frame to:
  - `0` = `not_shoplifting_person`
  - `1` = `shoplifting_person`

  The 17 keypoint triplets after the bbox are dropped — this project's
  YOLO half is a plain 2-class detector, not a pose model.
- **Train/val split is by *video*, not by frame.** Frames are sampled
  every 3rd frame from the same 8 source videos, so a per-frame random
  split would put near-duplicate consecutive frames from one video on
  both sides of the split — leaking information into validation and
  making its metrics meaningless. Instead, one whole video per class is
  held out for validation (currently the alphabetically-last video per
  class: `not_shoplifting4`, `shoplifting4`).
- **Real output from the last run:**

  | Split | Images | not_shoplifting_person | shoplifting_person |
  |---|---|---|---|
  | train | 326 | 147 | 179 |
  | val | 130 | 49 | 81 |

  (val is smaller in video *count* but not in frame count here, since
  `shoplifting4` happens to be one of the two longer source videos.)

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
