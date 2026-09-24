# CCTV Shoplifting Detection

A completed computer vision portfolio project comparing two approaches to
detecting shoplifting behavior in CCTV footage:

1. **Classical baseline** — a non-deep-learning motion/SSIM-based approach
   that flags frames with anomalous motion patterns, with no training
   required.
2. **YOLOv8 detector** — fine-tuned on a synthetic CCTV shoplifting dataset
   with YOLO-format pose/keypoint annotations (single "person" class).
3. **VLM captioning** — pairs detections with the dataset's own
   ground-truth scene-description captions for a qualitative read on the
   results.

Built and run locally in VS Code across four steps, then consolidated into
one self-contained Google Colab notebook —
[`notebooks/CCTV_Shoplifting_Detection_Colab.ipynb`](notebooks/CCTV_Shoplifting_Detection_Colab.ipynb) —
that reproduces the entire pipeline end-to-end after mounting Google Drive,
runnable top-to-bottom via **Runtime → Run all**.

## Results at a glance

- **Classical SSIM/ROI baseline:** 130–138 of 145 frames flagged on real
  sample clips. Correctly detects "something in the ROI changed," but
  cannot tell a returned item from a concealed one — see "Baseline" below
  for the concrete evidence.
- **YOLOv8 (`yolov8n`, 30 epochs, fine-tuned):** precision 0.497, recall
  0.445, mAP50 0.283, mAP50-95 0.181 overall. Reliably finds *people*, but
  is markedly less reliable at classifying `shoplifting_person`
  specifically (9/164 correct in the real confusion matrix) — see "Model:
  YOLOv8 fine-tuning" below.
- **Together:** complementary, not redundant — motion detection for
  "something happened here," object detection for "here's who." Neither
  is deployment-ready alone; see "Classical vs. deep learning" and "Future
  work" below.

## Project status

Built locally in VS Code across four steps, then consolidated into one
Colab notebook as the final deliverable (see "Run everything in Google
Colab instead" under Setup).

- [x] Step 1: Repo scaffolding, dataset download, EDA
- [x] Step 2: Classical motion/SSIM baseline
- [x] Step 3: YOLOv8 fine-tuning + VLM captioning + classical-vs-DL comparison
- [x] Step 4: Final Colab notebook combining both approaches — tagged [`v1.0.0`](https://github.com/hrshimpi/cctv-shoplifting-detection/releases/tag/v1.0.0)

## Repository structure

```
.
├── data/                  # Raw dataset, exactly as downloaded (gitignored)
├── notebooks/
│   └── CCTV_Shoplifting_Detection_Colab.ipynb  # the final, self-contained notebook
├── src/
│   ├── data/              # download_dataset.py, eda.py
│   ├── classical_cv/      # motion_baseline.py (headless), select_roi_interactive.py
│   │                       # (local-only GUI), run_baseline_demo.py
│   ├── detection/         # prepare_yolo_data.py, train_yolo.py, evaluate_yolo.py,
│   │                       # infer.py, caption_module.py
│   └── utils/             # dataset_discovery.py (shared discovery logic) and
│                           # compare_pipeline.py (classical vs. YOLO, same video)
├── outputs/               # Generated plots, prepared datasets, weights (gitignored)
│   ├── eda/               # plots from eda.py
│   ├── classical_baseline/ # annotated videos + alert logs from run_baseline_demo.py
│   ├── alerts/            # flagged-frame JPEGs from the classical baseline
│   ├── yolo_dataset/      # 2-class YOLO dataset built by prepare_yolo_data.py
│   ├── weights/           # cached pretrained checkpoints (e.g. yolov8n.pt)
│   ├── yolo_runs/         # train/eval/infer run folders (ultralytics' own layout)
│   └── compare_pipeline/  # combined classical+YOLO side-by-side videos
├── reports/yolo/          # small, real eval artifacts that ARE committed:
│                           # confusion_matrix.png, pr_curve.png, loss_curves.png,
│                           # results.csv, val_metrics_summary.csv
├── LICENSE                # MIT (code only - see "License" below)
├── requirements.txt
└── README.md
```

Trained weights (`*.pt`), the `outputs/yolo_runs/` tree, and the raw
dataset are all gitignored - large and/or fully regenerable by re-running
the scripts. The handful of small plots/CSVs this README quotes numbers
from are copied into `reports/yolo/`, which is *not* gitignored, so the
actual evidence behind the numbers ships with the repo.

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
python src/detection/prepare_yolo_data.py
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

### Train, evaluate, and run the YOLOv8 detector

```bash
python src/detection/train_yolo.py --epochs 30 --imgsz 320 --batch 8 --device cpu
python src/detection/evaluate_yolo.py
python src/detection/infer.py --source outputs/yolo_dataset/images/val
```

`train_yolo.py` fine-tunes a pretrained `yolov8n.pt` (cached to
`outputs/weights/`, not left at the repo root) via `ultralytics`. Model
size, epochs, batch, and image size are all CLI args (or pass a config
dict straight to `train()`) so the same script scales from this CPU run
up to a real GPU without code changes. Runs are written to
`outputs/yolo_runs/<name>/` (ultralytics' own layout).

`evaluate_yolo.py` runs `model.val()` (which saves its own confusion
matrix + PR/F1/P/R curves) and separately reloads the training run's
`results.csv` to plot loss curves - both copied into the committed
`reports/yolo/` for this README (see "Model: YOLOv8 fine-tuning" below).

`infer.py` accepts an image, a folder of images, or a video as
`--source` and saves annotated output (boxes + class + confidence, drawn
by `ultralytics` itself) under `outputs/yolo_runs/<name>/`.

### Caption demo

```bash
python src/detection/caption_module.py
```

Pairs each sample frame's YOLO detection with the dataset's own
ground-truth VLM caption for that exact frame (see "Dataset" above) -
step 1's EDA already found real captions here, so this loads and pairs
them rather than running a pretrained captioning model. Prints
`YOLO: <class>, <confidence> confidence` next to `Caption: <ground-truth
scene description>` for 6 sample val frames, and logs the same as JSONL
to `outputs/yolo_runs/caption_demo.jsonl`.

### Compare classical vs. YOLO

```bash
python src/utils/compare_pipeline.py
```

Runs both detectors - step 2's classical SSIM/ROI baseline and this
step's fine-tuned YOLOv8 - on the same real clips, frame by frame, and
writes one combined side-by-side annotated video per clip to
`outputs/compare_pipeline/`, plus a short table of how many frames each
one flagged. See "Classical vs. deep learning" below for the real
numbers and what they show.

### Run everything in Google Colab instead

All of the above — dataset, EDA, classical baseline, YOLO training/eval/
inference, captioning, comparison — is also reproducible with **no local
setup at all**, via
[`notebooks/CCTV_Shoplifting_Detection_Colab.ipynb`](notebooks/CCTV_Shoplifting_Detection_Colab.ipynb):

1. Open the notebook in Google Colab (upload it, or open directly from
   GitHub via Colab's "File → Open notebook → GitHub" using this repo's URL).
2. Optionally add two Colab secrets (key icon, left sidebar) before running:
   `KAGGLE_USERNAME` / `KAGGLE_KEY` (skip if you've already cached the
   dataset to Drive from a previous run — see the notebook's "Dataset"
   section).
3. **Runtime → Run all.** The notebook mounts your Drive, clones this repo,
   downloads/caches the dataset, and runs every step end-to-end — with
   trained weights and logs written to
   `/content/drive/MyDrive/cctv_shoplifting_outputs/` so they survive a
   Colab disconnect.

No GitHub token, password, or API key is ever pasted into a cell as plain
text; see the notebook's own "Syncing results back to GitHub" cell for how
to get results out again (Drive sync is the simple default; a Colab-secret
GitHub token is the alternative if you want to push from inside Colab).

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

`prepare_yolo_data.py` turns the raw download above into the dataset
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

## Model: YOLOv8 fine-tuning

Fine-tuned `yolov8n.pt` (the smallest YOLOv8 checkpoint, so this fits a
free Colab GPU - or, as actually run here, a plain CPU) on the 2-class
dataset above: 30 epochs, `imgsz=320`, `batch=8`, `device=cpu`. Full
training took about 18 minutes on a 12-core CPU with no GPU - the same
settings would run in a couple of minutes on a Colab T4.

**Real validation metrics** (130 val images, 275 instances; full CSV and
plots in `reports/yolo/`):

| Class | Images | Instances | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|---|---|
| all | 130 | 275 | 0.497 | 0.445 | 0.283 | 0.181 |
| not_shoplifting_person | 49 | 111 | 0.393 | 0.441 | 0.263 | 0.189 |
| shoplifting_person | 81 | 164 | 0.600 | 0.449 | 0.303 | 0.173 |

Training and validation loss both decrease over the 30 epochs
(`reports/yolo/loss_curves.png`) - train loss drops smoothly, val loss is
noisier (130 images is a small validation set) but trends down too, so
the model is genuinely learning, not memorizing noise.

**The confusion matrix (`reports/yolo/confusion_matrix.png`) tells a more
specific story than the aggregate numbers**: of 164 true `shoplifting_person`
instances, the model correctly predicted only 9. It confused 35 of them
for `not_shoplifting_person` and missed 120 entirely (predicted
background). In other words, the model is much better at noticing *a
person* than at deciding *which* class that person belongs to - which
makes sense for a single-frame detector on this dataset: the visual
difference between "holding an item" and "concealing an item" is often a
matter of hand position and motion *over time*, not something visible in
one static frame. 30 epochs on 326 training images is also a genuinely
small fine-tune - more data or more epochs would likely close some of
this gap, but the model shouldn't be expected to have solved single-frame
shoplifting classification from this alone.

**Caption demo** (`caption_module.py`, real output, val frames the model
was not trained on) - pairs each detection with the dataset's own
ground-truth caption for that exact frame:

| Frame | YOLO prediction | Ground-truth caption |
|---|---|---|
| `not_shoplifting4_f0000.png` | `not_shoplifting_person`, 0.35 | "The subject stands in the aisle carrying a white tote bag on their left shoulder, holding a folded black garment in both hands and inspecting it." |
| `shoplifting4_f0081.png` | `not_shoplifting_person`, 0.30 (**misclassified**) | "The subject lowers the garment to waist level, uses both hands to open the top of the white tote bag, and stuffs the folded clothing completely inside the bag." |
| `shoplifting4_f0162.png` | `not_shoplifting_person`, 0.26 (**misclassified**) | "With empty hands, the subject adjusts the tote bag strap on their shoulder, looks briefly down the aisle, and begins to step away from the display rack." |

The `shoplifting4_f0081` example is the actual concealment moment
according to the ground-truth caption, and the model calls it
`not_shoplifting_person` - a concrete instance of the confusion matrix
finding above, not just an abstract number.

## Classical vs. deep learning

`compare_pipeline.py` ran both detectors on the same two real clips
(`shoplifting1`, `not_shoplifting1` - same camera, same table, same
actor; see "Baseline" above):

| Video | Frames | Classical alerts | YOLO-flagged frames | YOLO avg. confidence |
|---|---|---|---|---|
| `shoplifting1` | 145 | 130 | 145 | 0.382 |
| `not_shoplifting1` | 145 | 138 | 145 | 0.441 |

YOLO flags essentially every frame (145/145 in both clips) because a
person is visible almost the entire time and the model reliably detects
*a person* - that part of the task is easy for it. The classical
detector flags fewer frames (130-138/145) because it only reacts once
the ROI's pixels actually change, not just because someone is on screen.

**The real difference is qualitative, not just the counts** - the same
frame (frame 60 of `shoplifting1`, the moment the box gets pushed into
the jacket) side by side in
`outputs/compare_pipeline/shoplifting1_side_by_side.mp4`:
- **Classical (left):** a red box over the now-empty tabletop spot,
  `SSIM=0.555`. It correctly notices *something changed* but has no idea
  what.
- **YOLO (right):** a blue box correctly drawn around the *person*, but
  labeled `not_shoplifting_person 0.49` - it correctly localizes *who*,
  but at this exact moment gets *what they're doing* wrong.

Neither one alone gives a trustworthy verdict: the classical detector
can't tell restocking from theft (see "Baseline" limitations above), and
this fine-tune's class prediction is unreliable on the harder class (see
the confusion matrix above). What each one is actually good at is
different and complementary - motion detection for "something happened
here," and object detection for "here's where the person is" - which is
the practical case for combining both rather than picking one, even
though this model's current per-frame class accuracy isn't yet good
enough to trust on its own.

## Citations

**Dataset:** Simuletic. *CCTV Shoplifting Detection Dataset (YOLO and VLM)*.
Kaggle. https://www.kaggle.com/datasets/simuletic/cctv-shoplifting-detection-dataset-yolo-and-vlm
— synthetic CCTV footage with YOLO bounding-box/pose annotations and
VLM-style scene captions. Used here for non-commercial research/portfolio
purposes; see "Dataset" above for the real structure and numbers.

**Classical baseline technique:** the SSIM/ROI change-detection approach in
`src/classical_cv/motion_baseline.py` (watch a fixed region, flag it when
it visibly changes from a reference) is a well-known, generic OpenCV/
computer-vision technique for simple change detection — it isn't adapted
from one specific article or paper, so no single citation applies here.

## License

Code in this repository is licensed under the [MIT License](LICENSE).

The dataset itself is **not** covered by this license — it remains subject
to Simuletic's terms on Kaggle (see the citation above) and is used here
for non-commercial research/portfolio purposes only.

## Future work

- **More training data and epochs.** 456 frames from 8 videos and a
  30-epoch CPU fine-tune are both small; more synthetic (or real) footage
  and a longer/GPU training run would likely improve `shoplifting_person`
  accuracy specifically.
- **Temporal context for the YOLO detector.** The confusion matrix finding
  — the model struggles to tell "holding" from "concealing" in a single
  frame — suggests a multi-frame or video-level model (even a simple
  classifier on top of per-frame detections) could close a real gap that
  more single-frame data alone might not.
- **Real (non-synthetic) validation footage**, even a small hand-labeled
  set, to get an honest read on the sim-to-real gap flagged throughout
  this README.
- **A single combined alerting pipeline** that actually fuses the
  classical and YOLO signals (e.g. only alert when both agree, or use the
  classical detector as a cheap pre-filter before running YOLO) instead of
  running them side by side for comparison only.
- **Multi-camera / multi-ROI support** for the classical baseline, since
  today it's one hand-picked ROI per camera angle.
