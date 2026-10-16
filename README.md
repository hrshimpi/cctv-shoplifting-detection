# 3D-Pose-Assisted Temporal Anomaly Detection for Shoplifting in Retail Surveillance

Course project for CSE 6367. Team: Harsha Prasad, Himanshu Shimpi.

Status: step 1 of 5 (repo scaffolding, dataset verification, pose
preprocessing pipeline). See "Project status" below.

## Overview

This project studies whether shoplifting behavior can be detected from
2D/3D human pose sequences in retail CCTV footage, rather than from raw
pixels. The pipeline under development is:

```
RetailS 2D pose -> MotionBERT -> 3D pose -> temporal anomaly detector
```

with an external-validation path intended to test the same trained
detector on a different data source:

```
UCF-Crime raw video -> 2D pose detector -> MotionBERT -> 3D pose -> same temporal anomaly detector
```

Working from pose sequences instead of raw video is motivated by privacy
(no raw pixels/identities need to be stored or processed downstream) and
by the hypothesis that shoplifting-relevant motion (concealment gestures,
dwell time, approach/retreat patterns) is a temporal signal that a
pose-sequence model can learn more directly than a frame-level image
classifier can.

## Project status

- [x] Step 1: Repo scaffolding, RetailS dataset verification, pose
      preprocessing pipeline, external-video package integration
- [ ] Step 2: (not started)
- [ ] Step 3: MotionBERT integration (2D -> 3D pose lifting)
- [ ] Step 4: Temporal anomaly detector, training/evaluation
- [ ] Step 5: Colab notebook consolidating the full pipeline

## Repository structure

```
.
├── configs/
│   └── preprocessing.yaml       # pose preprocessing parameters (not hardcoded in code)
├── data/
│   ├── retails/                 # RetailS pose dataset (gitignored - see Dataset below)
│   └── external_video/          # UCF-Crime clips + extracted frames (gitignored)
│       ├── README.md            # carried over from the provided package, unmodified
│       ├── videos/{shoplifting,normal}/
│       └── frames/{shoplifting,normal}/
├── scripts/                     # standalone CLI entry points (no notebook required)
│   ├── extract_frames.py        # samples frames from external_video/videos/ -> frames/
│   ├── run_preprocessing.py     # runs the pose preprocessing pipeline end to end
│   └── run_eda.py               # schema check + real dataset statistics + sample plot
├── src/
│   ├── data/                    # retails.py (loader), preprocessing.py (pipeline)
│   ├── motionbert/               # step 3
│   ├── features/                 # step 3/4
│   ├── models/                   # step 4
│   ├── experiments/               # step 4 (configs/experiment_*.yaml will live here)
│   ├── evaluation/                # step 4
│   ├── demo/                      # step 4/5
│   └── utils/
├── notebooks/                   # step 5 builds the Colab notebook here
├── outputs/                     # checkpoints, logs, plots (gitignored)
├── requirements.txt
└── README.md
```

`data/`, `outputs/`, and model checkpoints are gitignored; `.gitkeep`
files preserve the directory structure and `data/external_video/README.md`
is committed as-is.

## Setup

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

pip install -r requirements.txt
```

`requirements.txt` currently covers the preprocessing pipeline: numpy,
pandas, matplotlib, pyyaml, tqdm, torch, opencv-python. MotionBERT's own
dependencies are a TODO in that file - they get pinned in step 3 once
it's clear exactly what the pretrained checkpoint needs.

### Getting the RetailS dataset

RetailS is distributed as a single archive via Google Drive (not
Kaggle/git) by its authors:

1. Download it from the link on the [official repo](https://github.com/TeCSAR-UNCC/RetailS#dataset-download-instructions).
2. Extract it so that `data/retails/RetailS/RetailS_train/`,
   `RetailS_test_realworld/`, and `RetailS_test_staged/` exist (i.e.
   extract the archive directly into `data/retails/`).

`src/data/retails.py` discovers the dataset root at runtime
(`discover_retails_root()`) rather than assuming a fixed path, so minor
differences in how the archive is extracted won't break anything as long
as a `RetailS_train` directory ends up somewhere under `data/retails/`.

### Getting the external validation videos (UCF-Crime)

```bash
python scripts/extract_frames.py
python scripts/extract_frames.py --frames-per-class 150   # override the default of 100
```

Place UCF-Crime shoplifting clips in `data/external_video/videos/shoplifting/`
and normal clips in `data/external_video/videos/normal/` (see
`data/external_video/README.md` for exact source links and the licensing
note - UCF-Crime is a third-party dataset and is not redistributed here).
`extract_frames.py` samples frames approximately uniformly across
whatever videos are present and writes them to `data/external_video/frames/`;
if one class folder is still empty it prints a message and skips that
class instead of failing, since videos get added incrementally.

### Running the pose preprocessing pipeline

```bash
python scripts/run_preprocessing.py
python scripts/run_preprocessing.py --subset test_staged --limit 20   # quick smoke test
```

Runs every person track in the chosen RetailS subset through
`src/data/preprocessing.py`'s full pipeline (sort, confidence-mask,
interpolate short gaps, smooth, center on the pelvis, scale by torso
length, slice into temporal windows) and reports how many tracks/windows
were produced. Parameters come from `configs/preprocessing.yaml`, not
hardcoded values.

### Running the EDA

```bash
python scripts/run_eda.py
```

Prints the real schema found in a sample pose file and label file (and
how it differs from the course spec's summary - see "Dataset" below),
real per-camera track/frame counts for all three RetailS subsets, the
real normal/anomaly label balance for the two labeled test subsets, and
saves a before/after-normalization skeleton plot to
`outputs/plots/skeleton_before_after_normalization.png`.

### Colab setup

Not available yet - step 5 builds `notebooks/` as a Colab notebook that
imports this same `src/` and `scripts/` code (no logic is reimplemented
inline in the notebook). Anything that turns out to only be runnable in
Colab and not locally will be called out explicitly here when that
happens; nothing in steps 1-4 is expected to be Colab-only.

## Dataset

### RetailS (primary)

Source: [TeCSAR-UNCC/RetailS](https://github.com/TeCSAR-UNCC/RetailS) -
*"From Offline to Periodic Adaptation for Pose-Based Shoplifting
Detection in Real-world Retail Security"* (Yao, Rashvand, Pazho, Tabkhi;
IEEE Internet of Things Journal, 2026). Pose sequences (COCO-17
keypoints) extracted via YOLOv8 + ByteTrack + HRNet from 6 indoor cameras
in a real retail store, privacy-preserving (no raw video/pixel data, no
identities). See "Citations" below for the full reference.

**Real schema**, confirmed by actually downloading, extracting, and
loading the archive on 2026-10-15 (not assumed from the spec) - see
`src/data/retails.py`'s module docstring for the complete writeup:

```
<subset>/pose/<train|test>/<camera_id>_<video_id>[_<part>].json
  {
    "<person_id>": {
      "<frame_id>": {
        "keypoints": [x0, y0, c0, ..., x16, y16, c16],  # 51 floats, COCO-17 (x, y, confidence)
        "scores": null
      }, ...
    }, ...
  }

<subset>/gt/test_frame_mask/<camera_id>_<video_id>[_<part>].npy
  uint8 array, one value per video frame (0=normal, 1=anomalous) - only
  present for the two test subsets.
```

**Confirmed differences from the course spec's summary** (fields
expected: `person_id, frame_id, camera_id, bounding_box, COCO-17
keypoints, keypoint confidence, normal/anomaly label`):

- There is **no `bounding_box` field** anywhere in the pose JSON.
- There is **no `camera_id` field inside the JSON** - it's the first
  segment of the filename, not a per-record value.
- The **anomaly label is not an inline field** on any frame/person
  record - it's a separate per-video `.npy` array, indexed by frame, not
  by person.
- **Train and test filenames follow different conventions**: train files
  are `<camera_id>_<video_id>_<part>.json` (long videos split into
  numbered parts); test files (both realworld and staged) are
  `<camera_id>_<video_id>.json`, no part segment. This wasn't in the spec
  summary and was only found by actually running the loader against the
  real files - `parse_filename()` treats `part` as optional.
- `person_id`, `frame_id`, and the 17 COCO keypoints (x, y, confidence)
  all check out as described. The joint *order* within the 17 isn't
  explicitly documented anywhere we found - it was verified statistically
  (see below), not just assumed, since it's load-bearing for the
  pelvis-centering and torso-scaling normalization steps.
- `RetailS_train` ships **no `gt/` directory at all** - the train subset
  has zero anomaly labels by design (it's built for the paper's
  unsupervised/continual-adaptation setup, not a conventional labeled
  training split).

**Joint-order verification**: rather than trust the assumed COCO-17 index
order (0=nose ... 11/12=hips ... 15/16=ankles) on faith, we computed
average pairwise distances between joints across 33,657 high-confidence
frames. Assumed-adjacent joints (e.g. shoulder-elbow, hip-knee) average
56.8px apart vs. 121.7px for all joint pairs, and nose-to-ankle is the
single largest average distance (265-266px) of any pair - exactly what
real human proportions predict under the assumed ordering. This is the
basis for trusting `LEFT_HIP`/`RIGHT_HIP`/`LEFT_SHOULDER`/`RIGHT_SHOULDER`
in `src/data/preprocessing.py`.

**Real dataset statistics** (from `scripts/run_eda.py`, run against the
full downloaded archive - 942 train + 47 realworld-test + 624
staged-test pose files, 2,298 files / 4.2 GB total):

| Subset | Pose files (video/parts) | Person tracks | Pose entries (person-frames) | Cameras |
|---|---|---|---|---|
| train | 942 | 36,555 | 8,137,263 | 6 |
| test_realworld | 47 | 78 | 4,620 | 6 |
| test_staged | 624 | 837 | 46,768 | 6 |

Per-camera breakdown (train): camera 1-6 have 156-159 video/parts each
and between 4,559 and 7,454 person tracks; no single camera dominates the
dataset. Full per-camera numbers print from `scripts/run_eda.py` directly.

Label balance (frame-level, the two labeled test subsets only - train has
none):

| Subset | Normal frames | Anomaly frames | % anomalous |
|---|---|---|---|
| test_realworld | 3,312 | 1,534 | 31.65% |
| test_staged | 20,578 | 20,335 | 49.70% |

**Note on these numbers vs. the official repo's summary table**: the
official README reports slightly different figures (e.g. "2,432 normal /
53 shoplifting frames" for the realworld test set, vs. the 3,312 / 1,534
computed here directly from the downloaded `.npy` files). The staged
subset's normal-frame count matches exactly (20,578); the anomaly-frame
count for staged also matches the official table's "Shoplifting Events"
column exactly (20,335), suggesting that column is actually a frame
count despite its name - but the realworld subset doesn't match as
closely (1,534 vs. 1,933). This project treats the raw `.npy` files in
the archive we actually downloaded as authoritative, since that's the
real data being used, not the summary table.

### External video package (UCF-Crime, secondary/validation)

`data/external_video/` is the provided `Shoplifting_Video_Project_Package`,
merged in as-is (its own `README.md` is preserved unmodified at
`data/external_video/README.md`). It documents:

- Source: [UCF-Crime](https://www.crcv.ucf.edu/research/real-world-anomaly-detection-in-surveillance-videos/)
  real-world anomaly detection dataset (official download, or a
  [Kaggle mirror](https://www.kaggle.com/datasets/minhajuddinmeraj/anomalydetectiondatasetucf)).
- UCF-Crime videos are **not redistributed in this repo** - they're a
  third-party research dataset; download from the official/provider
  source and follow their license terms.
- `scripts/extract_frames.py` (adapted into this project's layout, see
  "Setup" above) extracts a configurable number of frames per class for
  use as external validation input once the step 3/4 pipeline exists.

No UCF-Crime videos have been added yet as of this step - both video
folders are currently empty, which `extract_frames.py` handles by
printing a clear message rather than failing (verified above).

## Features / capabilities (as of step 1)

- **RetailS loader** (`src/data/retails.py`): discovers the dataset root
  at runtime, parses both the train and test filename conventions,
  builds per-person pose tracks, and loads the matching anomaly-label
  array when one exists. Raises rather than silently guessing if a file
  doesn't match either known naming convention.
- **Pose preprocessing pipeline** (`src/data/preprocessing.py`):
  - Per-track sorting by frame_id (tracks are not guaranteed contiguous
    - confirmed real tracks have gaps, e.g. a person leaving and
      re-entering camera view).
  - Per-joint visibility masking by confidence threshold.
  - Short-gap linear interpolation (gaps measured in real frame_id units,
    not array position; interpolates x/y only, never confidence).
  - Optional temporal smoothing (centered moving average).
  - Position normalization: centers every frame on the pelvis (hip
    midpoint).
  - Scale normalization: scales by torso length, falling back to
    shoulder width then hip width if torso length is degenerate.
  - Temporal windowing with configurable `window_length`/`stride`,
    built only over contiguous frame_id runs (never across a real track
    gap).
- **External video frame extraction** (`scripts/extract_frames.py`):
  approximately-uniform frame sampling across all videos in a class,
  configurable target frame count, graceful handling of not-yet-populated
  video folders.
- **EDA / schema verification** (`scripts/run_eda.py`): prints the real
  schema against actual files (not the spec summary), real per-camera
  statistics, real label balance, and a before/after normalization
  skeleton plot - everything in the "Dataset" section above came from
  running this script.

Not yet implemented: MotionBERT integration (2D->3D lifting), the
temporal anomaly detector itself, training/evaluation scripts, the demo
script, and the Colab notebook - these are steps 3-5.

## Results

No model has been trained yet - step 1 is data verification and
preprocessing only. This section gets filled in starting with step 4
(the temporal anomaly detector's training/evaluation results) and
finalized once the external UCF-Crime validation path exists.

## Limitations (known as of step 1)

- RetailS's train subset has zero anomaly labels; whatever
  training/adaptation approach step 4 uses has to account for that (it's
  not a conventional supervised train/test split problem), consistent
  with the source paper's own unsupervised/continual-adaptation framing.
- The realworld test subset is small (47 videos, 4,620 total pose
  entries) relative to the staged test subset - any evaluation split
  between them should account for that size difference.
- COCO-17 joint order was verified statistically, not against an
  explicit schema document from the dataset authors - see "Joint-order
  verification" above for exactly what was checked.
- UCF-Crime external validation hasn't started (no videos added yet);
  until it does, everything here is RetailS-only and hasn't been tested
  against a different camera setup/data source.

## Citations

**RetailS dataset / paper:**

```bibtex
@ARTICLE{11370135,
  author={Yao, Shanle and Rashvand, Narges and Pazho, Armin Danesh and Tabkhi, Hamed},
  journal={IEEE Internet of Things Journal},
  title={From Offline to Periodic Adaptation for Pose-Based Shoplifting Detection in Real-world Retail Security},
  year={2026},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/JIOT.2026.3660205}}
```

Repository: https://github.com/TeCSAR-UNCC/RetailS
Contact (per the dataset repo): nrashvan@charlotte.edu, syao@charlotte.edu

**UCF-Crime dataset:**

Sultani, W., Chen, C., & Shah, M. (2018). *Real-world Anomaly Detection
in Surveillance Videos*. CVPR 2018.
https://www.crcv.ucf.edu/research/real-world-anomaly-detection-in-surveillance-videos/

**MotionBERT**: citation to be added in step 3 once the specific
pretrained checkpoint/repo being used is finalized.
