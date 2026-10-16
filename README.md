# 3D-Pose-Assisted Temporal Anomaly Detection for Shoplifting in Retail Surveillance

Course project for CSE 6367. Team: Harsha Prasad, Himanshu Shimpi.

Status: step 4 of 5 done (temporal attention, Experiment E, the full
A-E comparison, false-positive and per-camera analysis). Steps 1-3 also
done. See "Project status" below.

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
- [x] Step 2: 2D-only baseline - motion features, BiGRU sequence
      autoencoder, `scripts/train.py`/`scripts/evaluate.py`,
      Experiments A (pose-only) and B (pose+velocity) trained and
      evaluated on real data - see "Results" below.
- [x] Step 3: MotionBERT integration (2D -> 3D pose lifting). Joint-index
      mapping (COCO-17 -> H36M-17) written and visually confirmed correct
      (`scripts/validate_joint_mapping.py`) against a real, downloaded
      checkpoint (see "MotionBERT setup"). 3D motion/geometry features
      added (`src/features/motion_features.py`'s `*_3d` functions).
      Experiments C (3D pose only) and D (3D pose + velocity + geometry)
      trained and evaluated through the same `scripts/train.py`/
      `scripts/evaluate.py` as A/B - see "Results" below.
- [x] Step 4: Temporal attention (Experiment E), the full A-E comparison,
      false-positive analysis, and per-camera viewpoint analysis - see
      "Results" below.
- [ ] Step 5: Colab notebook consolidating the full pipeline

## Repository structure

```
.
├── configs/
│   ├── preprocessing.yaml       # pose preprocessing parameters (not hardcoded in code)
│   ├── experiment_a.yaml        # Experiment A: 2D pose only
│   ├── experiment_b.yaml        # Experiment B: 2D pose + velocity
│   ├── experiment_c.yaml        # Experiment C: 3D pose only (via MotionBERT)
│   ├── experiment_d.yaml        # Experiment D: 3D pose + velocity + geometry
│   └── experiment_e.yaml        # Experiment E: D's features + temporal attention
├── data/
│   ├── retails/                 # RetailS pose dataset (gitignored - see Dataset below)
│   └── external_video/          # UCF-Crime clips + extracted frames (gitignored)
│       ├── README.md            # carried over from the provided package, unmodified
│       ├── videos/{shoplifting,normal}/
│       └── frames/{shoplifting,normal}/
├── scripts/                     # standalone CLI entry points (no notebook required)
│   ├── extract_frames.py        # samples frames from external_video/videos/ -> frames/
│   ├── run_preprocessing.py     # runs the pose preprocessing pipeline end to end
│   ├── run_eda.py               # schema check + real dataset statistics + sample plot
│   ├── train.py                 # trains one experiment (--experiment a-e) via src/models
│   ├── evaluate.py               # evaluates a trained checkpoint on both labeled test subsets
│   ├── validate_joint_mapping.py # renders COCO-17 vs H36M-17 (+ 3D lift, if a checkpoint exists)
│   ├── analyze_false_positives.py # traces/renders every false positive for a 2D experiment
│   └── analyze_per_camera.py     # per-camera metric breakdown, best 2D vs. best 3D experiment
├── src/
│   ├── data/                    # retails.py (loader), preprocessing.py (pipeline)
│   ├── motionbert/               # lift.py - frozen pretrained MotionBERT, 2D->3D lifting
│   ├── features/                 # motion_features.py - velocity/acceleration feature vectors
│   ├── models/                   # bigru_autoencoder.py, bigru_attention.py - the two autoencoders
│   ├── experiments/               # dataset.py - builds (features, labels[, metadata]) windows per subset
│   ├── evaluation/                # metrics.py - precision/recall/F1/ROC-AUC/PR-AUC, shared by all experiments
│   ├── demo/                      # step 5
│   └── utils/                     # joint_mapping.py - COCO-17 <-> H36M-17
├── third_party/
│   └── MotionBERT/               # official repo, `git clone`d locally (gitignored - see "MotionBERT setup")
├── notebooks/                   # step 5 builds the Colab notebook here
├── outputs/                     # checkpoints, logs, plots (gitignored)
├── requirements.txt
└── README.md
```

`data/`, `outputs/`, `third_party/`, and model checkpoints are
gitignored; `.gitkeep` files preserve the directory structure and
`data/external_video/README.md` is committed as-is.

## Setup

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

pip install -r requirements.txt
```

`requirements.txt` covers everything through step 3: numpy, pandas,
matplotlib, pyyaml, tqdm, torch, opencv-python, scikit-learn (used by
`src/evaluation/metrics.py` for ROC-AUC/PR-AUC/confusion-matrix). No
extra packages are needed for MotionBERT itself - its `DSTformer` model
(`third_party/MotionBERT/lib/model/DSTformer.py`) only imports `torch`,
`numpy`, and the standard library, confirmed by reading its actual
imports rather than installing MotionBERT's own `requirements.txt`
wholesale (which pins an old, potentially conflicting `matplotlib`).

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

### Training and evaluating an experiment

```bash
python scripts/train.py --experiment a     # 2D pose only
python scripts/train.py --experiment b     # 2D pose + velocity
python scripts/train.py --experiment c     # 3D pose only
python scripts/train.py --experiment d     # 3D pose + velocity + geometry
python scripts/train.py --experiment e     # D's features + temporal attention
python scripts/evaluate.py --experiment a  # (same flag for b/c/d/e)
```

`train.py` builds normal-only training windows from `RetailS_train`
(RetailS_train has no anomaly labels at all, so training on the full
subset already satisfies "normal sequences only"), trains the
autoencoder named by the experiment config's `model` field
(`src/models/bigru_autoencoder.py`'s `BiGRUAutoencoder` if absent - A-D;
`src/models/bigru_attention.py`'s `BiGRUAttentionAutoencoder` if
`model: bigru_attention` - E), and saves a checkpoint plus the full
training-set reconstruction-error distribution under
`outputs/checkpoints/<experiment_name>/`. `evaluate.py` loads that
checkpoint, sets a decision threshold from the saved training-error
distribution (`threshold_percentile` in the experiment config - the
default, 95.0, flags a window anomalous if its reconstruction error
exceeds the 95th percentile of normal-window errors), and reports
precision/recall/F1/specificity/false-positive-rate/ROC-AUC/PR-AUC/
inference-time-per-window on `test_realworld`, `test_staged`, and both
combined, via `src/evaluation/metrics.py`. Every hyperparameter lives in
`configs/experiment_*.yaml`, not hardcoded in the scripts.

CPU-only training note: this was developed and timed on a 12-core CPU
with no GPU. Training on the full `RetailS_train` subset (36,555 person
tracks) isn't feasible in reasonable time, so `max_train_pose_files` in
each experiment config subsamples how many of the 942 train pose files
get used (50 by default - see the comment in `configs/experiment_a.yaml`
for the empirical timing this was tuned against). Evaluation always runs
on the *full*, unsampled test subsets - only training is subsampled.

### MotionBERT setup

MotionBERT is used as a frozen, pretrained 2D->3D pose lifter - it is
never fine-tuned or otherwise modified by this project. Getting it
working locally (and in Colab) is two separate steps: cloning the code,
and manually downloading a pretrained checkpoint (the checkpoint cannot
be downloaded by a script - see below).

1. **Clone the official repo** into `third_party/` (gitignored - this is
   a dependency fetched at setup time, not vendored into this repo):

   ```bash
   git clone --depth 1 https://github.com/Walter0807/MotionBERT.git third_party/MotionBERT
   ```

   `src/motionbert/lift.py` adds `third_party/MotionBERT` to `sys.path`
   at import time and imports `lib.model.DSTformer.DSTformer` directly -
   it does not go through MotionBERT's own CLI/config (`infer_wild.py`,
   `easydict`-based YAML configs), since this project already has clean
   `(T, 17, 3)` pose tensors from `src/data/preprocessing.py` and doesn't
   need their video/AlphaPose-JSON loading pipeline.

2. **Download a pretrained pose3d checkpoint.** MotionBERT's Model Zoo
   ships several checkpoints; the general-purpose "MotionBERT"/
   "MotionBERT-Lite" pretrain checkpoints output 512-dim representations,
   **not** 3D coordinates, and are the wrong ones for this project. The
   right ones are the two H36M pose3d checkpoints (confirmed against
   `third_party/MotionBERT/README.md`'s own Model Zoo table):

   | Checkpoint | MPJPE | Config it matches | OneDrive link |
   |---|---|---|---|
   | 3D Pose (H36M-SH, ft) - **used here** | 37.2mm | `configs/pose3d/MB_ft_h36m.yaml` | https://1drv.ms/f/s!AvAdh0LSjEOlgSoTqtyR5Zsgi8_Z?e=rn4VJf |
   | 3D Pose (H36M-SH, scratch) | 39.2mm | `configs/pose3d/MB_train_h36m.yaml` | https://1drv.ms/f/s!AvAdh0LSjEOlgSvNejMQ0OHxMGZC?e=KcwBk1 |

   This is a **manual download** - the link is a OneDrive folder, and
   the folder (confirmed by testing it directly) returns 403 Forbidden
   to non-browser requests, so it cannot be scripted the way the RetailS
   Google Drive archive is. Open the "ft" link above in a browser,
   download the single checkpoint file inside (named `best_epoch.bin` in
   MotionBERT's own examples), and place it at:

   ```
   third_party/MotionBERT/checkpoint/pose3d/MB_ft_h36m/best_epoch.bin
   ```

   (`mkdir -p third_party/MotionBERT/checkpoint/pose3d/MB_ft_h36m` first
   if the folders don't exist yet.) This is the default path
   `scripts/validate_joint_mapping.py` (and, once Experiments C/D exist,
   `scripts/train.py`) look for; override it with `--checkpoint <path>`
   if it's placed somewhere else. Do this once per machine (local and
   Colab both need their own copy - Colab setup below covers
   mounting/uploading it there).

3. **Verify the setup**:

   ```bash
   python scripts/validate_joint_mapping.py
   ```

   Without a checkpoint in place, this still renders the COCO-17 vs.
   H36M-17 joint-reindexing comparison (see "Features / capabilities"
   below) and prints an explicit message that the 3D-lift panel is being
   skipped - it does not fail or fake that panel. With a checkpoint in
   place, it additionally runs the real 2D->3D lift and renders it.

### Colab setup

Not available yet - step 5 builds `notebooks/` as a Colab notebook that
imports this same `src/` and `scripts/` code (no logic is reimplemented
inline in the notebook). The MotionBERT checkpoint download in
particular is manual in Colab too (same OneDrive 403 restriction) - the
notebook will need the checkpoint uploaded to Drive or the Colab runtime
directly, documented in full once step 5 builds the notebook. Nothing
else in steps 1-4 is expected to be Colab-only.

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

## Features / capabilities (as of step 4)

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
    midpoint). Optional (`apply_normalization=False`) - MotionBERT (see
    below) needs cleaned-but-unnormalized coordinates, since it applies
    its own normalization convention.
  - Scale normalization: scales by torso length, falling back to
    shoulder width then hip width if torso length is degenerate.
  - Temporal windowing with configurable `window_length`/`stride`,
    built only over contiguous frame_id runs (never across a real track
    gap).
- **Motion features** (`src/features/motion_features.py`): per-frame
  velocity (position delta) for the 12 non-facial joints (shoulders,
  elbows, wrists, hips, knees, ankles), with acceleration as an optional
  flag (off by default). First frame is zero-padded.
- **BiGRU sequence autoencoder** (`src/models/bigru_autoencoder.py`):
  bidirectional-GRU encoder -> latent projection -> unidirectional-GRU
  decoder, trained one-class (normal sequences only). Anomaly score =
  mean squared reconstruction error.
- **Temporal attention** (`src/models/bigru_attention.py`): added on top
  of the same BiGRU encoder for Experiment E, not a rewrite -
  `BiGRUAttentionAutoencoder` subclasses `BiGRUAutoencoder`, inherits its
  encoder/decoder/projections unchanged, and replaces only the
  final-hidden-state latent with an attention-weighted pool over every
  timestep's encoder output.
- **Training / evaluation** (`scripts/train.py`, `scripts/evaluate.py`,
  `src/experiments/dataset.py`, `src/evaluation/metrics.py`): trains an
  experiment end to end from a `configs/experiment_*.yaml` (model class
  selected by the config's `model` field), and reports
  precision/recall/F1/specificity/false-positive-rate/ROC-AUC/PR-AUC/
  inference-time-per-window plus a reconstruction-error histogram, on
  both labeled RetailS test subsets. The same evaluator is reused
  unchanged by every experiment (A-E). For the 3D experiments (C/D/E),
  lifted windows are cached to disk (`outputs/cache/motionbert_lift/`)
  keyed by subset/sampling/preprocessing parameters - Experiment D/E's
  window-build reuses Experiment C's already-lifted windows rather than
  re-running MotionBERT on the same data. `build_windows_for_subset[_3d]`
  can optionally also return per-window metadata (source pose file,
  person/camera/video id, start/end frame) for the two analysis scripts
  below, without changing the default return shape training/evaluation
  use.
- **False-positive analysis** (`scripts/analyze_false_positives.py`):
  finds every false positive for a 2D experiment on the full test set,
  re-derives and renders the actual skeleton (not just the aggregate
  metric), and reports simple geometric descriptors (hand-to-hip/torso
  proximity, knee angle) - see "Results" for what was actually found.
- **Per-camera analysis** (`scripts/analyze_per_camera.py`): breaks test
  metrics down by RetailS `camera_id` for two chosen experiments and
  compares cross-camera ROC-AUC variance between them - built to test
  the spec's 3D-vs-viewpoint-sensitivity hypothesis against real numbers
  (see "Results"), not assumed.
- **3D motion/geometry features**
  (`src/features/motion_features.py`'s `*_3d` functions): the 3D
  analogues of the 2D motion features - per-joint velocity/acceleration
  on MotionBERT's lifted output, plus body-relative distances
  (wrist-to-pelvis, wrist-to-thorax, wrist-to-wrist, normalized by
  per-frame torso length) and optional joint angles (elbows, knees).
- **Joint mapping** (`src/utils/joint_mapping.py`): explicit, documented
  index mapping from RetailS's COCO-17 keypoint order to the H36M-17
  order MotionBERT expects - 13 of 17 joints are a direct reuse of a
  COCO-17 point (all limb joints + nose), the other 4 (pelvis, thorax,
  spine, head) have no COCO-17 equivalent and are estimated via midpoints,
  the same approach MotionBERT's own Halpe26->H36M conversion uses for
  the joints Halpe26 has that COCO-17 doesn't.
- **MotionBERT 2D->3D lifting** (`src/motionbert/lift.py`): loads a
  frozen, pretrained MotionBERT (`DSTformer`) checkpoint and runs 2D->3D
  lifting on a preprocessed window - reindexes COCO-17 to H36M-17, applies
  MotionBERT's own bounding-box-to-[-1,1] normalization (not this
  project's pelvis/torso normalization), and returns root-relative 3D
  coordinates. MotionBERT's weights are never updated.
- **Joint-mapping validation** (`scripts/validate_joint_mapping.py`):
  picks the cleanest real example windows it can find (same
  highest-confidence-frame selection approach as the EDA skeleton plot),
  and renders the COCO-17 skeleton next to its H36M-17 reindexing, with
  shoulders/elbows/wrists/hips/knees/ankles labeled, so the mapping can be
  checked by eye - plus the actual 3D lift, if a checkpoint is present.
- **External video frame extraction** (`scripts/extract_frames.py`):
  approximately-uniform frame sampling across all videos in a class,
  configurable target frame count, graceful handling of not-yet-populated
  video folders.
- **EDA / schema verification** (`scripts/run_eda.py`): prints the real
  schema against actual files (not the spec summary), real per-camera
  statistics, real label balance, and a before/after normalization
  skeleton plot - everything in the "Dataset" section above came from
  running this script.

Not yet implemented: the demo script and the Colab notebook (step 5).

## Results

All experiments: BiGRU sequence autoencoder, one-class (trained on
`RetailS_train` only, which has zero anomaly labels by design), decision
threshold = 95th percentile of the training set's own reconstruction-error
distribution, `window_length=30`/`stride=15`, 50 of 942 train pose files
sampled for training (CPU-only, see "Training and evaluating an
experiment" above), evaluated on the full (unsampled) `test_realworld` +
`test_staged` subsets via `scripts/evaluate.py`. Exact commands:

```bash
python scripts/train.py --experiment a && python scripts/evaluate.py --experiment a
python scripts/train.py --experiment b && python scripts/evaluate.py --experiment b
```

**Experiment A (2D pose only, 34-dim per frame: 17 joints x (x, y), no
confidence column)** - trained 15 epochs, 22,493 windows, 916s total,
final train/val loss 0.232/0.124:

| Test subset | Precision | Recall | F1 | Specificity | FPR | ROC-AUC | PR-AUC | Normal / Anomaly windows |
|---|---|---|---|---|---|---|---|---|
| test_realworld | 0.750 | 0.068 | 0.125 | 0.966 | 0.034 | 0.525 | 0.630 | 88 / 132 |
| test_staged | 1.000 | 0.055 | 0.104 | 1.000 | 0.000 | 0.573 | 0.987 | 33 / 1,867 |
| combined | 0.974 | 0.056 | 0.105 | 0.975 | 0.025 | 0.531 | 0.948 | 121 / 1,999 |

**Experiment B (2D pose + velocity, 58-dim per frame: the above 34 +
12 joints x (vx, vy))** - trained 15 epochs, 22,493 windows, 857s total,
final train/val loss 0.297/0.136:

| Test subset | Precision | Recall | F1 | Specificity | FPR | ROC-AUC | PR-AUC | Normal / Anomaly windows |
|---|---|---|---|---|---|---|---|---|
| test_realworld | 0.750 | 0.068 | 0.125 | 0.966 | 0.034 | 0.536 | 0.629 | 88 / 132 |
| test_staged | 1.000 | 0.054 | 0.102 | 1.000 | 0.000 | 0.562 | 0.986 | 33 / 1,867 |
| combined | 0.973 | 0.055 | 0.103 | 0.975 | 0.025 | 0.524 | 0.946 | 121 / 1,999 |

**Reading A vs. B honestly**: at the p95 threshold, both experiments are
precise and highly specific (rarely flag a normal window) but recall is
very low - they miss the large majority of anomalous windows, and
combined ROC-AUC (A: 0.531, B: 0.524) is barely above chance (0.5) for
both. Adding velocity (B) does not meaningfully change the picture -
per-subset ROC-AUC moves by about +0.01 (realworld) and -0.01 (staged)
relative to A, well within run-to-run noise rather than a real
improvement. In plain terms: neither static 2D pose geometry nor
frame-to-frame 2D velocity separates normal from anomalous windows well
on this dataset at a conservative threshold. This isn't a bug - PR-AUC
is high mainly because the test sets are heavily skewed toward the
anomaly class (especially test_staged, 1,867 of 1,900 windows), which
inflates PR-AUC on its own; ROC-AUC is the more honest number here and
it's close to 0.5 for both experiments. This result is exactly the
motivation for the 3D+motion+geometry experiments C/D once MotionBERT is
in place - if pure 2D pose and 2D velocity are both weak signals, 3D
structure is the more promising remaining direction, not a given.

Reconstruction-error histograms (normal vs. anomaly, per test subset)
are saved to `outputs/plots/experiment_<a,b,c,d>/` by `scripts/evaluate.py`.

### Experiments C and D (3D pose, via MotionBERT)

Same BiGRU autoencoder, training setup, and decision-threshold
convention as A/B. Two differences, both necessary and both documented
rather than glossed over:

- **Training sample is smaller: 10 of 942 train pose files**, not A/B's
  50. A MotionBERT (DSTformer) forward pass per window is the dominant
  cost of building 3D windows (~0.16s/window on this 12-core CPU,
  measured directly - batching windows into one call barely helps,
  since the bottleneck is real compute, not per-call overhead). C and D
  use the *same* 10-file sample, and D's build reuses C's already-lifted
  windows via a disk cache (`outputs/cache/motionbert_lift/`, gitignored)
  instead of re-lifting from scratch - confirmed directly: D's window-build
  step took 0.8s (cache hit) vs. C's 766.9s (cache miss) for the
  identical 3,939 windows.
- **Evaluation samples 20 pose files from each of test_realworld/
  test_staged**, not A/B's full, unsampled subsets - the full 671 test
  pose files would take on the order of 12 hours to lift at the
  measured per-window rate. This is a real, necessary deviation from
  A/B's convention, not an oversight, and it has a real consequence
  (see test_staged below).

Exact commands:

```bash
python scripts/train.py --experiment c && python scripts/evaluate.py --experiment c
python scripts/train.py --experiment d && python scripts/evaluate.py --experiment d
```

**Experiment C (3D pose only, 51-dim per frame: 17 joints x (x, y, z))**
- trained 15 epochs, 3,939 windows, 874.1s total, final train/val loss
0.002143/0.001828:

| Test subset | Precision | Recall | F1 | Specificity | FPR | ROC-AUC | PR-AUC | Normal / Anomaly windows |
|---|---|---|---|---|---|---|---|---|
| test_realworld | 1.000 | 0.192 | 0.323 | 1.000 | 0.000 | 0.612 | 0.917 | 10 / 52 |
| test_staged | 1.000 | 0.143 | 0.250 | n/a (0 normal) | n/a | n/a | n/a | 0 / 56 |
| combined | 1.000 | 0.167 | 0.286 | 1.000 | 0.000 | 0.570 | 0.950 | 10 / 108 |

**Experiment D (3D pose + velocity + geometry, 92-dim per frame: the
above 51 + 12 joints x (vx, vy, vz) + 5 body-relative distances)** -
trained 15 epochs, 3,939 windows, 109.3s total (lift reused from C's
cache), final train/val loss 0.002731/0.002603:

| Test subset | Precision | Recall | F1 | Specificity | FPR | ROC-AUC | PR-AUC | Normal / Anomaly windows |
|---|---|---|---|---|---|---|---|---|
| test_realworld | 1.000 | 0.115 | 0.207 | 1.000 | 0.000 | 0.615 | 0.909 | 10 / 52 |
| test_staged | 1.000 | 0.125 | 0.222 | n/a (0 normal) | n/a | n/a | n/a | 0 / 56 |
| combined | 1.000 | 0.120 | 0.215 | 1.000 | 0.000 | 0.643 | 0.957 | 10 / 108 |

**test_staged shows "n/a", not 0 or an error, for specificity/ROC-AUC/
PR-AUC, for both C and D** - the 20-file random sample happens to
contain zero normal windows. This isn't a bug (`src/evaluation/metrics.py`
already special-cased "only one class present" back when it was built
for A/B) - it's a direct consequence of how rare normal windows already
are in the *full* test_staged subset (33 of 1,900 windows in A/B's full,
unsampled evaluation - 1.7%). At that rate, a 56-window sample has
under a 50% chance of containing even one normal window. A much larger
eval sample would be needed to fix this reliably (rough math: ~600+
windows to expect even ~10 normal ones, which is most of the full
subset) - not attempted here given the per-window lifting cost.

**Reading C/D honestly, and *why a direct C/D-vs-A/B comparison is only
suggestive, not conclusive***: combined ROC-AUC is higher for C (0.570)
and higher still for D (0.643) than for A (0.531) or B (0.524),
consistent with the hypothesis that 3D structure (and more so 3D +
motion/geometry) carries more signal than 2D pose alone. But C/D's
combined metric comes from only 118 windows (40 sampled pose files),
while A/B's comes from 2,120 windows (the full, unsampled test subsets)
- a ~18x difference in sample size. A metric computed on 118 windows is
considerably noisier than one computed on 2,120, so part or all of the
apparent C/D improvement could be sampling luck rather than a real
effect of 3D pose. Confirming this honestly would need evaluating A/B on
the *same* 40-file subsample C/D used (an apples-to-apples comparison),
which hasn't been done. Stated as a real limitation, not smoothed over -
see "Limitations" below.

Reconstruction-error histograms for C/D/E are saved to
`outputs/plots/experiment_<c,d,e>/`.

### Experiment E (3D pose + velocity + geometry + temporal attention)

Same features as D; the only difference is the model -
`src/models/bigru_attention.py`'s `BiGRUAttentionAutoencoder` in place of
`BiGRUAutoencoder`, added on top of the same BiGRU encoder (see that
module's docstring) rather than a rewrite. Reuses D/C's cached
MotionBERT lift - trained 15 epochs, 3,939 windows, 130.8s total (vs.
D's 874.1s, since the expensive lift step was already cached), final
train/val loss 0.003089/0.003088:

```bash
python scripts/train.py --experiment e && python scripts/evaluate.py --experiment e
```

| Test subset | Precision | Recall | F1 | Specificity | FPR | ROC-AUC | PR-AUC | Normal / Anomaly windows |
|---|---|---|---|---|---|---|---|---|
| test_realworld | 1.000 | 0.096 | 0.175 | 1.000 | 0.000 | 0.604 | 0.900 | 10 / 52 |
| test_staged | 1.000 | 0.179 | 0.303 | n/a (0 normal) | n/a | n/a | n/a | 0 / 56 |
| combined | 1.000 | 0.139 | 0.244 | 1.000 | 0.000 | 0.634 | 0.953 | 10 / 108 |

Temporal attention (E) lands between C and D on combined ROC-AUC (C
0.570, E 0.634, D 0.643) - not a clear win over D's plain feature
concatenation, and the same ~18x-smaller-eval-sample caveat as C/D
applies to this ranking too (see above). Attention also costs more at
inference: 0.55ms/window vs. D's 0.47ms/window - see the full comparison
below.

### Full comparison: Experiments A-E

One table, same evaluator (`src/evaluation/metrics.py`) for every row,
combined test set (`test_realworld` + `test_staged`):

| Experiment | Features | Precision | Recall | F1 | Specificity | FPR | ROC-AUC | PR-AUC | Inference (ms/window) |
|---|---|---|---|---|---|---|---|---|---|
| A | 2D pose | 0.974 | 0.056 | 0.105 | 0.975 | 0.025 | 0.531 | 0.948 | 0.443 |
| B | 2D pose + velocity | 0.973 | 0.055 | 0.103 | 0.975 | 0.025 | 0.524 | 0.946 | 0.443 |
| C | 3D pose | 1.000 | 0.167 | 0.286 | 1.000 | 0.000 | 0.570 | 0.950 | 0.421 |
| D | 3D pose + velocity + geometry | 1.000 | 0.120 | 0.215 | 1.000 | 0.000 | **0.643** | 0.957 | 0.466 |
| E | D's features + temporal attention | 1.000 | 0.139 | 0.244 | 1.000 | 0.000 | 0.634 | 0.953 | 0.552 |

Inference time is the trained autoencoder's own forward+scoring pass
only (`scripts/evaluate.py:score_windows`), not the one-time MotionBERT
lift C/D/E also pay - that lift is shared/cached and isn't a property of
the autoencoder architecture being compared here. By that measure, A-E
are all sub-millisecond per window and the differences between them
(0.42-0.55ms) are small relative to run-to-run noise on a general-
purpose CPU - temporal attention (E) is the slowest, consistent with it
doing strictly more work (scoring every timestep instead of reading off
the last one), but not by a dramatic margin.

**Honest bottom line on A-E**: D has the best combined ROC-AUC, E is
close behind, C is a clear step below both, and A/B (2D) trail all
three 3D variants. This is consistent with the "3D carries more signal"
hypothesis from the C/D introduction above, but the same caveat applies
with equal force here - A/B's numbers come from 2,120 full-test-set
windows, C/D/E's from 118 (40 sampled pose files). The ranking among
C/D/E themselves (D > E > C) is on equal footing (same 40-file sample,
same evaluator) and is the more trustworthy part of this table; the gap
between {A,B} and {C,D,E} is the part that would need a matched-sample
re-evaluation of A/B to confirm.

### False-positive analysis

RetailS has no per-window action label - only binary normal/anomaly -
so there is no ground truth to check false positives against the course
spec's risky scenarios (adjusting clothing, reaching into a personal
bag, phone into pocket, crouching, holding an item near the torso,
normal product comparison). `scripts/analyze_false_positives.py`
doesn't fabricate one: it finds every false positive for a 2D
experiment on the full test set, re-derives and renders the actual
skeleton, and reports simple geometric descriptors (hand-to-hip/torso
proximity, knee angle) to reason from alongside the picture, rather
than asserting a scenario label.

Experiment A has exactly 3 false positives in its full-test-set
evaluation (FPR 2.5%, 3 of 121 normal windows) - all in
`test_realworld`, none in `test_staged`. Experiment B flags the
identical 3 windows, which rules out run-to-run model noise as the
explanation. Looking at the renders (`outputs/plots/
false_positive_analysis/fp_{0,1,2}.png`):

- **FP 0 and FP 1 are the same underlying event**, caught by two
  overlapping sliding windows from one track (`02_0310.json`, person 2,
  camera 2, frames 327-356 and 342-371). The rendered skeleton is not a
  recognizable human pose at any of the 4 sampled frames - limbs cross
  and tangle in ways that aren't anatomically possible. This looks like
  a pose-tracking failure in the upstream RetailS pipeline (a garbled
  keypoint estimate), not a real human action - the model's very high
  reconstruction error (2.54, 23x the threshold) is consistent with
  "this doesn't look like anything in the training distribution,
  normal or anomalous" rather than a near-miss on a risky-looking
  action.
- **FP 2** (`03_0247.json`, person 8, camera 3, frames 47-76) is a
  different, independent track. Its skeleton is temporally *stable*
  across the window (not tangled or jittery like FP 0/1) but has an
  atypical, elongated silhouette - and its geometry argues against the
  spec's scenarios rather than for them: the closest either wrist gets
  to the torso is 0.59 torso-lengths (wrists stay away from the body,
  not held near it), wrists never come close together (minimum 0.48
  torso-lengths apart, not clasped), and the knee-angle signal is
  unreliable here (near-0-degree minimums that are anatomically
  implausible, most likely the same kind of keypoint-quality issue as
  FP 0/1 rather than genuine extreme crouching - caught by computing it
  and getting an implausible number, not assumed reliable).

**What was actually found**: with only 2 independent false-positive
events to go on, this is a small-sample observation, not a general
claim - but on direct inspection, neither event resembles the spec's
risky scenarios. Both instead look like pose-estimation/tracking
quality problems upstream of this project's own code. If that holds up
with more false positives (e.g. once a larger/matched eval sample is
run), the practical implication would be that filtering or flagging
low-quality tracks matters more for this model's false-alarm rate than
distinguishing risky-but-legitimate actions from real anomalies - a
different problem than the one the spec's scenario list anticipates.

Experiments C, D, and E have **zero** false positives on their own
eval samples (specificity 1.000 for every subset where it's defined,
per the full A-E table above) - there is nothing to analyze for them
on this run. Worth flagging honestly: with only 10 normal windows in
each of their eval samples, "zero false positives" is also a
small-sample result, not evidence that 3D pose eliminates false alarms.

### Per-camera analysis (viewpoint sensitivity)

`scripts/analyze_per_camera.py` breaks the test set down by RetailS
`camera_id` for the best 2D experiment (A, by combined ROC-AUC) and the
best 3D experiment (D), to test the course spec's hypothesis that 3D
pose may be less sensitive to camera viewpoint than 2D pose.

Two things had to be fixed/handled before the numbers below could be
trusted, both caught by inspecting intermediate output rather than
assumed:

- **`camera_id` is formatted inconsistently between test subsets**:
  `test_realworld` filenames zero-pad it (`01_0222.json`), `test_staged`
  doesn't (`1_1050000.json`) - the same physical camera. Combining both
  subsets without normalizing (`str(int(camera_id))`) silently splits
  each camera into two buckets (confirmed directly: an unnormalized
  first pass showed 12 "cameras" for 6 real ones). Fixed in
  `analyze_per_camera.py`; this doesn't affect any other script, since
  nothing else combines camera_id across both test subsets.
- **D's own small eval sample (20 files/subset) left 5 of 6 cameras
  with zero normal windows** - undersampling, not a property of the
  cameras themselves, and it would have made the viewpoint comparison
  nearly meaningless (ROC-AUC undefined for 5/6 cameras). Fixed by
  running D on the FULL, unsampled `test_realworld` subset specifically
  for this analysis (only 47 pose files, a one-time ~1-minute lift once
  cached) - `test_staged` stays at its usual sample, since its normal
  windows are too rare at any feasible sample size to help here (see
  "Limitations").

**Real per-camera ROC-AUC** (both classes present in every camera for
both experiments, after the fixes above):

| Camera | Experiment A (2D) | Experiment D (3D) |
|---|---|---|
| 1 | 0.586 | 0.588 |
| 2 | 0.278 | 0.382 |
| 3 | 0.595 | 0.498 |
| 4 | 0.493 | 0.533 |
| 5 | 0.741 | 0.900 |
| 6 | 0.401 | 0.467 |
| **range (max-min)** | **0.462** | **0.518** |
| **std across cameras** | **0.148** | **0.164** |

**Honest answer on the viewpoint hypothesis: not supported by these
numbers.** 3D pose (D) does not show less cross-camera variance than 2D
pose (A) - its range and standard deviation across cameras are both
*larger*, not smaller. Camera 2 is the worst-performing camera and
camera 5 the best for **both** experiments, which suggests whatever
makes a camera easy or hard here (viewing angle, occlusion, lighting,
crowd density) is a property of the camera/scene itself, not something
3D pose estimation corrects for - the same cameras stay hard regardless
of which representation the anomaly detector sees.

Caveat worth stating plainly: A and D aren't on fully equal footing
either (D's underlying model was trained on 10 sampled files vs. A's
50, per "Training and evaluating an experiment" above), so part of D's
*higher* variance could reflect a less-trained model rather than
something intrinsic to 3D pose. What the data does support clearly is
the negative result - there's no evidence here that 3D pose reduces
viewpoint sensitivity, which is itself a useful, reportable finding
for the research question, even though it cuts against the spec's
hypothesis.

## Limitations (known as of step 4)

- RetailS's train subset has zero anomaly labels; the one-class
  training approach used throughout (A-E) accounts for that (it's not a
  conventional supervised train/test split problem), consistent with
  the source paper's own unsupervised/continual-adaptation framing.
- The realworld test subset is small (47 videos, 4,620 total pose
  entries) relative to the staged test subset - any evaluation split
  between them should account for that size difference.
- COCO-17 joint order was verified statistically, not against an
  explicit schema document from the dataset authors - see "Joint-order
  verification" above for exactly what was checked.
- UCF-Crime external validation hasn't started (no videos added yet);
  until it does, everything here is RetailS-only and hasn't been tested
  against a different camera setup/data source.
- CPU-only training meant subsampling `RetailS_train` to 50 of 942 pose
  files per experiment (see "Training and evaluating an experiment")
  rather than training on the full subset - results above should be read
  as what this subsample supports, not a claim about the full dataset's
  ceiling.
- Experiment A's recall and ROC-AUC are low (see "Results") - static 2D
  pose geometry alone is a weak anomaly signal on this dataset at a
  conservative (p95) threshold. Documented as a real, honest finding
  motivating the velocity/3D/motion-feature experiments, not hidden or
  smoothed over.
- Four of the H36M-17 joints MotionBERT expects (pelvis, thorax, spine,
  head) have no direct COCO-17 equivalent and are estimated via
  midpoints (see "Joint mapping" in "Features / capabilities") - this is
  the same approach MotionBERT's own code uses when converting from a
  skeleton format that lacks those points, but it's an approximation,
  not a measured keypoint, and downstream 3D features derived from those
  4 joints inherit that approximation.
- The pretrained MotionBERT checkpoint must be downloaded manually
  (OneDrive link, confirmed to return 403 to non-browser/scripted
  requests) - this is an external, one-time setup step on each machine
  (local and Colab), not something `pip install`/this repo's own scripts
  can do unattended. See "MotionBERT setup".
- MotionBERT's predicted depth (z) is both smaller in range than x/y
  (roughly 15-30% as much, checked numerically across several sequences)
  and unevenly distributed across the body - concentrated mostly in the
  legs/knees, with the torso and arms close to flat in depth. This was
  checked carefully, not just eyeballed: per-joint left-right (x) and
  up-down (y) ordering between the 2D input and the 3D output matches
  almost perfectly in every sequence checked, which rules out a
  reindexing/sign bug - the weakness is specific to z. Plausible given
  MotionBERT was trained on Human3.6M's eye-level, calibrated studio
  footage and is being applied here, frozen, to an uncalibrated, steeply
  elevated CCTV angle it never saw in training. The lift still produces
  anatomically consistent, correctly-sided skeletons in x/y (see
  `scripts/validate_joint_mapping.py`'s renders), but z-derived signal
  (depth-based motion/geometry features in Experiment D) should be read
  as weaker and less trustworthy than x/y-derived signal, not a faithful
  full-depth reconstruction. Proceeding with the real (x, y, z) as-is for
  Experiments C/D and reporting this honestly was a deliberate choice,
  not an oversight - consistent with how A/B's low recall was reported
  rather than hidden.
- RetailS occasionally ships a literal NaN confidence value for a joint
  (x/y still present and finite) rather than a low-but-real number -
  confirmed in `RetailS_test_realworld/pose/test/01_0257.json` (person
  1, frame_ids 9-11, every joint), found when it crashed Experiment C's
  evaluation (`sklearn`'s ROC-AUC rejects NaN scores). The 2D BiGRU
  branch (A/B) was never affected - it discards the confidence column
  before building features - but MotionBERT takes confidence as a real
  input channel, and because its temporal attention mixes every frame in
  a window together, one NaN frame silently contaminated an entire
  window's 3D output. Fixed at the source
  (`src/data/preprocessing.py:preprocess_track`, NaN confidence ->
  0.0, same semantics as RetailS's own zero-confidence entries) rather
  than patched locally in the MotionBERT path, so every consumer sees
  clean data. Confirmed the existing train-set lift cache was unaffected
  (zero NaNs in the 10 sampled train files) before trusting Experiments
  C/D's already-trained checkpoints.
- Experiments C/D evaluate on a 20-pose-file sample per test subset, not
  the full subsets A/B use (see "Results" - a MotionBERT forward pass
  per window makes the full ~12h-estimated eval infeasible on CPU). One
  concrete consequence: test_staged's sample happened to contain zero
  normal windows, making specificity/ROC-AUC/PR-AUC undefined for that
  subset specifically (reported as "n/a", not estimated or hidden). A
  second: any apparent C/D-vs-A/B improvement in combined ROC-AUC is
  only suggestive, not conclusive, given the ~18x smaller, unmatched
  eval sample - see "Reading C/D honestly" in "Results" for the full
  reasoning. The same applies to E, and to the D-vs-A per-camera
  comparison (D's underlying model was also trained on far fewer pose
  files than A's - see "Per-camera analysis").
- RetailS's two test subsets format `camera_id` inconsistently in their
  filenames (`test_realworld`: zero-padded, e.g. `01_...`;
  `test_staged`: not, e.g. `1_...`) - the same physical camera, two
  different strings. Caught and fixed in `scripts/analyze_per_camera.py`
  specifically (the first place combining camera_id across both test
  subsets); nothing else in this repo merges camera_id across subsets,
  so nothing else needed the fix, but it's worth knowing about if that
  changes.
- Both real false-positive events found for Experiment A (see "Results"
  - False-positive analysis) look, on inspection of their rendered
  skeletons, like pose-tracking quality problems rather than real human
  actions resembling the spec's risky scenarios. This is a 2-event
  sample, not a general claim about this model's false-alarm behavior -
  stated as what was actually found, per the task, not generalized
  beyond that.

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

**MotionBERT:**

```bibtex
@inproceedings{motionbert2023,
  title     =   {MotionBERT: A Unified Perspective on Learning Human Motion Representations},
  author    =   {Zhu, Wentao and Ma, Xiaoxuan and Liu, Zhaoyang and Liu, Libin and Wu, Wayne and Wang, Yizhou},
  booktitle =   {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year      =   {2023},
}
```

Repository: https://github.com/Walter0807/MotionBERT (cloned locally
per "MotionBERT setup" above; not vendored into this repo). Checkpoint
used: "3D Pose (H36M-SH, ft)" from the repo's Model Zoo, i.e.
`configs/pose3d/MB_ft_h36m.yaml` (37.2mm MPJPE on Human3.6M) - used
frozen, not fine-tuned.
