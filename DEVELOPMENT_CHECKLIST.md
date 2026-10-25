# Development checkpoints and how to verify them

Run commands in PowerShell from the repository directory. Use the existing
Python 3.11 environment explicitly:

```powershell
Set-Location 'D:\MS-UTA\Third-Sem\Computer Vision\ShopLifting\cctv-shoplifting-detection'
$env:YOLO_CONFIG_DIR = Join-Path (Get-Location) 'outputs\yolo_settings'
```

This checklist describes the current local development milestones. The main
README's historical results are not evidence that the corresponding datasets,
weights, or experiments are available or reproduced in this checkout.

## 1. Environment and retail-video inputs

```powershell
.\.venv\Scripts\python.exe scripts/check_project_setup.py
```

Pass: exit code 0 and `ready_for_2d_pose_development: true`. All seven input
videos should be readable: five Shoplifting and two Normal, at 30 FPS. There
are currently 300 sampled images per class. The images are for inspection;
temporal pose extraction uses the original video frames, not the sparse JPEGs.

Missing RetailS/MotionBERT/trained checkpoints are reported separately. They
do not block the 2D pose milestone. Raw datasets, model weights, and generated
outputs must stay gitignored.

## 2. Real-video 2D poses and tracking

First-run commands (YOLO weights download automatically if absent):

```powershell
.\.venv\Scripts\python.exe scripts/extract_video_poses.py --video data/external_video/videos/shoplifting/Shoplifting033_x264.mp4
.\.venv\Scripts\python.exe scripts/extract_video_poses.py --video data/external_video/videos/normal/Normal_Videos_100_x264.mp4
```

Each video gets an `outputs/poses/<video_stem>/` directory containing:

- `poses.json`: person ID -> zero-based frame ID -> 51 numbers (17 joints,
  each with pixel x, pixel y, and confidence), plus a null `scores` field.
- `summary.json`: source metadata, tracked-frame coverage, observation counts,
  preprocessing settings and smoke-test window count.
- `skeletons.mp4`: original frames with skeletons and track IDs, at source FPS.
- `preview.jpg`: one representative overlay for quick inspection.

Exports are protected from accidental reruns. Existing outputs can be checked
without re-running inference:

```powershell
.\.venv\Scripts\python.exe scripts/extract_video_poses.py --verify-only --out-dir outputs/poses/Shoplifting033_x264
.\.venv\Scripts\python.exe scripts/extract_video_poses.py --verify-only --out-dir outputs/poses/Normal_Videos_100_x264
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Automated pass: `PASS`, finite 17-joint poses, confidence in [0,1], frame IDs
within the source, and a fully decodable skeleton video matching source frame
count, dimensions, and FPS. The synthetic regression tests need no model or
network. They check source FPS preservation, missing-video errors and invalid
pose-export rejection, and that temporal windows do not bridge tracking gaps.

Manual pass: play each `skeletons.mp4`. Arms/legs should follow the visible
person, and the main person's ID should remain stable while continuously
visible. Check occlusions, partial bodies, mannequins/clothing displays and
people passing one another. **Track count is not a count of unique people.**
Background false detections and fragmented tracks can increase that number.
Do not equate file validity or high frame coverage with detection accuracy.

To compare another confidence threshold without overwriting the baseline:

```powershell
.\.venv\Scripts\python.exe scripts/extract_video_poses.py --video data/external_video/videos/shoplifting/Shoplifting033_x264.mp4 --conf-threshold 0.5 --out-dir outputs/poses/Shoplifting033_conf05
```

This stage performs pose estimation/tracking only. It does not predict theft.
Do not interpret the overlay color or a dataset video label as a per-frame
anomaly prediction. Recorded local results are in
`outputs/logs/development_pose_milestone.json` once the runs are verified.

Verified local runs:

| Video / setting | Decoded preview frames | Tracks | Preprocessing windows |
| --- | ---: | ---: | ---: |
| Shoplifting033, confidence 0.3 | 899 | 19 | 83 |
| Normal_Videos_100, confidence 0.3 | 627 | 17 | 74 |
| Shoplifting033, confidence 0.5 | 899 | 8 | 75 |

All three exports preserve 30 FPS and passed finite-pose and full preview
decode checks. All four regression tests passed. On the sampled clothing-store
overlay, confidence 0.5 removes a false skeleton on hanging clothes while the
longest track still spans all 899 frames. This is a development comparison,
not measured precision/recall. Review the entire videos before selecting
settings for the remaining clips; camera cuts and short/background tracks
still need attention. See `outputs/logs/pose_threshold_comparison.jpg`.

## 3. Temporal preprocessing and pose quality

The extraction command already runs a preprocessing smoke test using
`configs/preprocessing.yaml`: confidence mask, interpolation, smoothing,
normalization, 30-frame windows and stride 15. Check the printed window count
and `summary.json`; it should be nonzero for sufficiently long tracks.

The model-development notebook now persists the longest-track inspection windows
with video/track/frame provenance, verifies their saved round-trip, and shows
before/after skeletons. It also exports finite A/B feature tensors. These are
external inspection outputs, not a labeled training dataset. The next
development checkpoint was a reusable batch export. This is now implemented
in `scripts/run_external_preprocessing.py` and `src/data/external_sequences.py`.
Tests should verify shape
`(30, 17, 3)`, finite coordinates, matching visibility masks, and that windows
never cross gaps in frame IDs. Missing joints must retain their visibility
information. The current extraction step reports window counts; it does not
yet persist the window tensors; the notebook exports the longest-track example
under `outputs/model_development/`. See `notebooks/README.md` for launch and
fresh-kernel verification instructions.

After manual quality review, run the pose command for each remaining video
using the same agreed settings. Keep each video's outputs separate.

### Batch temporal-export checkpoint

```powershell
.\.venv\Scripts\python.exe scripts/run_external_preprocessing.py --extract-missing
.\.venv\Scripts\python.exe scripts/run_external_preprocessing.py --verify-only
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The first command reuses matching pose exports and generates missing ones at
the provisional detection confidence 0.5. The second runs no inference. It
checks source/pose hashes, preprocessing implementation/configuration hashes,
NPZ checksum, dimensions, finite values, frame continuity, track provenance,
feature alignment and technical quality thresholds. Changed inputs or code
cause stale exports to fail; choose a new `--out-root` to rebuild them.

Per-video outputs under `outputs/external_sequences/conf05/<class>/<video>/`
contain `windows.npz` and `manifest.json`. Windows preserve normalized
keypoints, visibility masks, exact source frame IDs, track IDs and aligned
A/B features. Manifests retain source identity, settings, per-track counts,
filtering counts and manual-review status. `batch_report.json` at the output
root summarizes every video. The notebook's section **8b** verifies and plots
that report, including warnings for videos with low tracked-frame coverage.

Default filters require mean usable-joint visibility of at least 0.5 and all
four shoulder/hip anchors usable in at least half the window's frames.
Configure these using `--min-visible-fraction` and `--min-anchor-fraction`,
with a separate output root for comparisons. Retained windows have **not**
been certified as correct person tracks or theft events. Video labels remain
metadata: no per-window anomaly labels or train/test split are invented, and
`training_ready` remains false. Overhead views still need pose-quality review.

Smoothing now operates on each contiguous frame-ID segment independently,
so poses on opposite sides of a gap cannot influence each other. Regression
tests cover this fix, low visibility, missing torso anchors, feature alignment,
frame gaps, saved round-trips and changed source detection.

### Overhead-camera development profiles

The original uniform nano/confidence-0.5 export remains under `conf05`.
Sampled overlay review and quantitative comparisons are recorded under
`outputs/tracking_review/`. The stronger small-model/confidence-0.15 candidates
raise tracked-frame coverage from 26% to 76% for Shoplifting017 and from 30%
to 93% for Shoplifting031, retaining 18 and 22 windows instead of one each.
Some ID changes remain; these are development improvements, not measured
precision/recall or a certification of correct identities.

```powershell
.\.venv\Scripts\python.exe scripts/run_external_preprocessing.py --profiles configs/external_video_profiles.yaml --out-root outputs/external_sequences/candidate_profiles --extract-missing
.\.venv\Scripts\python.exe scripts/run_external_preprocessing.py --profiles configs/external_video_profiles.yaml --out-root outputs/external_sequences/candidate_profiles --verify-only
```

The candidate-profile export contains 378 technically retained windows across
the same seven videos. Section 8c of the notebook shows the comparison, using
identical source frames and joint-display threshold for the side-by-side plots.
Do not compare different thresholds solely by the number of detected tracks;
inspect occlusions, false detections, visibility and continuity. All external
video labels remain video-level and manual approval stays pending.

## 4. Main training data and 2D baseline

RetailS has now been downloaded from the authors' public Drive link and
verified locally: 942 train pose files, 47 real-world test pose/mask pairs,
and 624 staged test pose/mask pairs. The full EDA and a five-file training
preprocessing smoke test ran on the real data. The schema audit samples five
pose files per subset; it does not claim every coordinate was audited.

```powershell
.\.venv\Scripts\python.exe scripts/prepare_retails.py
.\.venv\Scripts\python.exe scripts/verify_retails.py
```

The preparation command reuses existing files and logs the downloaded archive
hash. Install `requirements-data.txt` only if the download dependency is needed.
RetailS source confidences can be slightly greater than one; the audit records
them rather than incorrectly applying YOLO sigmoid-score assumptions. Existing
preprocessing treats NaN confidence as missing.

Training now groups windows by camera/source-video ID, keeping pose-file parts
together, and saves the disjoint split in `split_manifest.json` and the model
checkpoint. The actual validation-window fraction is reported because groups
have different lengths. A/B no longer import optional MotionBERT code at startup.
Missing test masks now stop evaluation rather than silently becoming normal
labels. These fixes are covered by regression checks.

Experiment A has now run for 15 epochs on 50 sampled normal-training pose
files (22,493 windows). There are 44 source-video groups / 18,459 windows
in training and 5 groups / 4,034 windows in validation. The full 47-file
real-world and 624-file staged test subsets were evaluated (2,120 windows).
Combined measured precision is 0.9752, recall 0.0590, F1 0.1113 and ROC-AUC
0.5413 at the training-normal p95 threshold. This is a **weak baseline**:
most anomalous windows are missed. The test windows are heavily anomaly-rich,
so interpret PR-AUC alongside class prevalence and do not infer deployment
readiness from precision alone. No test labels were used to tune the threshold.

Inspect `outputs/logs/experiment_a/training_log.csv`, `split_manifest.json`
and `eval_metrics.json`, or notebook section **10b**. The source datasets and
model checkpoints remain ignored by Git. These results are for one sampled-data
2D baseline, not a completed A–E comparison or unseen-store validation.

RetailS is the training dataset expected by the existing code. Obtain and
extract its archive under `data/retails/`, following the README. Check it with:

```powershell
.\.venv\Scripts\python.exe scripts/run_eda.py
.\.venv\Scripts\python.exe scripts/run_preprocessing.py --subset train --limit 5
```

Pass: the loader discovers real pose files; EDA reports actual schema/counts;
preprocessing produces finite tracks/windows. Stop to investigate missing
files or zero usable windows before training.

```powershell
.\.venv\Scripts\python.exe scripts/train.py --experiment a
.\.venv\Scripts\python.exe scripts/evaluate.py --experiment a
```

Pass: an actual trained checkpoint and finite reconstruction errors, followed
by metrics on held-out labeled data. Repeat for B. Keep splits grouped by
source video/store; never split neighboring frames across training and test.
Set thresholds without consulting test labels. Report subset size and any
sampling limits alongside results.

### Matched A/B experiment and error-analysis checks

Experiment B adds velocity features (58 values per frame versus A's 34).
Both runs use the same hidden width/depth, 50 sampled normal pose files,
source-video split, 15 epochs and final-epoch checkpoint policy. The larger
feature vector also changes the input/output projection parameter counts.

```powershell
$env:OMP_NUM_THREADS='2'
$env:MKL_NUM_THREADS='2'
.\.venv\Scripts\python.exe scripts/train.py --experiment b
.\.venv\Scripts\python.exe scripts/evaluate.py --experiment b
.\.venv\Scripts\python.exe scripts/compare_2d_experiments.py
```

Skip training when inspecting existing checkpoints. The comparison requires
both saved evaluations and rejects mismatched splits, settings or test-window
identities. It reproduces the saved subset metrics and retains the original
normal-only p95 thresholds, with no test-label tuning.

Inspect `outputs/logs/ab_comparison/comparison.md` and notebook section **10c**.
`window_predictions.csv` links every score to its source pose file, camera,
person and frame range. The comparison also saves confusion counts, class
prevalence, per-camera errors, score quantiles and reconstructed skeleton strips.
The frame-mask labels do not identify which person performed an anomalous act.

Current measured combined results: A recall 0.0590 / F1 0.1113 / ROC-AUC
0.5413; B recall 0.0530 / F1 0.1005 / ROC-AUC 0.5284. Velocity did not improve
this run. A misses 1,881 anomalous windows with 3 false positives; B misses
1,893 with 4 false positives. Both use 2,120 identical test windows. These
results do not support deployment readiness.

Pass: matching split/window checks succeed, metrics are reproduced, all scores
are finite, and error examples can be traced to their original pose sequences.
Low recall or weak ROC-AUC is a measured modeling limitation, not a passing
accuracy result. Do not choose thresholds or checkpoints using these test labels.

## 5. 3D lifting

Follow the README to obtain the official MotionBERT code and the compatible
H36M pretrained pose checkpoint, then run:

```powershell
.\.venv\Scripts\python.exe scripts/validate_joint_mapping.py
```

Pass: mapping and lifted skeleton previews are plausible; outputs are finite
and joint ordering matches the model. Inspect left/right limbs and coordinate
normalization before training C/D/E. Missing checkpoints are a setup issue,
not a successful 3D test.

## 6. 3D experiments and video inference

Train/evaluate C, D and E through the existing `--experiment` options. Compare
them with A/B using comparable splits and report differences in sample sizes.
Once the matching checkpoints exist:

```powershell
.\.venv\Scripts\python.exe scripts/run_demo.py --video data/external_video/videos/shoplifting/Shoplifting033_x264.mp4
```

Pass: video inference loads real weights, scores completed windows and produces
a playable annotated video with correct timing. Repeat on Normal footage;
inspect false alarms and missed events. The seven UCF-Crime videos support
an external inspection, not a robust benchmark. Video-level labels do not
locate the exact theft interval; temporal evaluation needs event annotations.

Finish with reproducible commands, actual metrics, plots, failure examples,
limitations and the model-development report. Mark milestones complete
only after their checks pass and preserve generated summaries for evidence.
