# Model development and validation

Use Python 3.11 from the project `.venv`. Generated datasets, weights, arrays,
plots and videos remain under ignored directories. Legacy A/B checkpoints are
preserved separately from the versioned `research_v1` workflow.

## 1. Diagnose the original inputs

```powershell
.\.venv\Scripts\python.exe scripts/audit_pose_quality.py
```

The audit covers the original 50-file training sample and all test pose files.
It records missing anchors, extreme normalized coordinates, score outliers and
window/mask alignment. Inspect `outputs/logs/pose_quality_audit/audit.json` and
`window_quality.csv`.

The original features discarded visibility masks. The new pipeline sanitizes
unusable coordinates before interpolation/smoothing, normalizes only windows
with supported shoulders and hips, zeros unsupported feature inputs, and uses
a visibility-aware reconstruction loss. Velocity masks require both endpoints.
Quality thresholds are fixed in `configs/research_v1.yaml`; they are not tuned
against test labels.

### Coordinate interpretation

The authors describe XYC records. In the downloaded files, shoulder-to-hip
motion is dominated by the first coordinate in all six sampled cameras, and
rendering that coordinate horizontally produces sideways bodies. The authors'
published skeleton examples are upright. The new configuration therefore
explicitly interprets RetailS coordinates as `yx` and converts them to `xy`.
This is an empirical, dataset-specific interpretation with visual support;
source RGB/calibration and upstream confirmation are unavailable. It is not an
automatic rotation inferred independently for each test clip. YOLO exports
already use `xy` and are never swapped.

Source: https://github.com/TeCSAR-UNCC/RetailS

## 2. Obtain and validate MotionBERT

```powershell
git clone --depth 1 https://github.com/Walter0807/MotionBERT.git third_party/MotionBERT
.\.venv\Scripts\python.exe scripts/prepare_motionbert.py
.\.venv\Scripts\python.exe scripts/validate_research_inputs.py
```

Skip cloning if the checkout already exists. The downloader uses the author's
Hugging Face repository linked by the official MotionBERT README, pins its
revision, and checks the LFS SHA-256 and byte count. It downloads the H36M
finetuned **pose** checkpoint, not a representation-pretraining checkpoint.
Provenance is saved to `outputs/logs/motionbert_setup.json`.

Pass: direct limb mapping is exact, normalization matches the official utility,
model parameters are frozen, outputs are finite and root-relative, and the
six camera previews are plausible. COCO has no exact H36M head joint: the head
proxy and domain shift remain limitations. No RetailS 3D ground truth is
available, so this is structural/visual validation rather than 3D accuracy.

Inspect `outputs/logs/research_input_validation.json` and
`outputs/plots/research_input_validation/`.

Source: https://github.com/Walter0807/MotionBERT

## 3. Run matched A-E experiments

```powershell
.\.venv\Scripts\python.exe scripts/run_research_phase.py --stage matched
```

A: 2D positions; B: 2D positions and velocity; C: lifted 3D positions;
D: lifted positions, velocity and geometry; E: D features with temporal attention.
All use the same 10 sampled normal pose files, retained windows and grouped
source-video split. Evaluation reads all 47 real-world and 624 staged pose files.
Rejected windows remain unscored and are counted by class; they are not normal
predictions. All five models are evaluated on identical retained test windows.

The same hidden width/depth and 15-epoch budget are used. Input/output parameter
counts depend on feature dimension. Checkpoints are selected by normal validation
loss. Thresholds use p95 of the selected model's **normal validation errors**.
Test labels do not select thresholds, epochs or architectures.

Caches include source-data, implementation and MotionBERT-weight fingerprints.
Changed inputs create different caches; incompatible completed runs fail instead
of silently overwriting existing results. Each run logs its source-video split,
learning curve, selected epoch, subset metrics and source-linked predictions.

## 4. Expand normal-training coverage and repeat seeds

```powershell
.\.venv\Scripts\python.exe scripts/run_research_phase.py --stage expanded
```

The position-only A family is declared before these test results: 100 training
pose files, model seeds 0/1/2, and the same fixed source-video split. Independent
worker processes keep RNG states separate and reuse verified input caches.
Logs are under `outputs/logs/research_v1/jobs/`.

Report the three individual results and their mean, standard deviation and
range. Three seeds and overlapping windows do not establish statistical
significance or unseen-store generalization. RetailS is a single-store dataset.
This expanded comparison is separate from the matched 10-file A-E experiment.

To run stages 3 and 4 together, use `--stage all`. Completed compatible runs
are verified and reused. Do not edit the versioned training/processing code or
configuration during a run; create a new version for another protocol.

## 5. Run external-video inference

```powershell
.\.venv\Scripts\python.exe scripts/run_video_inference.py
```

The predeclared default is `matched_d_seed0`, independent of its measured test
metrics. Cached YOLO tracks are validated and reused, including the two overhead
camera profiles. All seven retail videos are processed through the same quality
preprocessing, frozen lift, features and scoring used in training.

Outputs are browser-compatible H.264 videos and per-window JSON records under
`outputs/video_inference/matched_d_seed0/`. Every output is fully decoded to
check its frame count, dimensions and FPS. Scores refer to completed windows and
expire after one stride; a later tracking gap cannot inherit an old result.
Gray tracks are unscored. Above-threshold scores are not theft probabilities.
Processing is retrospective/offline, including full-track interpolation and
smoothing; this is not a causal streaming implementation.

The external videos have video-level class labels, without person/interval
annotations. Their outputs support qualitative review, not measured detection
accuracy. Track identities and false person detections still require review.

## 6. Summarize and reproduce the notebook

For validated Colab results, first register the selected import without replacing
the stopped CPU checkpoints:

```powershell
.\.venv\Scripts\python.exe scripts/integrate_colab_results.py outputs/colab_results/629e1d946e23
```

The registry `outputs/research_runs.json` explicitly selects imported model and
derived log directories. Analysis verifies their hashes before use. Unregistered
matched runs retain their original paths. Original Colab reports stay intact;
compatible window CSVs and histories are derived under the import folder.

```powershell
.\.venv\Scripts\python.exe scripts/evaluate_frame_diagnostic.py
.\.venv\Scripts\python.exe scripts/summarize_research_phase.py
.\.venv\Scripts\python.exe scripts/analyze_research_errors.py
.\.venv\Scripts\python.exe scripts/compare_expanded_features.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe scripts/run_model_notebook.py
```

The summary rejects mismatched model hashes, splits, window identities or labels.
It writes comparisons, per-camera metrics, legacy scores restricted to the same
retained windows, learning curves and three-seed results. It requires all eight
original model runs and seven verified videos unless `--allow-incomplete` is explicit.
Once any expanded B run is registered, all three B seeds are required as well
(eleven runs total). `compare_expanded_features.py` requires all six imported A/B
runs and a current frame diagnostic; it verifies shared windows, split/config,
model hashes and sampled GPU-to-CPU score portability.

Inspect `outputs/logs/research_v1/results.md`, `final_report.json`,
`comparison.csv`, `per_camera_metrics.csv`, and the model-development notebook.
The report distinguishes conditional recall on retained windows from candidate
window recall, which counts rejected positive windows as misses. Neither is a
person-specific theft localization metric.

Legacy and new runs differ in masking, coordinates, filters, training sample,
checkpoint selection and calibration. Their differences are not a single-factor
ablation. Weak scores and failed hypotheses must remain in the results.

### Secondary frame-level diagnostic

Window selection produces a heavily anomaly-rich cohort. To expose that bias,
a separate exploratory diagnostic takes the maximum score over overlapping
windows and people at each source frame. The same aggregation is applied to
normal validation windows to calibrate p95; no test labels tune the threshold.
Training pose-file parts remain separate frame timelines.

This diagnostic is reported alongside the predeclared window experiment rather
than replacing it. Unknown frames remain unscored; report conditional metrics,
full annotation coverage, and recall with all unscored positive frames counted
as misses. These retrospective video-frame scores do not establish person-level
localization or causal real-time performance. Inspect `frame_comparison.csv` and
`frame_diagnostic/report.json` under the research output folder.

The frame diagnostic spreads each offline window score over its full interval;
the video overlay displays completed-window scores with a one-stride expiry.
These are separate evaluation/display protocols, not instant theft timestamps.
The frame threshold is calibrated separately because maximum aggregation changes
the normal validation score distribution.

### Imported expanded-run findings

The three Colab A runs have been integrated and evaluated. They score
34,537/45,759 annotated frames (75.5% coverage). Mean frame recall is 10.3% on
scored anomalous frames, or 8.9% over all annotated anomalous frames, with mean
frame ROC-AUC 0.520. There are 2,938 unscored anomalous frames. These secondary
frame figures use separately calibrated normal-validation frame thresholds;
they do not replace the original 12.12% mean retained-window recall.

Real-world camera 6 has 24.5% score coverage and a mean 47.2% false-positive rate
on scored normal frames. This identifies an investigation target, not proof of
a camera or pose failure. Review all cameras and their sample sizes in
`outputs/logs/research_v1/error_analysis/frame_camera_summary.csv`.

Inspect `error_analysis/analysis.md`, `score_quantiles.csv`, pose support/movement
summaries, score-distribution plots and the traceable FP/FN strips. Normal and
anomalous test scores overlap strongly. Test-normal window false-positive rate
is 16.7%, versus about 5% in normal calibration. This documents calibration
mismatch without establishing its cause.

The next proposed controlled experiment compares expanded A with B (positions
plus joint velocities), holding cohort, split, masks, training settings and
seeds fixed. No further training is started by the report scripts. Use normal
validation for calibration/checkpoint selection and seek fresh confirmation
data before repeated anomaly-based tuning. GPU/CPU backend and cohort changes
mean the existing matched-versus-expanded comparison is not a causal ablation.

That expanded A/B follow-up is now complete. B mean retained-window recall is
11.14% and F1 0.1987, versus A 12.12% and 0.2144; ROC-AUC is effectively unchanged
(0.5212 versus 0.5211). Mean scored-frame recall is 9.91% versus 10.27%.
See `expanded_ab_comparison/comparison.md` for per-seed metrics and limitations.
The motion-feature hypothesis was not supported for recall/F1 under this
protocol. Audit pose/label alignment and seek independently reviewed person/time
annotations on separate retail clips before more architecture or threshold tuning.

## 7. Alignment audit and pilot observations

Follow [VALIDATION_REVIEW.md](VALIDATION_REVIEW.md). The structural alignment
audit checks original pose/mask hashes, source frame IDs, inclusive labels and
retained person-window support. It currently passes 671 pose/mask pairs and
1,856 retained windows. It cannot establish visible action timing without RGB.

The local results page provides seven retail clips with annotated model videos
and original source views. Sampled AI observations remain uncertain and cannot
supply independent person/action reference labels. These previously inspected
clips are a pilot, not independent confirmation data.

## 8. Frozen label and failure audit

`scripts/audit_model_failure_modes.py` checks original source hashes and compares
all six expanded A/B runs on the same 1,856 retained test windows. It exports
per-window label overlap, feature-mask support and model agreement; per-source
label inventories; contiguous mask-interval score coverage; and a diagnostic
queue of shared misses and false alarms. Outputs live in
`outputs/model_failure_audit/`; the read-only results page and model-development
notebook include the findings.

All six runs miss 1,476 of 1,754 positive windows and flag 13 of 102 normal
windows. Of the positive windows, 1,474 contain mixed positive/normal frame
labels and 478 overlap a positive frame containing multiple raw person tracks.
The primary A seed 0 model alerts on only 9.3% of the 280 fully positive windows,
so mixed window labels alone do not explain weak recall. These are descriptive
test diagnostics, not verified causal explanations.

The source masks contain 239 positive frames without a raw person observation.
Existing quality gates reject 245 positive candidate windows. These counts and
retained-window misses measure different populations and must not be summed.
Contiguous positive-mask intervals are not independent theft events, and
retrospective score overlap cannot establish detection delay or acting-person
identity.

This phase creates no new verified action labels and changes no model or
threshold. Local RetailS data lacks source RGB and person-specific action
labels; independent action validation remains incomplete. Preserve A/B as
baselines and use separate person/time-labeled RGB validation and confirmation
data before iterative action-based tuning. New model proposals remain untested
until evaluated under that protocol.
