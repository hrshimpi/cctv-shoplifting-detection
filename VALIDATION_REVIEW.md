# Retail-video validation and alignment

## Current autonomous workflow

The assistant has completed an AI-attributed visual review of all seven retail
clips. No annotation task is assigned to the user. The active `index.html` is now
a read-only results page; the original optional form is preserved separately.

Evidence: 433 source frames sampled across all seven timelines at 2 FPS, 52
overlay samples, and denser source samples around verified camera motion.
Intervening frames were decoded but not all visually inspected. All seven
action classifications remain uncertain. Four uncertain intervals describe
visible hand/body movements without asserting theft or intent. These records
are explicitly ineligible for supervised training and are not independent
reference labels. See `outputs/manual_validation/ai_review.json` and its
hash-identified validation record.

The new external scoring path excludes a verified zoom in RV002 (frames 415-449)
and pan in RV007 (406-449), splitting tracks before interpolation, smoothing and
windowing. The exclusions are engineering boundaries, not action annotations.
All six frozen A/B models have been scored on all seven clips with unchanged
normal-validation thresholds. Accepted windows change from 64 to 60 and 58 to
54 respectively; the other five clips are unchanged. Reduced coverage or fewer
flags does not establish an improvement in detection accuracy. Original RetailS
results and existing model/video artifacts are preserved.

Artifacts: `outputs/retail_pilot_analysis/report.md`, `report.json`,
`pose_visibility.csv`, `coverage_and_flags.csv`, `window_scores.csv` and the
read-only local page. The following commands reproduce the scoring/report:

```powershell
.\.venv\Scripts\python.exe scripts/validate_retail_review.py outputs/manual_validation/ai_review.json
.\.venv\Scripts\python.exe scripts/score_review_sequences.py
.\.venv\Scripts\python.exe scripts/build_autonomous_review_report.py
```

The manual-review instructions below are retained as optional reference only.

The B ZIP supplied from `ShopLifting/output/` has the same SHA256 as the already
imported archive. The existing three-seed A/B results remain the measured baselines.

## Alignment audit

Run `scripts/audit_pose_label_alignment.py` to verify original pose/mask hashes,
source frame bounds and person IDs, consecutive frame support for retained
windows, and inclusive ANY-overlap window labels.

The current audit checked 671 RetailS pose/mask pairs, 51,388 person-frame
observations and 1,856 retained windows. It found zero out-of-bounds source
frames, malformed pose shapes, window-label mismatches, missing people or
missing frame support in retained windows. Raw data contain 2,550 nonfinite
joint observations and six finite confidence values outside [0,1]. The existing
quality pipeline masks nonfinite observations and clips finite confidence; no
new preprocessing, offsets or coordinate changes are made by the audit.

Structural consistency does not verify the semantic timing of visible actions.
RetailS RGB and person-specific labels are unavailable locally. The current
dataset-specific yx interpretation remains an empirical assumption.
See `outputs/validation_alignment/report.md`, `report.json` and `per_file.csv`.

## Review the seven existing retail clips

Open http://127.0.0.1:8765/index.html in Chrome, Edge or the app browser while the
review server is running. Start it from the repository root when needed:

```powershell
.\.venv\Scripts\python.exe scripts/serve_validation_review.py
```

Keep this terminal running during review. The server binds only to localhost
and supports byte-range requests for video seeking. The package also contains
`outputs/manual_validation/index.html`; keep its media folder beside the page.
The page displays neutral case IDs, original video and
track overlays. It hides source categories and anomaly-model scores. Track IDs
come from cached pose estimates; they are not verified person identities.
Display copies normalize presentation timestamps to frame zero at the source
FPS, preserving frame order, count and resolution in both views. Original-video
and display-media hashes are recorded separately. The original inputs are unchanged.

1. Enter a reviewer name or identifier.
2. Start with **RV001**, watch the whole original video, and use the frame slider
   or previous/next-frame buttons to inspect actions.
3. Switch to **Skeletons and track IDs** to check joints, ID continuity, missing
   wrists/hips and occlusion. Switching views retains the current timestamp.
4. Assess whether this is a retail-store scene and whether tracking is usable.
5. Choose **No visible anomalous action**, **Observable anomalous action**, or
   **Uncertain**. Do not label a clip from its filename, prior model score or
   a suspicious-looking movement alone. The observations do not establish intent.
6. For an observable event, record inclusive start/end frames, an optional
   matching track ID, a concrete action description and confidence. Use uncertain
   intervals if evidence is insufficient. Do not assign an unknown actor to an
   arbitrary track.
7. Save the assessment and repeat for each case. Export **Retail_Validation_Review.json**.
   Browser saving is a convenience; retain the downloaded JSON. Partial exports
   are allowed but remain explicitly incomplete. Pending cases never become normal.
8. Return the JSON for schema/source/boundary validation and subsequent pilot
   analysis. No labels are created automatically by the preparation script.

The existing seven clips were already inspected during development and exposed
to model outputs. This is a pilot annotation exercise, not a fresh blinded or
independent test. A separate confirmation set still needs new retail clips with
documented provenance and human-reviewed labels. If a second reviewer is
available, have them review independently and retain disagreement before
adjudication. Do not mix pilot labels into reported RetailS benchmark metrics.

## Validate returned observations

From the repository root, use the downloaded JSON's actual path:

```powershell
.\.venv\Scripts\python.exe scripts/validate_retail_review.py "C:\Users\hbpra\Downloads\Retail_Validation_Review.json"
```

This verifies source/overlay hashes, source frame boundaries, known track IDs,
reviewer fields, complete-video assessments and label consistency. Accepted
observations and a validation report are saved under an annotation-hash folder
in `outputs/manual_validation/annotations/`. Uncertain and pending records are
reported separately; none is silently treated as negative. This step does not
train a model or change A/B thresholds.

## Rebuild and check

```powershell
.\.venv\Scripts\python.exe scripts/audit_pose_label_alignment.py
.\.venv\Scripts\python.exe scripts/prepare_retail_validation_review.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Rebuilding review media can change the manifest identity. Keep the original
review package while annotations are in progress. All generated media and
annotation data remain in git-ignored `outputs/`.

Browser verification checked source/overlay duration, timestamp preservation on
view changes and next-frame seeking for all seven cases. An out-of-bounds
synthetic interval was rejected. The browser checks created zero human labels;
see `outputs/manual_validation/browser_verification.json`.


## Annotated video outputs

All seven retail clips now have 1280 x 960 H.264 analysis videos at the original 30 FPS, with all 6,356 source frames preserved and fully decoded for verification. The read-only results page displays each annotated video above its original source.

Each video includes skeletons and person IDs, visible joints, mean joint confidence, wrist and normalization-anchor visibility, track observations and duration, window timing, all six expanded A/B model scores, thresholds and score/threshold ratios. A seed 0 is the declared primary alert model. Red banners indicate possible shoplifting / suspicious activity, explicitly unverified. Scores are reconstruction errors, not theft probabilities; below-threshold periods do not establish normal behavior. Camera movement pauses scoring and expired windows are not carried forward. Processing uses offline tracks, not a verified real-time pipeline.

The benchmark metrics on the panel describe held-out RetailS results, not accuracy on the current video. Raw per-frame joint coordinates, confidences, model values and states are retained in `outputs/video_analysis/RVxxx/frame_values.jsonl`; model alert intervals are exported separately as CSV. These intervals are not reference theft labels.

The renderer is `scripts/render_model_analysis_videos.py`. Aggregate integrity checks, video hashes, frame counts, thresholds and model provenance are recorded in `outputs/video_analysis/verification.json`. Encoded videos live under `outputs/manual_validation/media/` and remain excluded from git with all other outputs.
