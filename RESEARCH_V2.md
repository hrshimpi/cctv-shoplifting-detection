# research_v2: frame-level protocol and step-by-step accuracy work on RetailS

Run it with `notebooks/CCTV_Shoplifting_AllInOne_Colab_v2.ipynb` (section 8) on a Colab GPU, or from a shell:

```bash
python scripts/run_research_v2.py --step 0   # then 1, 2, ... 6 (or --step all)
```

Config: `configs/research_v2.yaml`. Outputs: `outputs/{cache,checkpoints,logs}/research_v2/`. Nothing under
research_v1, `quality_v1` or experiment_a-e is written. The MNNIT clip classifier is not involved.

## Protocol

* **Data:** the full real RetailS test set (`test_realworld`, `test_staged`). Every labelled frame is
  scored. Frames with no retained person get the subset minimum score. Training and calibration use RetailS_train
  (normal only). No synthetic data is used for any reported number.
* **Frame scores:** each person-window score is written to the frames it covers (max where windows overlap),
  optionally smoothed per track, then the frame takes the max over persons, optionally smoothed again. Neither
  smoothing pass crosses a gap. AUC is reported per subset. "combined" is informational only.
* **Dev folds (step 0):** test videos are split into folds A and B, grouped by (subset, camera, video) and
  stratified per camera (`folds.json`). Each choice is made on one fold and applied to the other, so every
  reported video is scored with a setting chosen without its labels (cross-fitting).
* **Thresholds:** p95 of the frame scores of normal training-validation windows. Scores are calibrated
  (log, then standardised on normal validation, optionally per camera) and shifted so that threshold = 0.
* **Metrics:** frame ROC-AUC, PR-AUC with base rate, balanced accuracy, recall at 5% FPR, specificity at 90%
  recall, per-camera AUC mean/std, and a 95% CI from a video-grouped bootstrap (1000 draws). All are reported
  as mean ± std over seeds 0, 1, 2. The secondary window-level AUC and the fully-positive-window AUC are kept.
* **Keep or revert:** a step's change is adopted on a fold only if it raises that fold's dev objective
  (mean per-subset frame AUC, averaged over seeds). `stepN_selection.json` logs every candidate and flags
  drops larger than the CI half-width.

## Steps

| Step | Change | Files |
|---|---|---|
| 0 | dev folds | `src/evaluation/folds.py` |
| 1 | research_v1 baseline under the frame protocol + new metrics. Reuses research_v1 `expanded_a_seed*` checkpoints read-only when present, otherwise retrains the same recipe | `src/evaluation/frame_protocol.py`, `src/evaluation/metrics.py` |
| 2 | scoring variant (mean, max-joint, top-3/5 frames, upper body, no face) × per-camera normalisation | `src/evaluation/v2_scoring.py` |
| 3 | Gaussian smoothing, sigma in {0, 2, 4, 8, 16} frames | `frame_protocol.py` |
| 4 | every train part (windows capped at 600k, uniform) + flip/scale/rotate/jitter/joint-dropout augmentation, AMP, batch 1024, cosine LR, early stopping | `src/experiments/v2_train.py`, `src/experiments/v2_data.py` |
| 5 | window 30/15 vs 24/6 vs 16/4 | `v2_data.py` |
| 6 | experiment F, STG-NF-style flow (runs automatically only on a plateau; set `experiment_f: always` or `--experiment-f always` to force it) | `src/models/stg_nf.py` |

## GPU notes

* JSON parsing and cleaning run once per file in parallel and are cached as `.npz`. If Colab disconnects, the
  next run resumes where it stopped.
* Training windows stay on the GPU as fp16 and batches are indexed on the device. The autoencoder uses
  autocast (fp16 with a GradScaler on T4, bf16 on A100). The flow runs in fp32 for log-det stability.
* One inference pass per model saves the per-joint error tensor, so steps 2 and 3 need no new inference.
