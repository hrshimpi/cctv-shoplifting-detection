# Retail shoplifting model development

Use `Retail_Shoplifting_Model_Development.ipynb` for local development and validation.

For the prepared expanded training cohort, use `Retail_Shoplifting_GPU_Training.ipynb` and [COLAB_TRAINING.md](../COLAB_TRAINING.md). It trains three seeds on a GPU, saves resumable checkpoints to Drive and exports results for local validation. `CCTV_Shoplifting_Colab.ipynb` is an older workflow.

For the follow-up with joint motion, use `Retail_Shoplifting_GPU_Training_B.ipynb` and [COLAB_EXPERIMENT_B.md](../COLAB_EXPERIMENT_B.md). Its separate ZIP and Drive output directory preserve A; B uses the identical cohort and split with 58 input features.

Section 10g shows the pose/label alignment audit and pending human observations.
Use [VALIDATION_REVIEW.md](../VALIDATION_REVIEW.md) for the local retail-video
review page and annotation export. These existing clips are a pilot, not a new
independent test set.

## Run the notebook

```powershell
Set-Location 'D:\MS-UTA\Third-Sem\Computer Vision\ShopLifting\cctv-shoplifting-detection'
.\.venv\Scripts\python.exe -m notebook notebooks/Retail_Shoplifting_Model_Development.ipynb
```

Choose **Python 3.11 (Shoplifting)**, then **Restart Kernel and Run All**. The notebook reuses verified artifacts by default. Inspect each stage's outputs and checks; enable expensive stages only for a new run.

If an older notebook is open, close it without saving and open this filename. This avoids overwriting updated source with an older browser copy.

For a new checkout, install the main requirements first, then:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-notebook.txt
.\.venv\Scripts\python.exe -m ipykernel install --prefix .venv --name shoplifting-py311 --display-name "Python 3.11 (Shoplifting)"
```

Datasets and weights are not included in Git. Follow `DEVELOPMENT_CHECKLIST.md` to obtain the data and generate missing pose exports.

## Development stages and checks

1. **Inputs:** verify video metadata, store scenes, sampled JPEGs and chronological frames.
2. **Pose estimation:** inspect YOLO Pose and ByteTrack overlays, joint visibility and tracking gaps.
3. **Preprocessing:** validate normalized skeletons, contiguous windows, masks and source provenance.
4. **External-video profiles:** compare uniform settings with overhead-camera candidates on identical frames. Review identities before approving tracks.
5. **RetailS training:** inspect dataset counts, grouped source-video splits and measured learning curves.
6. **A/B evaluation (section 10c):** compare held-out precision, recall, F1, ROC-AUC and average precision; inspect confusion counts, class balance and traceable missed-anomaly examples.
7. **Regression checks:** run the unit tests and save the current workflow report.

Training, EDA, 3D lifting and video inference have separate controls, disabled by default. Missing assets are reported explicitly. The architecture inspection cell uses an untrained instance; evaluation sections load measured results from trained checkpoints.

External video labels do not locate theft intervals. RetailS window labels use any anomalous video frame and are not person-specific annotations. These limitations apply to interpreting error examples and detection metrics.

## Reproduce the A/B comparison

```powershell
$env:OMP_NUM_THREADS='2'
$env:MKL_NUM_THREADS='2'
.\.venv\Scripts\python.exe scripts/evaluate.py --experiment a
.\.venv\Scripts\python.exe scripts/evaluate.py --experiment b
.\.venv\Scripts\python.exe scripts/compare_2d_experiments.py
```

Both trained checkpoints must exist. The comparison rejects mismatched training splits, configurations, test-window identities or labels. Thresholds retain each model's normal-only p95 calibration; no test-label threshold search occurs. Reports and individual window predictions are under `outputs/logs/ab_comparison/`.

## Execute all notebook checks

```powershell
.\.venv\Scripts\python.exe scripts/run_model_notebook.py
```

Success prints `PASS` and the number of code cells executed without errors. Artifacts are saved under `outputs/model_development/`:

- `Retail_Shoplifting_Model_Development_executed.ipynb`
- `execution_report.json` and `workflow_report.json`
- Source-linked window/feature arrays and browser-compatible video excerpts
- `Retail_Shoplifting_Model_Development.html`, when exported from the executed notebook

Keep source notebooks free of saved outputs. Datasets, trained weights, rendered videos and feature tensors remain ignored by Git. Shared processing belongs in the existing Python modules rather than a second notebook-only implementation.

## Quality-corrected research workflow

Sections 10d-10f cover the input audit, explicit RetailS coordinate interpretation,
frozen 3D validation, matched A-E experiments, expanded three-seed validation and
all seven offline inference videos. See `MODEL_DEVELOPMENT.md` for stage commands.
The original A/B measurements in 10b-10c remain reference results. New metrics
come from `outputs/logs/research_v1/final_report.json`; they are never fabricated
or inferred from a successful script execution.
