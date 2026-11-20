# Experiment B: 2D pose positions and joint motion

This experiment tests explicit motion features against completed expanded Experiment A.
Improvement is unproven. It is an exploratory follow-up after test inspection, so
fresh confirmation data will be needed before claiming generalization.

## What changes

- A inputs: 17 joints × 2 normalized coordinates = 34 features per frame.
- B inputs: the same 34 positions plus 12 joints × 2 coordinate displacements = 58 features.
- Displacements use adjacent frames with valid joint support on both sides. First-frame displacements are masked. These are body-relative changes per source frame, not physical speeds.
- Model input/output dimensions and their parameter counts increase; the BiGRU hidden size/depth and training settings remain fixed. Raw reconstruction losses have different feature composition, so compare detection metrics rather than A/B loss magnitudes.

## What stays fixed

The same 100 normal RetailS files, exact window identities and labels, 35,557 training
windows, 6,284 normal validation windows, 1,856 retained test windows, source-video
split, 30-frame length, stride 15, quality gates and position masks.

Train all three seeds (0, 1, 2), 15 epochs, batch size 32, Adam learning rate 0.001,
hidden size 128, two layers and dropout 0.2. Choose best weights using normal
validation loss and calibrate the p95 threshold using normal validation errors.
Test labels do not select epochs or thresholds. Keep Colab's installed CUDA PyTorch.

## Run one cell at a time

1. Upload `outputs/colab_handoff/experiment_b/Retail_Shoplifting_Colab_Training_B.zip` into **My Drive → Shoplifting**. Leave the ZIP compressed and keep its exact filename.
2. Upload `notebooks/Retail_Shoplifting_GPU_Training_B.ipynb` to Colab.
3. Select a GPU runtime (T4 when available).
4. Run the first cell. Authorize Drive access. Verify that the ZIP assertion succeeds.
5. Run **Extract and verify inputs**. Expect `Integrity check passed`. Cohort and split counts match A.
6. Run **Check the GPU and dependencies**. CUDA must be available.
7. Run **Train Experiment B — three seeds**. Each seed reaches 15/15. Models save in **My Drive/Shoplifting/Retail_Shoplifting_Training_B_v1**, separate from A.
8. Run **Verify and inspect results**. Expect three completed B runs, checksum checks, and an A/B mean comparison. Record all seeds, including any negative result.
9. Run **Export results** and download **Retail_Shoplifting_Colab_Results_B.zip**. Keep Drive checkpoints until local validation finishes.

Rerun setup and training cells after a disconnect to resume the last saved epoch.
Do not rename the B output directory or point the notebook at A's output directory.
Epoch checkpoints include optimizer and RNG state. Hardware/software changes may
produce numerical differences; use the same GPU type where possible.

## Return results locally

Place the results ZIP in `outputs/colab_handoff/experiment_b/`, then use:

```powershell
.\.venv\Scripts\python.exe scripts/import_colab_results.py outputs/colab_handoff/experiment_b/Retail_Shoplifting_Colab_Results_B.zip --package outputs/colab_handoff/experiment_b/retail_training_b
```

Use the printed import folder with `scripts/integrate_colab_results.py`. B uses
`expanded_b_seed0/1/2` names, so A models stay preserved. We will then add B to the
local window/frame reports and compare score coverage, recall, false alarms,
ROC-AUC and PR-AUC with A. No additional raw dataset download or pose extraction is required.

## Reproducibility

`experiment_protocol.json` freezes the configuration, cohort identities and A
baseline hashes. `baseline_a.json` contains the verified reference metrics.
Every packaged file has a SHA256 entry in `bundle_manifest.json`.
The original A ZIP, prepared tensors and trainer snapshot remain unchanged.
Generated datasets, archives and weights stay under git-ignored `outputs/`.

Builder (refuses to overwrite an already frozen B ZIP):

```powershell
.\.venv\Scripts\python.exe scripts/prepare_colab_motion_bundle.py
```

## Completed results

The returned B archive was validated and imported under
`outputs/colab_results/dc0cbd394228/`. All three seeds completed 15 epochs.
The derived reports are registered separately from A. CPU scores matched the
returned GPU scores within tolerance on 288 sampled windows.

Mean retained-window results (three seeds): A recall 12.12%, F1 0.2144 and
ROC-AUC 0.5211; B recall 11.14%, F1 0.1987 and ROC-AUC 0.5212. Explicit motion
features did not improve mean recall or F1 under this protocol. The ROC-AUC
difference is too small to claim a meaningful improvement from three seeds on
one fixed split.

Mean scored-frame recall is 10.27% for A versus 9.91% for B. Mean frame ROC-AUC
is 0.5200 versus 0.5256; both have 75.48% frame coverage. These secondary
retrospective frame metrics use their own normal-validation calibration.

Inspect `outputs/logs/research_v1/expanded_ab_comparison/comparison.md`,
`per_seed_metrics.csv`, `prediction_transitions.csv` and `comparison.json`.
Regenerate the comparison after registered input changes with:

```powershell
.\.venv\Scripts\python.exe scripts/evaluate_frame_diagnostic.py
.\.venv\Scripts\python.exe scripts/summarize_research_phase.py
.\.venv\Scripts\python.exe scripts/compare_expanded_features.py
.\.venv\Scripts\python.exe scripts/analyze_research_errors.py
```

The next investigation is pose/label alignment and independently reviewed
person/time annotations on separate retail clips. Preserve A/B as measured
baselines before further architecture or threshold tuning.
