# Google Colab training

Local expanded training has been stopped. Existing checkpoints and the completed five matched A–E experiments are preserved. The remaining training task is the expanded 2D BiGRU experiment with 100 normal RetailS pose files and three seeds (0, 1, 2).

## Files

- `outputs/colab_handoff/Retail_Shoplifting_Colab_Training.zip`: prepared training/validation/test tensors, visibility masks, metadata, split, provenance, trainer and model code.
- `notebooks/Retail_Shoplifting_GPU_Training.ipynb`: the notebook to upload to Colab. A copy is also inside the ZIP.
- `outputs/colab_handoff/handoff.json`: ZIP size, SHA256, exact cohort and split counts.

No raw videos, extracted JPEGs, virtual environment, credentials or pretrained MotionBERT weights are needed for this training stage. The original dataset remains local. All generated data and weights remain under git-ignored `outputs/`.

## Run in Colab

1. Upload the training ZIP to the top level of Google Drive **My Drive**. Do not extract it there.
2. Open https://colab.research.google.com/ and choose **File → Upload notebook**. Upload `Retail_Shoplifting_GPU_Training.ipynb`.
3. Select **Runtime → Change runtime type → GPU** (T4 is suitable when available).
4. Run notebook cells in order. Mount Drive when prompted. Check the integrity output, normal-only training labels, disjoint source-video groups and available CUDA GPU.
5. Run training. Each of three seeds must reach **15/15**. Models use the existing batch size 32, hidden size 128, two layers, dropout 0.2 and learning rate 0.001. The best epoch is selected using normal validation loss; the threshold uses the 95th percentile of normal validation scores. Test labels do not select the model or threshold.
6. Inspect the three-row results table and mean/standard deviation. Run the final export cell to download **Retail_Shoplifting_Colab_Results.zip**.

Checkpoints are saved after each epoch in `My Drive/Retail_Shoplifting_Training_v1`. If disconnected, reconnect with a GPU, rerun the setup cells and the same training cell. It resumes completed epochs and skips completed compatible seeds. Keep the output path unchanged. Use a new output directory if intentionally changing code/config/data. Do not mix different training protocols in one directory.

The stopped local checkpoints contain best weights but lack optimizer and RNG state. Colab therefore starts these three runs afresh; it cannot exactly continue the stopped local epoch. New Colab checkpoints include optimizer and RNG state. CPU and GPU numerical results can differ; GPU access and speed depend on Colab availability. The notebook retains Colab's existing PyTorch installation.

## Return the trained models

Copy `Retail_Shoplifting_Colab_Results.zip` to the repository's `outputs/colab_handoff/` directory, then run from the repository root in PowerShell:

```powershell
.\.venv\Scripts\python.exe scripts\import_colab_results.py outputs\colab_handoff\Retail_Shoplifting_Colab_Results.zip
```

This validates the bundle identity, checksums, model shapes, all three 15-epoch histories, calibration scores and test labels/scores against the local prepared data. It imports into a new `outputs/colab_results/` folder without overwriting local runs. Keep the printed import path. We can then integrate these results with the matched experiments, run frame-level diagnostics and regenerate the local development notebook and reports. The returned GPU models are portable to the local CPU environment.

To integrate a validated import into the local analysis, use its printed folder:

```powershell
.\.venv\Scripts\python.exe scripts\integrate_colab_results.py outputs\colab_results\629e1d946e23
.\.venv\Scripts\python.exe scripts\evaluate_frame_diagnostic.py
.\.venv\Scripts\python.exe scripts\summarize_research_phase.py
.\.venv\Scripts\python.exe scripts\analyze_research_errors.py
```

The integration writes an explicit hash-checked run registry and derived reports.
It preserves original imported files and stopped local training checkpoints.

## Interpretation and scope

The bundle contains 41,841 accepted normal windows from 100 pose files and 1,856 accepted test windows. Source-video grouping separates training and validation. Exact split counts and rejected-window counts are in the manifest and notebook output. Quality-gated metrics exclude rejected windows; excluded windows must not be counted as normal. Test anomaly labels originate from video frame masks and do not identify a particular person's theft action. Retained windows are strongly imbalanced, so high precision alone is insufficient evidence of useful detection.

This handoff preserves the current RetailS-specific `yx` coordinate interpretation and pose quality gates. The five matched 2D/3D runs and mixed 3D visual quality remain separate local evidence. This ZIP trains the remaining expanded 2D experiment; it does not repeat 3D lifting or resolve existing dataset/pose limitations.

To regenerate the handoff after a deliberate preprocessing change:

```powershell
.\.venv\Scripts\python.exe scripts\prepare_colab_bundle.py
```

The builder checks cached preprocessing implementation hashes and array checksums before exporting. It does not start training.
