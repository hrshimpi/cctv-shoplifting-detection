# Supervised staged clip classification

## Scope and source

This is a new exploratory clip-classification experiment, separate from the
RetailS pose anomaly baselines. It predicts the publisher's normal versus
shoplifting clip class; it does not localize an acting person or exact action.

Source: Ansari, Mohd. Aquib; Singh, Dushyant Kumar (2023), *Shoplifting Dataset
(2022) - CV Laboratory MNNIT Allahabad*, Mendeley Data V1,
[doi:10.17632/r3yjf35hzr.1](https://data.mendeley.com/datasets/r3yjf35hzr/1).
The publisher licenses this release under CC BY 4.0. Attribution is retained here
and in the data report. No data or weights are committed to git.

The archive was downloaded through the publisher's public browser link because
direct shell HTTP requests returned 403. Its size is 760,562,047 bytes and SHA256
matches the publisher's metadata:
`57657afeff84c44482327c46fdbb1f45b22e9f045bf44cdd31e368b07bebc93c`.

## Data checks and frozen split

- Original files: `data/action_sources/mnnit_v1/Dataset.zip` and `extracted/`.
- 182 original clips: 90 normal and 92 shoplifting-labeled.
- All 60,702 source frames decoded successfully; source FPS is 30.
- Actual sizes include both 640x480 and 1920x1080.
- Three exact duplicate files were excluded. Eighteen sampled near-duplicate
  pairs were linked into groups that cannot cross data splits.
- Retained cohort: 179 clips. Train: 107 (52 normal, 55 positive). Validation:
  36 (17 normal, 19 positive). Test: 36 (18 normal, 18 positive).
- Split seed: 2026. First fold of five stratified group folds is test; first fold
  of four on the remaining development groups is validation.

Grouping protects the detected duplicate components and original files. The
release supplies no verified actor or session identifiers, so actor/session
independence cannot be claimed. Sampling can miss additional related clips.
Midpoint contact sheets show all 182 clips and were inspected for scene/quality
context; they are not an independent adjudication of full-clip action labels.
The setting is a staged laboratory with repeated backgrounds and clothing.

The split, descriptor and classifier settings were written before fitting in
`configs/mnnit_clip_v1.json` and `outputs/mnnit_clip_classification/frozen_split.json`.
There was no hyperparameter or threshold search. Test results must not drive
repeated tuning of this split.

## Model and runtime

The frozen MobileNetV3-small encoder uses official Torchvision ImageNet1K V1
weights. Sixteen uniformly sampled chronological frames per clip are cached at
320x240 RGB, then processed by the pretrained encoder's resize/crop and ImageNet
normalization transforms. This initial cache resizes full scenes and can distort
the original aspect ratio, especially for 1920x1080 clips; it is not an action
crop or actor-localized representation.

Each frame yields 576 features. The declared primary descriptor concatenates
their temporal mean, standard deviation and mean absolute consecutive changes
(1,728 dimensions). The appearance-only control omits changes (1,152 dimensions).
These changes are embedding differences, not optical flow or physical velocity;
the pooled descriptor also discards exact temporal ordering and direction.

A StandardScaler and class-balanced logistic regression head are fitted on
training clips only: C=1, maximum 2,000 iterations, fixed score threshold 0.5.
The encoder is not fine-tuned. The saved portable NPZ head was checked against
scikit-learn predictions; no pickle is needed for inference.

The existing environment had a CUDA torchvision build incompatible with its CPU
PyTorch runtime. A matching `torchvision==0.29.1+cpu` was installed with no other
dependencies into `outputs/runtime_rgb/`; only the RGB training script prepends
this isolated location. The virtual environment and existing pose experiments
were preserved. Pretrained weights are in `outputs/rgb_pretrained/`.

## First measured results

| Model and subset | Accuracy | Recall | F1 | ROC-AUC | FP | FN |
|---|---:|---:|---:|---:|---:|---:|
| Appearance control validation | 94.44% | 100.00% | 0.950 | 0.978 | 2 | 0 |
| Appearance and changes validation | 94.44% | 100.00% | 0.950 | 0.981 | 2 | 0 |
| Appearance control test | 66.67% | 77.78% | 0.700 | 0.750 | 8 | 4 |
| Appearance and changes test | 69.44% | 77.78% | 0.718 | 0.790 | 7 | 4 |

The primary model correctly classifies 25/36 test clips, with TP=14, TN=11,
FP=7 and FN=4. Both models fit training clips perfectly. The validation/test gap,
small cohort, repetitive scenes and unavailable actor/session IDs limit the
conclusion. A change of one clip in a 36-clip test changes accuracy by 2.78
percentage points. The difference between descriptors is not established as
statistically significant. Logistic scores are not calibrated theft probabilities.

These numbers cannot be presented as a measured improvement over RetailS: the
data, supervision, split and prediction task differ. Real-store generalization
and actor/time localization remain unverified. Existing real-store overlay
videos continue to use their original frozen pose models and are not relabeled.

## Artifacts and reproduction

Preparation: `scripts/prepare_mnnit_clips.py` verifies the publisher archive,
safely extracts it, decodes all frames, records exact/sampled hashes and creates
duplicate groups and contact sheets. The archive must already be present.

Training: `scripts/train_mnnit_clip_baseline.py` checks source hashes, freezes
the split and protocol, reuses verified features, fits the two fixed heads, and
saves predictions and full metrics without altering pose artifacts.

```powershell
.\.venv\Scripts\python.exe scripts/prepare_mnnit_clips.py
.\.venv\Scripts\python.exe scripts/train_mnnit_clip_baseline.py
```

Results live in `outputs/mnnit_clip_classification/`: `data_report.json`,
`inventory.csv`, `duplicate_pairs.json`, `frozen_split.json`,
`feature_provenance.json`, `evaluation.json`, `predictions.csv`,
`appearance_and_changes_model.npz`, `appearance_control_model.npz` and the RGB
and feature caches. Large assets remain ignored under `data/action_sources/`
and `outputs/`.

The next research gate is broader source/session-disjoint labeled retail video
validation. This lightweight baseline completes the first supervised feasibility
test; a stronger temporal network or further tuning requires a fresh protocol
and independent evaluation rather than optimizing this exposed test split.
