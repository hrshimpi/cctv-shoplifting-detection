# Retail action data acquisition

## Result

A public person-window candidate was obtained from
[bassatbassat/shoplifting-vlm-finetune-ucf](https://huggingface.co/datasets/bassatbassat/shoplifting-vlm-finetune-ucf),
pinned to revision `4d378a4f2507ae90204b6d5eefeb5c38996c4951`.
This is a partial match to the requested data, not verified action-boundary ground truth.

Downloaded 24 sampled RGB clips (12 publisher-positive and 12 publisher-normal),
containing 384 sampled frames from six source videos. RGB payload size is
46,752,993 bytes (44.6 MiB), in addition to annotation tables and documentation.
Every downloaded clip passed upstream SHA256, byte-count, RGB-shape and original
frame-index checks. No dataset content is tracked in git and no training was run.

| Source video | Publisher fold-1 role | Normal windows | Positive windows |
|---|---|---:|---:|
| Shoplifting001_x264 | test | 2 | 2 |
| Shoplifting003_x264 | test | 2 | 2 |
| Shoplifting004_x264 | test | 2 | 2 |
| Shoplifting015_x264 | train_dev | 2 | 2 |
| Shoplifting016_x264 | train_dev | 2 | 2 |
| Shoplifting018_x264 | train_dev | 2 | 2 |

None of these six source-video names overlaps the previously inspected local
seven-video pilot. Because these samples were downloaded and visually inspected,
they are inspection data, not an untouched confirmation set. Fold roles are
preserved as provenance, not authorization to treat inspected test samples as a
fresh final evaluation set.

## What is actually available

The publisher states that binary person-window labels derive from manually
labeled event intervals. The release contains person IDs, window start/end frame
numbers, RGB frame indices, bounding-box fields and sampled RGB arrays. The
downloaded `box_frame` view already has a green rectangle embedded in the pixels.
The IDs belong to the publisher's tracker and do not match existing project IDs.

The complete downloaded upstream table contains 14,147 windows and 2,019 positive
labels. Of 2,466 video/person groups, 126 have both positive and negative window
labels. These are inventory facts, not proof of annotation correctness.

The public export does **not** contain the original event start/end annotations
or an annotation-adjudication record. Exported window boundaries must not be
relabeled as exact action boundaries. Original FPS is not declared in this pack,
and full source MP4 paths refer to the publisher's machine. No action times in
seconds are fabricated.

Three frames per downloaded window were visually inspected (72 frames). The six
scenes show retail counters, shelves or aisles. Inspection found occlusion, low
resolution and boxes that can partly leave the image; this does not independently
confirm theft or the correctness of acting-person labels. All candidate records
retain `independently_verified=false`, null action boundaries and
`training_eligible=false`.

No explicit license file or dataset-card license was present at the pinned
revision. Publisher clarification is needed before training or redistribution;
this acquisition does not assert such permission.

## Other sources checked

| Source | Available evidence | Why it does not fully meet this request |
|---|---|---|
| [Official RetailAction](https://huggingface.co/datasets/standard-cognition/RetailAction) | RGB views, action intervals, interaction points and subject-specific poses | Labels take/put/touch interactions, not shoplifting |
| [MNNIT Shoplifting](https://data.mendeley.com/datasets/r3yjf35hzr/2) | Staged normal/shoplifting clips | Public description supplies clip classes, not person-specific action boundaries |
| [RetailS](https://github.com/TeCSAR-UNCC/RetailS) | Pose tracks and video anomaly masks already available locally | Raw RGB and actor-specific action ground truth are unavailable locally |
| [UCFCrime2Local](https://aimagelab-legacy.ing.unimore.it/imagelab/page.asp?IdPage=30) | Spatial/temporal anomaly annotations | Its six listed categories omit Shoplifting |
| [CrimeVideoLocalization](https://github.com/benediktusbryan/CrimeVideoLocalization) | A claimed UCF test-video/bounding-box package via Google Drive | Linked Drive folder could not be inspected through the web tool; exact timing/actor schema and download eligibility remain unverified |

## Local artifacts

- Acquisition script: `scripts/acquire_action_label_sample.py`.
- Data, upstream metadata and selected rows:
  `data/action_sources/ucf_person_windows/`.
- Selected row manifest: `data/action_sources/ucf_person_windows/sample_manifest.csv`.
- Machine-readable qualification and checks:
  `outputs/data_acquisition/action_data_report.json`.
- Six visual contact sheets: `outputs/data_acquisition/Shoplifting*_candidate.jpg`.

The next data requirement remains original person-linked event boundaries plus
RGB, documented annotation quality and usage terms. The downloaded candidate can
support that investigation, but it does not close this requirement or justify a
new accuracy claim. No existing checkpoint, threshold or benchmark split changed.
