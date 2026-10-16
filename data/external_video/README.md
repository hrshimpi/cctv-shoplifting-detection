# Shoplifting Video Project Package

This package is prepared for the CSE 6367 shoplifting + pose-lifting project.

## Dataset plan

Primary research dataset:
- RetailS (pose-based retail shoplifting dataset)
- Official repo: https://github.com/TeCSAR-UNCC/RetailS

Raw-video external validation:
- UCF-Crime
- Official UCF page: https://www.crcv.ucf.edu/research/real-world-anomaly-detection-in-surveillance-videos/
- Official full dataset download: https://www.crcv.ucf.edu/data1/chenchen/UCF_Crimes.zip
- Kaggle mirror: https://www.kaggle.com/datasets/minhajuddinmeraj/anomalydetectiondatasetucf

## Where to put videos

Place shoplifting videos in:

videos/shoplifting/

Place normal videos in:

videos/normal/

Example:

videos/
  shoplifting/
    Shoplifting001_x264.mp4
    Shoplifting005_x264.mp4
  normal/
    Normal_Videos_001_x264.mp4
    Normal_Videos_002_x264.mp4

## Extract at least 100 frames

Run:

python scripts/extract_frames.py

By default the script extracts:
- 100 shoplifting frames
- 100 normal frames

The extracted frames will be written to:

frames/shoplifting/
frames/normal/

## Recommended experiment

RetailS 2D pose
 -> MotionBERT
 -> 3D pose
 -> temporal anomaly detector

External validation:

UCF-Crime raw video
 -> 2D pose detector
 -> MotionBERT
 -> 3D pose
 -> same temporal anomaly detector

Important:
UCF-Crime and Kaggle-hosted copies are third-party research datasets. Download them from the official/provider source and follow their usage/license requirements. This ZIP intentionally does not redistribute those videos.