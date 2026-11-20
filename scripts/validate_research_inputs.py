"""Validate quality-corrected input geometry and the real frozen MotionBERT lift."""
import json
import os
import sys
import time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'outputs/matplotlib'))
sys.path.insert(0, str(ROOT))
import matplotlib.pyplot as plt
import numpy as np
import yaml
from src.data.retails import discover_retails_root
from src.experiments.quality_dataset import collect, sha256
from src.motionbert.lift import crop_scale, load_motionbert, lift_2d_to_3d_batch
from src.utils.joint_mapping import coco17_to_h36m17, DIRECT_H36M_TO_COCO, H36M17_BONES
from scripts.validate_joint_mapping import (plot_2d_skeleton, plot_3d_skeleton,
    COCO17_BONES, COCO17_LABELS, H36M17_LABELS)


def plot_supported(ax, points, bones, labels, title):
    visible = points[:, 2] > 0
    xy = points[:, :2]
    ax.scatter(xy[visible, 0], -xy[visible, 1], s=25)
    for a, b in bones:
        if visible[a] and visible[b]:
            ax.plot(xy[[a, b], 0], -xy[[a, b], 1], color='tab:blue')
    for joint, label in labels.items():
        if visible[joint]:
            ax.annotate(label, (xy[joint, 0], -xy[joint, 1]), fontsize=7)
    ax.set_aspect('equal'); ax.set_title(title, fontsize=10)


def main():
    cfg = yaml.safe_load((ROOT / 'configs/research_v1.yaml').read_text())
    root = discover_retails_root(ROOT / 'data/retails')
    arrays, manifest = collect(root, 'train', cfg, cfg['matched_train_files'])
    checkpoint = ROOT / 'third_party/MotionBERT/checkpoint/pose3d/MB_ft_h36m/best_epoch.bin'
    model = load_motionbert(checkpoint)
    assert not model.training and all(not p.requires_grad for p in model.parameters())
    indices = [next(i for i, m in enumerate(manifest['metadata']) if m['camera_id'] == camera)
               for camera in sorted({m['camera_id'] for m in manifest['metadata']})]
    windows = arrays['raw'][indices]
    mapped = coco17_to_h36m17(windows)
    for target, source in DIRECT_H36M_TO_COCO.items():
        np.testing.assert_array_equal(mapped[..., target, :], windows[..., source, :])
    from lib.utils.utils_data import crop_scale as official_crop_scale
    for w in mapped:
        np.testing.assert_allclose(crop_scale(w), official_crop_scale(w, [1, 1]), atol=1e-6)
    started = time.perf_counter()
    lifted = lift_2d_to_3d_batch(model, windows)
    elapsed = time.perf_counter() - started
    assert lifted.shape == windows.shape and np.isfinite(lifted).all()
    assert (lifted[..., 0, :] == 0).all()
    out = ROOT / 'outputs/plots/research_input_validation'
    out.mkdir(parents=True, exist_ok=True)
    examples = []
    for i, index in enumerate(indices):
        meta = manifest['metadata'][index]
        frame = int(arrays['visibility'][index].mean(1).argmax())
        fig = plt.figure(figsize=(15, 5))
        plot_supported(fig.add_subplot(131), windows[i, frame], COCO17_BONES, COCO17_LABELS, f"Camera {meta['camera_id']}: explicit RetailS yx to xy")
        plot_supported(fig.add_subplot(132), mapped[i, frame], H36M17_BONES, H36M17_LABELS, 'H36M joint mapping; unsupported joints omitted')
        plot_3d_skeleton(fig.add_subplot(133, projection='3d'), lifted[i, frame], H36M17_BONES, H36M17_LABELS, 'Frozen MotionBERT: equal axis scale')
        fig.tight_layout()
        path = out / f"camera_{meta['camera_id']}.png"
        fig.savefig(path, dpi=150); plt.close(fig)
        examples.append(dict(**meta, plot=str(path.relative_to(ROOT))))
    report = dict(checkpoint_sha256=sha256(checkpoint), cameras=len(indices),
        official_normalization_parity=True, direct_joint_mapping_exact=True,
        frozen_model=True, output_finite=True, root_relative=True, shape=list(lifted.shape),
        measured_seconds_per_window=elapsed / len(indices), data_fingerprint=manifest['fingerprint'],
        coordinate_order='Explicit dataset-specific yx-to-xy correction, based on joint geometry across six cameras and author sample renderings; source RGB/calibration is unavailable.',
        limitation='Visual/structural validation only: no RetailS 3D ground truth is available to measure lifting accuracy.',
        inference_protocol='30-frame windows, single-view forward pass without flip-test averaging; root-relative output',
        visual_review_status='Mixed: structure checks pass, but some limb predictions remain implausible.',
        visual_review_notes=[
            'Camera 3 example has a missing input ankle; it must be omitted from the 2D plot rather than connected to the origin.',
            'Camera 5 example has an implausibly elevated lifted leg despite a standing 2D pose. Do not claim reliable 3D reconstruction.',
            'Cameras 1, 2, 4 and 6 have recognizable body structure; this is visual inspection, not accuracy validation.'],
        examples=examples)
    target = ROOT / 'outputs/logs/research_input_validation.json'
    target.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
