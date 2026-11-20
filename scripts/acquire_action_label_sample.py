"""Download a bounded, versioned person-window sample; do not invent action truth."""
import hashlib
import io
import json
import sys
import urllib.request
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
import pandas as pd

REPO = 'bassatbassat/shoplifting-vlm-finetune-ucf'
REVISION = '4d378a4f2507ae90204b6d5eefeb5c38996c4951'
SOURCE = 'https://huggingface.co/datasets/' + REPO
MAX_TOTAL_BYTES = 160 * 1024 ** 2
EXCLUDE = {'Shoplifting017_x264', 'Shoplifting031_x264', 'Shoplifting033_x264',
           'Shoplifting045_x264', 'Shoplifting054_x264'}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch(url, limit=12 * 1024 ** 2):
    with urllib.request.urlopen(url, timeout=60) as response:
        data = response.read(limit + 1)
    if len(data) > limit:
        raise ValueError('Download exceeded bounded size')
    return data


def validate_clip(path, start, end):
    with np.load(path, allow_pickle=False) as data:
        image_keys = [k for k in data.files if data[k].ndim == 4 and data[k].shape[-1] == 3]
        if len(image_keys) != 1 or 'frame_indices' not in data:
            raise ValueError('Expected RGB frames and original frame indices')
        frames, ids = data[image_keys[0]], data['frame_indices']
        if frames.dtype != np.uint8 or len(frames) != 16 or ids.shape != (16,):
            raise ValueError('Unexpected sampled clip shape/type')
        if not np.isfinite(ids).all() or not np.equal(ids, ids.astype(np.int64)).all() or np.any(np.diff(ids) < 0):
            raise ValueError('Invalid original frame indices')
        if ids.min() < start or ids.max() > end:
            raise ValueError('Sampled frames fall outside exported window')
        return frames.copy(), ids.copy(), image_keys[0]


def main():
    folder = ROOT / 'data/action_sources/ucf_person_windows'
    meta = folder / 'upstream'; meta.mkdir(parents=True, exist_ok=True)
    out = ROOT / 'outputs/data_acquisition'; out.mkdir(parents=True, exist_ok=True)
    metadata_names = ['README.md', 'pack_meta.json', 'labels/manifest_meta.json',
                      'labels/window_manifest.csv', 'labels/source_videos.csv', 'labels/fold_1_splits.csv']
    hashes = {}
    for name in metadata_names:
        target = meta / name; target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(fetch(SOURCE + '/resolve/' + REVISION + '/' + name))
        hashes[name] = digest(target)
    info = json.loads(fetch('https://huggingface.co/api/datasets/' + REPO + '/revision/' + REVISION))
    if info['sha'] != REVISION or info.get('gated'):
        raise ValueError('Revision differs or dataset is gated')
    (meta / 'dataset_info.json').write_text(json.dumps(info, indent=2), encoding='utf-8')
    names = {s['rfilename'] for s in info['siblings']}
    windows = pd.read_csv(meta / 'labels/window_manifest.csv', dtype={'person_id': str})
    splits = pd.read_csv(meta / 'labels/fold_1_splits.csv')
    windows = windows.merge(splits, on='window_id', validate='one_to_one')
    windows['remote_path'] = windows.apply(lambda r:
        f'rgb_clips/box_frame/n16/{r.video_base}/{r.video_base}__{r.person_id}__{r.start_frame}.npz', axis=1)
    windows = windows[windows.remote_path.isin(names) & ~windows.video_base.isin(EXCLUDE)]
    counts = windows.groupby(['outer_split', 'video_base', 'label']).size().unstack(fill_value=0)
    candidates = counts[(counts[0] >= 2) & (counts[1] >= 2)].reset_index()
    selected = []
    # Preserve upstream fold-1 test membership. These are inspection roles only,
    # never untouched confirmation data after RGB inspection.
    for outer in ['train_dev', 'test']:
        videos = sorted(candidates[candidates.outer_split == outer].video_base)[:3]
        if len(videos) != 3:
            raise ValueError('Not enough source-disjoint candidate videos')
        for video in videos:
            for label in [0, 1]:
                choices = windows[(windows.video_base == video) & (windows.label == label)].sort_values(['start_frame', 'person_id'])
                selected.extend(choices.iloc[np.linspace(0, len(choices) - 1, 2, dtype=int)].to_dict('records'))
    records, total = [], 0
    for video in sorted({r['video_base'] for r in selected}):
        directory = f'rgb_clips/box_frame/n16/{video}'
        listing = json.loads(fetch(f'https://huggingface.co/api/datasets/{REPO}/tree/{REVISION}/{directory}?limit=1000'))
        files = {r['path']: r for r in listing}
        previews = []
        for row in [r for r in selected if r['video_base'] == video]:
            name = row['remote_path']; entry = files[name]
            total += entry['size']
            if total > MAX_TOTAL_BYTES:
                raise ValueError('Sample exceeds total download budget')
            expected = entry.get('lfs', {}).get('oid')
            if not expected:
                raise ValueError('Expected upstream SHA256 for binary file')
            target = folder / name; target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists() or digest(target) != expected:
                content = fetch(SOURCE + '/resolve/' + REVISION + '/' + name, min(MAX_TOTAL_BYTES, 16 * 1024 ** 2))
                if hashlib.sha256(content).hexdigest() != expected:
                    raise ValueError('Upstream RGB hash mismatch')
                target.write_bytes(content)
            if target.stat().st_size != entry['size']:
                raise ValueError('Upstream RGB byte count differs')
            images, ids, key = validate_clip(target, int(row['start_frame']), int(row['rgb_end_frame']))
            record = {k: row[k] for k in ['window_id', 'video_base', 'person_id', 'start_frame', 'end_frame', 'rgb_end_frame', 'label', 'outer_split']}
            record.update(local_rgb=str(target.relative_to(ROOT)), sha256=expected, bytes=entry['size'],
                rgb_key=key, rgb_shape=list(images.shape), original_frame_indices=ids.tolist(),
                action_start_frame=None, action_end_frame=None,
                label_provenance='Publisher states manually labeled event-associated person windows.',
                independently_verified=False, actor_identity_verified=False,
                training_eligible=False, evaluation_role='inspection_only')
            records.append(record)
            for sample in [0, len(images) // 2, len(images) - 1]:
                frame = cv2.cvtColor(images[sample], cv2.COLOR_RGB2BGR)
                frame = cv2.resize(frame, (384, 288))
                tile = np.full((328, 384, 3), 245, dtype=np.uint8); tile[40:] = frame
                title = f"Publisher label {row['label']} | person {row['person_id']} | f{int(ids[sample])}"
                cv2.putText(tile, title, (8, 25), cv2.FONT_HERSHEY_SIMPLEX, .46, (25, 25, 25), 1, cv2.LINE_AA)
                previews.append(tile)
            print(video, row['person_id'], row['start_frame'], 'label', row['label'], 'verified RGB/hash', flush=True)
        contact = np.vstack([np.hstack(previews[i:i+3]) for i in range(0, len(previews), 3)])
        cv2.imwrite(str(out / (video + '_candidate.jpg')), contact)
    pd.DataFrame(records).to_csv(folder / 'sample_manifest.csv', index=False)
    report = dict(source=SOURCE, revision=REVISION, downloaded_windows=len(records),
        source_videos=len({r['video_base'] for r in records}),
        publisher_positive_windows=sum(r['label'] == 1 for r in records),
        publisher_normal_windows=sum(r['label'] == 0 for r in records), bytes=total,
        rgb_frames_verified=sum(r['rgb_shape'][0] for r in records), upstream_metadata_sha256=hashes,
        rows=records, overlap_with_previous_local_pilot=sorted({r['video_base'] for r in records} & EXCLUDE),
        exact_action_boundaries_available=False, independently_verified_action_labels=False,
        training_started=False, training_eligible=False, license_status='No explicit license file or card license found at this revision; clarification needed before training or redistribution.',
        qualification='Candidate person-window dataset, not verified exact-boundary action data.',
        limitations=['Publisher person IDs are not linked to existing local tracker IDs.',
            'Window boundaries are not action start/end annotations; original FPS is not specified here.',
            'Source MP4 paths refer to the publisher machine; these are sampled RGB arrays, not full original videos.',
            'Sampled clips and labels have not been independently adjudicated. Inspection samples cannot be treated as untouched test data.',
            'RetailAction labels take/put/touch interactions, not shoplifting. MNNIT labels whole clips, not actor/time intervals.'])
    (out / 'action_data_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    (folder / 'provenance.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ['rows', 'upstream_metadata_sha256']}, indent=2))


if __name__ == '__main__':
    main()
