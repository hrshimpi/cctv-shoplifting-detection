"""Verify and inventory MNNIT V1 clips without modifying pose data or labels."""
import csv
import hashlib
import json
import shutil
import stat
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_SHA256 = '57657afeff84c44482327c46fdbb1f45b22e9f045bf44cdd31e368b07bebc93c'
BASE = ROOT / 'data/action_sources/mnnit_v1'
OUT = ROOT / 'outputs/mnnit_clip_classification'


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def safe_member(root, name):
    normalized = name.replace('\\', '/')
    if ':' in normalized or normalized.startswith('/'):
        raise ValueError('Unsafe archive path')
    target = (root / normalized).resolve()
    if not target.is_relative_to(root.resolve()):
        raise ValueError('Archive path escapes extraction root')
    return target


def sample_indices(count, samples=16):
    if count < 2:
        raise ValueError('Video must have at least two frames')
    return np.linspace(0, count - 1, samples, dtype=np.int64)


def dhash(frame):
    gray = cv2.cvtColor(cv2.resize(frame, (9, 8)), cv2.COLOR_BGR2GRAY)
    bits = (gray[:, 1:] > gray[:, :-1]).ravel()
    return sum(int(bit) << i for i, bit in enumerate(bits))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    archive = BASE / 'Dataset.zip'
    if sha256(archive) != ARCHIVE_SHA256:
        raise ValueError('Archive differs from publisher SHA256')
    extraction = BASE / 'extracted'
    with zipfile.ZipFile(archive) as z:
        members = z.infolist()
        for item in members:
            safe_member(extraction, item.filename)
            if stat.S_ISLNK(item.external_attr >> 16):
                raise ValueError('Archive contains a symbolic link')
        for item in members:
            path = safe_member(extraction, item.filename)
            if item.is_dir():
                path.mkdir(parents=True, exist_ok=True)
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                # Reading each member also validates its ZIP CRC.
                with z.open(item) as source, path.open('wb') as target:
                    shutil.copyfileobj(source, target)
    paths = sorted(extraction.rglob('*.mp4'))
    if not paths:
        raise ValueError('Archive has no MP4 clips')
    rows, samples, thumbs, hashes, tiny, failures = [], [], [], [], [], []
    for index, path in enumerate(paths):
        folder = path.parent.name.lower()
        if folder not in ['normal', 'shoplifting']:
            raise ValueError(f'Unknown label folder: {folder}')
        cap = cv2.VideoCapture(str(path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        expected = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not cap.isOpened() or not np.isfinite(fps) or fps <= 0 or expected < 2:
            failures.append(str(path)); cap.release(); continue
        ids = sample_indices(expected)
        selected, fid, size = {}, 0, None
        wanted = set(ids.tolist())
        while True:
            ok, frame = cap.read()
            if not ok: break
            if size is None: size = frame.shape[:2]
            if frame.shape[:2] != size: raise ValueError('Frame shape changes within a clip')
            if fid in wanted: selected[fid] = frame
            fid += 1
        cap.release()
        if fid != expected or len(selected) != len(wanted):
            failures.append(str(path)); continue
        images = [selected[int(i)] for i in ids]
        # Retain full scenes in 320x240 inspection/cache frames. Encoder has its
        # own explicit preprocessing; never infer actor IDs or exact event times.
        sampled = np.stack([cv2.cvtColor(cv2.resize(f, (320, 240)), cv2.COLOR_BGR2RGB) for f in images])
        small = np.stack([cv2.resize(f, (32, 24)) for f in images])
        sample_digest = hashlib.sha256(sampled.tobytes()).hexdigest()
        rows.append(dict(clip_id=f'MN{index+1:03d}', path=str(path.relative_to(ROOT)),
            label=int(folder == 'shoplifting'), class_name=folder,
            source_filename=path.name, source_sha256=sha256(path), sampled_sha256=sample_digest,
            frames=fid, fps=fps, width=size[1], height=size[0], duration_seconds=fid/fps,
            actor_id=None, recording_session=None, action_start_frame=None, action_end_frame=None,
            label_source='Publisher class folder; staged laboratory clip label'))
        samples.append(sampled); thumbs.append(images[len(images)//2]);
        hashes.append([dhash(f) for f in images]); tiny.append(small)
        if len(rows) % 20 == 0: print('Decoded and inventoried', len(rows), '/', len(paths), flush=True)
    if failures:
        (OUT / 'decode_failures.json').write_text(json.dumps(failures, indent=2))
        raise ValueError('Video decode/count failures; no split was created')
    exact = defaultdict(list)
    for i, row in enumerate(rows): exact[(row['source_sha256'], row['sampled_sha256'])].append(i)
    # Union exact binary and exact sampled-pixel matches, including remuxes.
    parent = list(range(len(rows)))
    def find(i):
        while parent[i] != i: parent[i] = parent[parent[i]]; i = parent[i]
        return i
    def union(a, b):
        a, b = find(a), find(b)
        if a != b: parent[max(a,b)] = min(a,b)
    pairs = []
    for i in range(len(rows)):
        for j in range(i):
            if rows[i]['source_sha256'] == rows[j]['source_sha256'] or rows[i]['sampled_sha256'] == rows[j]['sampled_sha256']:
                union(i,j); pairs.append(dict(a=rows[j]['clip_id'], b=rows[i]['clip_id'], kind='exact'))
    exact_groups = defaultdict(list)
    for i in range(len(rows)): exact_groups[find(i)].append(i)
    conflict = {i for group in exact_groups.values() if len({rows[k]['label'] for k in group}) > 1 for i in group}
    canonical = [min(group) for group in exact_groups.values() if not any(i in conflict for i in group)]
    # Conservative sampled near-duplicate candidates are kept in the same split.
    # This cannot establish participant/session independence.
    for pos, i in enumerate(canonical):
        for j in canonical[:pos]:
            ratio = rows[i]['duration_seconds'] / rows[j]['duration_seconds']
            if not .9 <= ratio <= 1.1: continue
            distances = [(a ^ b).bit_count() for a,b in zip(hashes[i], hashes[j])]
            mad = float(np.abs(tiny[i].astype(float) - tiny[j].astype(float)).mean())
            if np.mean(np.array(distances) <= 4) >= .8 and mad <= 6:
                union(i,j); pairs.append(dict(a=rows[j]['clip_id'], b=rows[i]['clip_id'], kind='sampled_near_duplicate', mean_abs_pixel_difference=mad))
    for i,row in enumerate(rows):
        row['duplicate_group'] = 'DG'+str(find(i)).zfill(3)
        row['included'] = i in canonical
        row['exclusion'] = 'conflicting_exact_duplicate_labels' if i in conflict else ('' if i in canonical else 'exact_duplicate')
    def save_csv(path, records):
        with path.open('w', newline='', encoding='utf-8') as stream:
            writer=csv.DictWriter(stream, fieldnames=list(records[0])); writer.writeheader();writer.writerows(records)
    save_csv(OUT / 'inventory.csv', rows)
    np.savez_compressed(OUT / 'sampled_rgb.npz', frames=np.stack(samples), frame_indices=np.array([sample_indices(r['frames']) for r in rows]), clip_ids=np.array([r['clip_id'] for r in rows]))
    (OUT/'duplicate_pairs.json').write_text(json.dumps(pairs,indent=2))
    # Contact sheets show every clip once. They support scene/quality checks,
    # not independent action-label adjudication.
    for offset in range(0,len(rows),20):
        sheet=np.full((5*190,4*250,3),245,dtype=np.uint8)
        for local,(row,frame) in enumerate(zip(rows[offset:offset+20],thumbs[offset:offset+20])):
            x=(local%4)*250;y=(local//4)*190
            sheet[y+30:y+185,x+3:x+247]=cv2.resize(frame,(244,155))
            cv2.putText(sheet, row['clip_id']+' '+row['source_filename'], (x+4,y+20),cv2.FONT_HERSHEY_SIMPLEX,.36,(15,15,15),1,cv2.LINE_AA)
        cv2.imwrite(str(OUT/f'contact_{offset//20+1:02d}.jpg'),sheet)
    report=dict(source='https://data.mendeley.com/datasets/r3yjf35hzr/1', doi='10.17632/r3yjf35hzr.1',
        authors=['Mohd. Aquib Ansari','Dushyant Kumar Singh'], license='CC BY 4.0', archive_sha256=ARCHIVE_SHA256,
        archive_bytes=archive.stat().st_size, videos=len(rows), class_counts=dict(Counter(r['class_name'] for r in rows)),
        canonical_videos=len(canonical), exact_duplicates_excluded=len(rows)-len(canonical)-len(conflict), conflicting_exact_label_files=len(conflict),
        near_duplicate_pairs=sum(p['kind']=='sampled_near_duplicate' for p in pairs), decoded_frames=sum(r['frames'] for r in rows),
        fps_values=sorted({round(r['fps'],4) for r in rows}), sizes=sorted({f"{r['width']}x{r['height']}" for r in rows}),
        decoded_all_frames=True, independent_actor_session_ids_available=False, inventory_sha256=sha256(OUT/'inventory.csv'),
        rgb_cache_sha256=sha256(OUT/'sampled_rgb.npz'), labels_are_exact_action_boundaries=False,
        scope='Staged laboratory clip classification, not real-store person/time action detection')
    (OUT/'data_report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)


if __name__ == '__main__': main()
