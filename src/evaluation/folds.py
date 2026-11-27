"""Two labelled dev folds over the RetailS *test* videos (research_v2, step 0).

RetailS train has no anomalies, so the normal-only validation split cannot rank
scoring variants by AUC. Instead the test videos are split into two folds grouped
by (subset, camera, video): every choice is made on one fold and reported on the
other, then swapped (cross-fitting). No video is ever in both folds, so the score
reported for a video never depends on its own labels.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

FOLDS = ('A', 'B')


def video_key(subset: str, camera: int, stem: str) -> str:
    return f'{subset}|{int(camera)}|{stem}'


def make_folds(videos: list[dict], seed: int = 0) -> dict:
    """videos: dicts with subset, camera (int), stem, normal (frames), anomaly (frames).

    Within each (subset, camera) group the videos are shuffled with ``seed`` and
    assigned greedily so both folds get similar normal and anomalous frame totals.
    Afterwards, any group that has both classes but leaves a fold without one of
    them is repaired by moving the smallest video that supplies the missing class
    (only when the donor fold keeps that class too).
    """
    rng = np.random.default_rng(seed)
    assignment = {}
    groups = {}
    for v in videos:
        groups.setdefault((v['subset'], int(v['camera'])), []).append(v)
    for key in sorted(groups):
        members = groups[key]
        order = rng.permutation(len(members))
        members = [members[i] for i in order]
        # Largest anomalous (then normal) videos first so greedy balancing works.
        members.sort(key=lambda v: (-(v['anomaly'] > 0), -v['anomaly'], -v['normal']))
        totals = {f: np.zeros(3) for f in FOLDS}   # normal frames, anomaly frames, videos
        for v in members:
            cls = 1 if v['anomaly'] > 0 else 0
            fold = min(FOLDS, key=lambda f: (totals[f][cls], totals[f][2], f))
            assignment[video_key(v['subset'], v['camera'], v['stem'])] = fold
            totals[fold] += (v['normal'], v['anomaly'], 1)
        _repair(members, assignment)
    return assignment


def _class_totals(members, assignment, fold):
    keys = [m for m in members if assignment[video_key(m['subset'], m['camera'], m['stem'])] == fold]
    return sum(m['normal'] for m in keys), sum(m['anomaly'] for m in keys)


def _repair(members, assignment):
    if len(members) < 2:
        return
    has_normal = sum(m['normal'] for m in members) > 0
    has_anomaly = sum(m['anomaly'] for m in members) > 0
    for _ in range(len(members)):
        changed = False
        for fold in FOLDS:
            other = 'B' if fold == 'A' else 'A'
            n, a = _class_totals(members, assignment, fold)
            for cls, missing in ((0, has_normal and n == 0), (1, has_anomaly and a == 0)):
                if not missing:
                    continue
                field = 'normal' if cls == 0 else 'anomaly'
                donors = sorted([m for m in members if assignment[video_key(m['subset'], m['camera'], m['stem'])] == other
                                 and m[field] > 0], key=lambda m: m['normal'] + m['anomaly'])
                for d in donors:
                    key = video_key(d['subset'], d['camera'], d['stem'])
                    assignment[key] = fold
                    on, oa = _class_totals(members, assignment, other)
                    if (on > 0 or not has_normal) and (oa > 0 or not has_anomaly) and \
                            sum(1 for m in members if assignment[video_key(m['subset'], m['camera'], m['stem'])] == other):
                        changed = True
                        break
                    assignment[key] = other
        if not changed:
            break


def fold_report(videos: list[dict], assignment: dict) -> dict:
    report = {}
    for fold in FOLDS:
        rows = [v for v in videos if assignment[video_key(v['subset'], v['camera'], v['stem'])] == fold]
        report[fold] = dict(videos=len(rows), normal_frames=int(sum(v['normal'] for v in rows)),
                            anomaly_frames=int(sum(v['anomaly'] for v in rows)),
                            per_group={f'{s}|cam{c}': dict(
                                videos=sum(1 for v in rows if (v['subset'], v['camera']) == (s, c)),
                                normal=int(sum(v['normal'] for v in rows if (v['subset'], v['camera']) == (s, c))),
                                anomaly=int(sum(v['anomaly'] for v in rows if (v['subset'], v['camera']) == (s, c))))
                                for s, c in sorted({(v['subset'], v['camera']) for v in videos})})
    return report


def check_folds(videos: list[dict], assignment: dict) -> list[str]:
    """Return a list of violated guarantees (empty = OK)."""
    problems = []
    keys = [video_key(v['subset'], v['camera'], v['stem']) for v in videos]
    if len(set(keys)) != len(keys):
        problems.append('duplicate video keys')
    if set(assignment) != set(keys):
        problems.append('assignment does not cover exactly the given videos')
    groups = {}
    for v in videos:
        groups.setdefault((v['subset'], v['camera']), []).append(v)
    for g, members in groups.items():
        if len(members) < 2:
            continue
        both = sum(m['normal'] for m in members) > 0 and sum(m['anomaly'] for m in members) > 0
        for fold in FOLDS:
            n, a = _class_totals(members, assignment, fold)
            if n + a == 0:
                problems.append(f'{g}: fold {fold} empty')
            elif both and (n == 0 or a == 0):
                # Only a violation if *some* assignment could give the fold both classes.
                problems.append(f'{g}: fold {fold} lacks a class (normal={n}, anomaly={a})')
    return problems


def save_folds(path: Path, videos, assignment, seed):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(seed=seed, assignment=assignment, report=fold_report(videos, assignment),
                                    problems=check_folds(videos, assignment)), indent=2), encoding='utf-8')
