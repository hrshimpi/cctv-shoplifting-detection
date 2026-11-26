"""Score one video with the trained MNNIT v2 clip classifier: shoplifting yes/no.

Any resolution or aspect ratio (4:3, 1:1, 16:9, 9:16, ...) is accepted: frames are
letterboxed exactly as in training. Long videos are split into ~10 s chunks (training
clips average ~11 s); each chunk is scored and the video is flagged if any chunk is.

    python scripts/predict_mnnit_clip.py --video my_clip.mp4 [--chunk-seconds 10]

Scores are model outputs, not calibrated theft probabilities. The model was trained on
staged laboratory clips; results on real store footage are unverified.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np

from src.data import video_clip_features as vcf

MODEL_DIR = ROOT / 'outputs/mnnit_clip_v2'


def load_head(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def predict(video, chunk_seconds=10.0, model_dir=MODEL_DIR, nets=None):
    head = load_head(model_dir / 'model.npz')
    feature_set = str(head['feature_set']); threshold = float(head['threshold'])
    meta = json.loads((model_dir / 'evaluation.json').read_text())
    nets = nets or vcf.Backbones(pretrained=meta.get('pretrained', True))
    total, fps = vcf.count_frames(video)
    import cv2
    cap = cv2.VideoCapture(str(video)); w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)); cap.release()
    segments = vcf.segments_for(total, fps, chunk_seconds if total / fps > 1.5 * chunk_seconds else None)
    chunks = []
    for (s, e), seg in zip(segments, vcf.read_letterboxed(video, segments)):
        f, wv = nets.embed(seg)
        x = vcf.descriptors(f, wv)[feature_set][None]
        score = float(vcf.head_probability(x, head)[0])
        chunks.append(dict(start_s=round(s / fps, 2), end_s=round(e / fps, 2), score=round(score, 4), shoplifting=score >= threshold))
    best = max(chunks, key=lambda c: c['score'])
    return dict(video=str(video), width=w, height=h, aspect=f'{w}:{h}', fps=round(fps, 2), frames=total,
                duration_s=round(total / fps, 2), threshold=round(threshold, 4), max_score=best['score'],
                shoplifting=any(c['shoplifting'] for c in chunks), chunks=chunks)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--video', required=True)
    ap.add_argument('--chunk-seconds', type=float, default=10.0)
    ap.add_argument('--model-dir', default=str(MODEL_DIR))
    args = ap.parse_args()
    r = predict(Path(args.video), args.chunk_seconds, Path(args.model_dir))
    print(json.dumps(r, indent=2))
    verdict = 'SHOPLIFTING SUSPECTED' if r['shoplifting'] else 'NO SHOPLIFTING DETECTED'
    print(f"\n{verdict}  (max score {r['max_score']:.3f}, threshold {r['threshold']:.3f}, {r['width']}x{r['height']}, {r['duration_s']} s)")


if __name__ == '__main__':
    main()
