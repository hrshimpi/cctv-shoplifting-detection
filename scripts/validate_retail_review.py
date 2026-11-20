"""Validate an exported human review; do not invent labels or train a model."""
import argparse
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.evaluation.manual_review import validate_review
from src.experiments.quality_dataset import sha256


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('review',type=Path)
    args=parser.parse_args()
    folder=ROOT / 'outputs/manual_validation'
    manifest=json.loads((folder/'review_manifest.json').read_text())
    private={r['case_id']:r for r in json.loads((folder/'_private_sources.json').read_text())}
    for case in manifest['cases']:
        source=ROOT/private[case['case_id']]['source_path']
        poses=ROOT/private[case['case_id']]['pose_dir']/'poses.json'
        if sha256(source)!=case['source_sha256'] or sha256(poses)!=case['pose_sha256']:
            raise ValueError('Original video or pose source changed')
        for key,expected in [('source_file','source_media_sha256'),('overlay_file','overlay_sha256')]:
            if sha256(folder/case[key])!=case[expected]:
                raise ValueError('Review media changed')
    payload=json.loads(args.review.read_text(encoding='utf-8-sig'))
    result=validate_review(payload,manifest)
    result.update(review_sha256=sha256(args.review),manifest_id=manifest['manifest_id'])
    destination=folder / 'annotations' / result['review_sha256'][:16]
    destination.mkdir(parents=True,exist_ok=True)
    (destination/'review.json').write_text(json.dumps(payload,indent=2))
    (destination/'validation.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2)); print('Saved validated observations:',destination)


if __name__=='__main__':
    main()
