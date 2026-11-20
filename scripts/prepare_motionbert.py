"""Download the official author's H36M-finetuned pose checkpoint with hash verification."""
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parents[1]
REPO = 'walterzhu/MotionBERT'
REMOTE = 'checkpoint/pose3d/FT_MB_release_MB_ft_h36m/best_epoch.bin'


def main():
    source = ROOT / 'third_party/MotionBERT'
    if not (source / 'lib/model/DSTformer.py').is_file():
        raise FileNotFoundError('Clone https://github.com/Walter0807/MotionBERT into third_party/MotionBERT first')
    session = requests.Session()
    response = session.get(f'https://huggingface.co/api/models/{REPO}', timeout=30)
    response.raise_for_status()
    revision = response.json()['sha']
    response = session.get(f'https://huggingface.co/api/models/{REPO}/tree/{revision}/checkpoint/pose3d/FT_MB_release_MB_ft_h36m', timeout=30)
    response.raise_for_status()
    entry = next(x for x in response.json() if x['path'] == REMOTE)
    expected_hash, expected_size = entry['lfs']['oid'], entry['size']
    destination = source / 'checkpoint/pose3d/MB_ft_h36m/best_epoch.bin'
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        if shutil.disk_usage(destination.parent).free < expected_size * 2:
            raise OSError('Insufficient disk space for checkpoint download')
        url = f'https://huggingface.co/{REPO}/resolve/{revision}/{REMOTE}'
        response = session.get(url, stream=True, timeout=(30, 120))
        response.raise_for_status()
        partial = destination.with_suffix('.partial')
        with partial.open('wb') as f:
            for chunk in response.iter_content(1024 * 1024):
                f.write(chunk)
        with partial.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        if partial.stat().st_size != expected_size or digest != expected_hash:
            raise ValueError('Downloaded checkpoint size/hash differs from author repository')
        partial.replace(destination)
    with destination.open('rb') as f:
        digest = hashlib.file_digest(f, 'sha256').hexdigest()
    if digest != expected_hash or destination.stat().st_size != expected_size:
        raise ValueError('Existing checkpoint does not match official file')
    report = dict(repository=REPO, revision=revision, remote_path=REMOTE,
        local_path=str(destination), bytes=expected_size, sha256=digest,
        source_commit=subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip())
    out = ROOT / 'outputs/logs/motionbert_setup.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
