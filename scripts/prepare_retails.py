"""Download the official RetailS archive, safely extract it and verify its layout.

Install requirements-data.txt only when a download is needed. Existing files
with matching ZIP-member sizes are reused. No dataset content belongs in Git.
"""

import hashlib
import json
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    target = (ROOT / "data/retails").resolve()
    target.mkdir(parents=True, exist_ok=True)
    archive = target / "RetailS.zip"
    if not archive.is_file():
        try:
            import gdown
        except ImportError as error:
            raise SystemExit("Install requirements-data.txt to download RetailS, or place RetailS.zip in data/retails/.") from error
        gdown.download(id="1uBCvDm7QdYjxwS9HT63lYaucATrecNuH", output=str(archive),
                       use_cookies=False, resume=True)
    if not zipfile.is_zipfile(archive):
        raise ValueError("RetailS.zip is not a valid ZIP archive; check the provider download/access")
    with zipfile.ZipFile(archive) as bundle:
        pending = []
        for entry in bundle.infolist():
            path = (target / entry.filename).resolve()
            if not path.is_relative_to(target):
                raise ValueError("Archive member would escape the RetailS directory")
            if not entry.is_dir() and (not path.is_file() or path.stat().st_size != entry.file_size):
                pending.append(entry)
        if shutil.disk_usage(target).free < sum(e.file_size for e in pending):
            raise OSError("Not enough free space to extract RetailS")
        for entry in pending:
            bundle.extract(entry, target)
        member_count = len(bundle.infolist())
    with archive.open("rb") as stream:
        checksum = hashlib.file_digest(stream, "sha256").hexdigest()
    evidence = {"archive": str(archive), "archive_bytes": archive.stat().st_size,
                "archive_sha256": checksum, "members": member_count,
                "members_extracted_this_run": len(pending),
                "official_source": "https://github.com/TeCSAR-UNCC/RetailS#dataset-download-instructions",
                "download_url": "https://drive.google.com/file/d/1uBCvDm7QdYjxwS9HT63lYaucATrecNuH/view"}
    log = ROOT / "outputs/logs/retails_download.json"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    print(f"RetailS archive ready: {archive}; {len(pending)} members extracted this run.")
    from scripts.verify_retails import main as verify
    verify()


if __name__ == "__main__":
    main()
