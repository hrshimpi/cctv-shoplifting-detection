"""
Download the CCTV shoplifting detection dataset from Kaggle via kagglehub.

Requires a Kaggle API token, either at ~/.kaggle/kaggle.json or supplied via
the KAGGLE_USERNAME / KAGGLE_KEY environment variables.

After downloading, this script prints the full folder tree of the dataset
and the contents of any data.yaml, classes.txt, or caption/annotation JSON
files it finds. This is intentional: the dataset's exact folder layout and
class list are not hardcoded anywhere in this project. Every downstream
script (EDA, training, etc.) should discover the structure by reading
whatever config files actually exist in the download, rather than assuming
a fixed layout.
"""

import json
import shutil
import sys
from pathlib import Path

import kagglehub

# The tree/box-drawing characters below aren't representable in the cp1252
# fallback Python uses on Windows when stdout isn't a real console (piped
# output, Git Bash, redirected to a file) — force UTF-8 so this doesn't
# crash mid-print in those cases.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

DATASET_SLUG = "simuletic/cctv-shoplifting-detection-dataset-yolo-and-vlm"

# Project-local copy of the dataset. kagglehub caches downloads outside the
# repo (e.g. ~/.cache/kagglehub); we mirror the download into data/ so the
# rest of the project has a stable, gitignored local path to read from.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"

CONFIG_FILENAMES = {"data.yaml", "classes.txt"}
ANNOTATION_KEYWORDS = ("caption", "annotation", "vlm")


def print_tree(root: Path, prefix: str = "") -> None:
    """Recursively print a folder tree rooted at `root`."""
    entries = sorted(root.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
    for i, entry in enumerate(entries):
        connector = "└── " if i == len(entries) - 1 else "├── "
        print(f"{prefix}{connector}{entry.name}")
        if entry.is_dir():
            extension = "    " if i == len(entries) - 1 else "│   "
            print_tree(entry, prefix + extension)


def find_config_files(root: Path) -> list[Path]:
    """Find data.yaml / classes.txt anywhere under root."""
    found = []
    for path in root.rglob("*"):
        if path.is_file() and path.name in CONFIG_FILENAMES:
            found.append(path)
    return sorted(found)


def find_annotation_files(root: Path) -> list[Path]:
    """Find likely caption/annotation JSON files anywhere under root."""
    found = []
    for path in root.rglob("*.json"):
        name_lower = path.name.lower()
        if any(keyword in name_lower for keyword in ANNOTATION_KEYWORDS):
            found.append(path)
    return sorted(found)


def print_file_contents(path: Path, max_chars: int = 3000) -> None:
    print(f"\n----- {path.relative_to(DATA_DIR) if path.is_relative_to(DATA_DIR) else path} -----")
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        print(f"  (could not read file: {exc})")
        return

    if path.suffix == ".json":
        try:
            data = json.loads(text)
            pretty = json.dumps(data, indent=2, ensure_ascii=False)
            text = pretty
        except json.JSONDecodeError:
            pass  # fall back to raw text

    if len(text) > max_chars:
        print(text[:max_chars])
        print(f"... [truncated, {len(text) - max_chars} more characters]")
    else:
        print(text)


def mirror_into_project(cache_path: Path) -> Path:
    """Copy the kagglehub cache download into the project's data/ dir."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if any(DATA_DIR.iterdir()):
        print(f"\ndata/ already has contents at {DATA_DIR}, skipping copy.")
        return DATA_DIR

    print(f"\nCopying dataset from kagglehub cache into {DATA_DIR} ...")
    for item in cache_path.iterdir():
        dest = DATA_DIR / item.name
        if item.is_dir():
            shutil.copytree(item, dest)
        else:
            shutil.copy2(item, dest)
    print("Copy complete.")
    return DATA_DIR


def main() -> None:
    print(f"Downloading dataset '{DATASET_SLUG}' via kagglehub ...")
    cache_path = Path(kagglehub.dataset_download(DATASET_SLUG))
    print(f"Dataset cached at: {cache_path}")

    local_root = mirror_into_project(cache_path)

    print(f"\n{'=' * 60}")
    print(f"Folder tree: {local_root}")
    print("=" * 60)
    print(local_root.name)
    print_tree(local_root)

    config_files = find_config_files(local_root)
    annotation_files = find_annotation_files(local_root)

    print(f"\n{'=' * 60}")
    print(f"Found {len(config_files)} config file(s): "
          f"{[str(p.relative_to(local_root)) for p in config_files]}")
    print(f"Found {len(annotation_files)} annotation/caption JSON file(s): "
          f"{[str(p.relative_to(local_root)) for p in annotation_files]}")
    print("=" * 60)

    for path in config_files:
        print_file_contents(path)

    for path in annotation_files:
        print_file_contents(path)

    if not config_files and not annotation_files:
        print("\nNo data.yaml, classes.txt, or caption/annotation JSON files "
              "found. Inspect the folder tree above manually.")


if __name__ == "__main__":
    main()
