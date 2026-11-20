"""Resolve explicitly registered imported runs without overwriting local training."""
import json
from pathlib import Path
from src.experiments.quality_dataset import sha256

ROOT = Path(__file__).resolve().parents[2]


def research_runs(root=ROOT):
    """Keep the original eight runs, adding all B seeds once B is registered."""
    names = [f'matched_{e}_seed0' for e in 'abcde'] + [f'expanded_a_seed{s}' for s in (0, 1, 2)]
    path = root / 'outputs/research_runs.json'
    registered = json.loads(path.read_text()) if path.exists() else {}
    if any(f'expanded_b_seed{s}' in registered for s in (0, 1, 2)):
        names += [f'expanded_b_seed{s}' for s in (0, 1, 2)]
    return names


def run_paths(name, root=ROOT):
    registry = root / 'outputs/research_runs.json'
    records = json.loads(registry.read_text()) if registry.exists() else {}
    if name not in records:
        return root / 'outputs/checkpoints/research_v1' / name, root / 'outputs/logs/research_v1' / name
    record = records[name]
    model, log = root / record['checkpoint_dir'], root / record['log_dir']
    for path in [model, log]:
        if not path.resolve().is_relative_to((root / 'outputs').resolve()):
            raise ValueError('Registered artifacts must remain inside outputs')
    for relative, expected in record['files'].items():
        path = root / relative
        if not path.resolve().is_relative_to((root / 'outputs').resolve()) or sha256(path) != expected:
            raise ValueError(f'Changed registered artifact: {relative}')
    return model, log
