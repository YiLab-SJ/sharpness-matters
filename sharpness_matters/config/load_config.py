from __future__ import annotations
from pathlib import Path
import yaml

# Path to the config file relative to this script
_CFG_PATH = Path(__file__).resolve().parent / "config.yaml"


def load_config() -> dict:
    """Load the YAML config file from the fixed location."""
    with open(_CFG_PATH, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg or {}


class DotDict(dict):
    """Allow dot access (e.g. cfg.pneumothorax.dataset_a)."""

    __getattr__ = dict.get

    def __setattr__(self, k, v):
        self[k] = v

    def __delattr__(self, k):
        del self[k]


def _to_dotdict(d):
    """Recursively convert nested dicts to DotDicts."""
    if isinstance(d, dict):
        return DotDict({k: _to_dotdict(v) for k, v in d.items()})
    return d


# Load once and expose as global singleton
cfg = _to_dotdict(load_config())
