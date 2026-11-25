"""One configuration file, one random seed."""
from pathlib import Path
import hashlib
import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_config():
    return yaml.safe_load((ROOT / 'config.yaml').read_text())


def config_hash():
    return hashlib.sha256((ROOT / 'config.yaml').read_bytes()).hexdigest()
