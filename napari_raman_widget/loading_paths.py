"""Loading-panel paths read from the package-local ``.env`` file."""

import os
from pathlib import Path


ENV_PATH = Path(__file__).resolve().parent / ".env"
LOADING_PATH_KEYS = (
    "MICRO_MANAGER_CONFIG_PATH",
    "TRANSFORMER_MODEL_PATH",
    "LIGHTFIELD_EXPERIMENT_PATH",
    "VANDERMONDE_MODEL_PATH",
)


def _env_file_values(path=ENV_PATH):
    """Read simple KEY=VALUE entries without requiring python-dotenv."""
    path = Path(path)
    if not path.is_file():
        return {}

    values = {}
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        key, separator, value = line.partition("=")
        key = key.strip()
        if not separator or not key:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        values[key] = value
    return values


def load_loading_paths(path=ENV_PATH):
    """Return configured loading paths, with process variables taking priority."""
    file_values = _env_file_values(path)
    return {
        key: os.environ.get(key, file_values.get(key, ""))
        for key in LOADING_PATH_KEYS
    }


LOADING_PATHS = load_loading_paths()

