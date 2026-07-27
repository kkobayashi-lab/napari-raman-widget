"""Notebook compatibility import for the production large-dataset helpers."""

from pathlib import Path
import sys


_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from napari_raman_widget.large_dataset_viewing import *  # noqa: F403
