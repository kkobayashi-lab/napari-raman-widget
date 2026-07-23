import json
import importlib.util
from pathlib import Path

import numpy as np

_DATASET_PATH = (
    Path(__file__).parents[1] / "napari_raman_widget" / "dataset.py"
)
_SPEC = importlib.util.spec_from_file_location("_raman_dataset", _DATASET_PATH)
_DATASET = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_DATASET)
_load_raman = _DATASET._load_raman


def test_load_raman_uses_saved_wavenumber_axis(tmp_path):
    axis = np.array([100.5, 101.5, 102.5])
    np.save(tmp_path / "wavenumbers.npy", axis)
    np.save(
        tmp_path / "raman_p000_t000_z000_data.npy",
        np.array([[1, 2, 3], [4, 5, 6]]),
    )
    np.save(
        tmp_path / "raman_p000_t000_z000_locations.npy",
        np.array([[0.25, 0.5], [0.75, 1.0]]),
    )
    with open(tmp_path / "raman_p000_t000_z000_meta.json", "w") as stream:
        json.dump({"time": "2026-07-23T12:00:00"}, stream)

    df, _, _ = _load_raman(
        tmp_path, img_x=100, img_y=200, batch=False
    )

    assert list(df.columns[:3]) == list(axis)
    assert df.attrs["spectral_axis"] == "wavenumber"
    assert df.attrs["spectral_axis_units"] == "cm^-1"


def test_load_raman_rejects_mismatched_axis(tmp_path):
    np.save(tmp_path / "wavenumbers.npy", np.array([100.5, 101.5]))
    np.save(
        tmp_path / "raman_p000_t000_z000_data.npy",
        np.array([[1, 2, 3]]),
    )

    try:
        _load_raman(tmp_path, img_x=100, img_y=200, batch=False)
    except ValueError as error:
        assert "length does not match" in str(error)
    else:
        raise AssertionError("Expected mismatched wavenumber axis to fail")
