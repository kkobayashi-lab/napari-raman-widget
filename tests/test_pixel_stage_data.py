import json

import numpy as np
import pytest
import tifffile
import xarray as xr
from useq import MDASequence

from napari_raman_widget.pixel_stage_data import load_pixel_stage_data


def _sequence(path, sequence=None):
    sequence = sequence or {
        "stage_positions": [{"x": i * 10, "y": i * 20} for i in range(5)]
    }
    (path / "useq-sequence.json").write_text(json.dumps(sequence))
    return sequence


def test_current_ome_camera_only_sparse_positions(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    _sequence(tmp_path)
    # Same naming/OME position metadata as current camera output. No Raman file.
    for p in (0, 2, 4):
        tifffile.imwrite(
            images / f"p{p}.ome.tiff",
            np.full((2, 2, 4, 5), p, dtype=np.uint16),
            ome=True,
            metadata={"axes": "TCYX", "Name": f"p{p}",
                      "Channel": {"Name": ["BF", "GFP"]}},
        )
    frames, xy = load_pixel_stage_data(tmp_path)
    assert frames.shape == (3, 4, 5)
    np.testing.assert_array_equal(frames[:, 0, 0], [0, 2, 4])
    np.testing.assert_array_equal(xy, [[0, 0], [20, 40], [40, 80]])
    # Selecting the images subfolder also finds the run sequence.
    np.testing.assert_array_equal(load_pixel_stage_data(images)[1], xy)


def test_legacy_tiffs_use_one_common_plane_without_filling(tmp_path):
    _sequence(tmp_path)
    for p in (0, 2, 4):
        tifffile.imwrite(tmp_path / f"t000_p{p:03}_c000_z000.tiff",
                         np.full((4, 5), p, dtype=np.uint16))
    tifffile.imwrite(tmp_path / "t001_p001_c000_z000.tiff",
                     np.ones((4, 5), dtype=np.uint16))
    frames, xy = load_pixel_stage_data(tmp_path)
    np.testing.assert_array_equal(frames[:, 0, 0], [0, 2, 4])
    np.testing.assert_array_equal(xy[:, 0], [0, 20, 40])


def test_dense_grid_positions_follow_snake_order(tmp_path):
    seq = MDASequence(channels=["BF"], metadata={"raman_grid": {
        "version": 1, "regions": [{"x": [10, 30, 3], "y": [40, 50, 2],
                                    "z": 0, "snake_axis": "x"}]
    }})
    _sequence(tmp_path, json.loads(seq.model_dump_json()))
    for p in (0, 3, 5):
        tifffile.imwrite(tmp_path / f"t000_p{p:03}_c000_z000.tiff",
                         np.zeros((4, 5), dtype=np.uint16))
    np.testing.assert_array_equal(load_pixel_stage_data(tmp_path)[1],
                                  [[10, 40], [30, 50], [10, 50]])


def test_legacy_zarr_uses_position_coordinates(tmp_path):
    ds = xr.Dataset({"image": (("t", "p", "c", "z", "y", "x"),
                                np.zeros((1, 3, 1, 1, 4, 5), dtype=np.uint16))},
                    coords={"p": [0, 2, 4]})
    ds.attrs["useq_sequence"] = json.dumps(_sequence(tmp_path))
    path = tmp_path / "legacy.zarr"
    ds.to_zarr(path)
    np.testing.assert_array_equal(load_pixel_stage_data(path)[1][:, 0], [0, 20, 40])


@pytest.mark.parametrize("failure", ["missing_sequence", "bad_index", "bad_xy", "native_grid"])
def test_invalid_coordinate_pairing_is_rejected(tmp_path, failure):
    seq = _sequence(tmp_path)
    if failure == "missing_sequence":
        (tmp_path / "useq-sequence.json").unlink()
    elif failure == "bad_index":
        seq["stage_positions"] = seq["stage_positions"][:2]
    elif failure == "bad_xy":
        seq["stage_positions"][0]["x"] = None
    else:
        seq["grid_plan"] = {"rows": 2, "columns": 2}
    if failure != "missing_sequence":
        _sequence(tmp_path, seq)
    for p in range(3):
        tifffile.imwrite(tmp_path / f"t000_p{p:03}_c000_z000.tiff",
                         np.zeros((4, 5), dtype=np.uint16))
    with pytest.raises(ValueError):
        load_pixel_stage_data(tmp_path)


def test_no_camera_frames_is_rejected(tmp_path):
    _sequence(tmp_path)
    with pytest.raises((ValueError, FileNotFoundError), match="No legacy|no.*images"):
        load_pixel_stage_data(tmp_path)
