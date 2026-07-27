import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import tifffile


_MODULE_PATH = (
    Path(__file__).parents[1]
    / "napari_raman_widget"
    / "large_dataset_viewing.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "large_dataset_viewing", _MODULE_PATH
)
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


def test_separate_folders_lazy_loading_and_stage_stitch(tmp_path):
    imaging = tmp_path / "imaging"
    raman = tmp_path / "spectra"
    metadata = tmp_path / "metadata"
    imaging.mkdir()
    raman.mkdir()
    metadata.mkdir()

    tifffile.imwrite(
        imaging / "t000_p000_c000_z000.tiff",
        np.ones((4, 4), dtype=np.uint16),
    )
    tifffile.imwrite(
        imaging / "t000_p001_c000_z000.tiff",
        np.full((4, 4), 2, dtype=np.uint16),
    )
    tifffile.imwrite(
        imaging / "t000_p000_c001_z000.tiff",
        np.full((4, 4), 3, dtype=np.uint16),
    )
    tifffile.imwrite(
        imaging / "t000_p001_c001_z000.tiff",
        np.full((4, 4), 4, dtype=np.uint16),
    )
    np.save(
        raman / "raman_p000_t000_z000_data.npy",
        np.arange(6, dtype=np.uint16).reshape(2, 3),
    )
    np.save(
        raman / "raman_p000_t000_z000_locations.npy",
        np.array([[0.5, 0.5], [0.5, 0.5]]),
    )
    (raman / "raman_p000_t000_z000_meta.json").write_text(
        json.dumps({"x": 7, "y": 0, "z": 0}),
        encoding="utf-8",
    )
    sequence_file = metadata / "useq-sequence.json"
    sequence_file.write_text(
        json.dumps(
            {
                "channels": [
                    {"config": "BF"},
                    {"config": "DAPI"},
                ],
                "stage_positions": [
                    {"x": 0, "y": 0, "z": 0},
                    {"x": 8, "y": 0, "z": 0},
                ]
            }
        ),
        encoding="utf-8",
    )
    model_file = metadata / "vandermonde.json"
    model_file.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "objectives": {
                    "test": {
                        "degree": 1,
                        "C": [[0, 0], [0, 2], [2, 0]],
                        "img_center": [2, 2],
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    acquisition = _MODULE.AcquisitionIndex.build(
        imaging_folder=imaging,
        raman_folder=raman,
        sequence_file=sequence_file,
    )

    assert acquisition.summary()["image_files"] == 4
    assert acquisition.summary()["raman_files"] == 1
    np.testing.assert_array_equal(
        acquisition.load_spectra(_MODULE.RamanKey(0, 0, 0)),
        np.arange(6, dtype=np.uint16).reshape(2, 3),
    )
    geometry = _MODULE.load_vandermonde_geometry(model_file, "test")
    np.testing.assert_allclose(
        geometry["stage_to_image_pixels_per_um"],
        np.eye(2) / 2,
    )
    assert geometry["pixel_size_mean_um"] == 2
    match = _MODULE.find_image_for_raman_index(
        acquisition, 0, geometry
    )
    assert match.image_key == _MODULE.ImageKey(0, 1, 0, 0)
    np.testing.assert_allclose(match.image_pixel_xy, [2.5, 2])
    assert match.center_distance_pixels == 0.5
    dapi_match = _MODULE.find_image_for_raman_index(
        acquisition, 0, geometry, imaging_channel="dapi"
    )
    assert dapi_match.image_key == _MODULE.ImageKey(0, 1, 1, 0)
    stitched_mosaic, marker_xy, stitch_info, stitched_match = (
        _MODULE.raman_stitched_preview(
            acquisition,
            0,
            geometry,
            imaging_channel="BF",
            max_side=100,
        )
    )
    assert stitched_match == match
    np.testing.assert_allclose(marker_xy, [2.5, 2])
    assert stitch_info["raman_index"] == 0
    cached_mosaic, *_ = _MODULE.raman_stitched_preview(
        acquisition,
        0,
        geometry,
        imaging_channel=0,
        max_side=100,
    )
    assert cached_mosaic is stitched_mosaic
    (raman / "raman_p000_t000_z000_meta.json").write_text(
        json.dumps({"x": 100, "y": 100, "z": 0}),
        encoding="utf-8",
    )
    assert _MODULE.find_image_for_raman_index(
        acquisition, 0, geometry
    ) is None

    mosaic, info = _MODULE.stitch_preview(
        acquisition,
        t=0,
        c=0,
        z=0,
        stage_to_image=geometry["stage_to_image_pixels_per_um"],
        max_side=100,
    )
    assert mosaic.shape == (4, 8)
    np.testing.assert_array_equal(mosaic[:, :4], 2)
    np.testing.assert_array_equal(mosaic[:, 4:], 1)
    assert info["tiles"] == 2
    assert info["downsample_stride"] == 1

    configuration = _MODULE.fiji_tile_configuration(
        acquisition,
        t=0,
        c=0,
        z=0,
        stage_to_image=geometry["stage_to_image_pixels_per_um"],
    )
    assert "dim = 2" in configuration
    assert "(0.000, 0.000)" in configuration
    assert "(4.000, 0.000)" in configuration

