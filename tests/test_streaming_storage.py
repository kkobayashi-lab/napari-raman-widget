import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
from pymmcore_plus.mda.events import MDASignaler
from useq import MDAEvent, MDASequence

from raman_mda_engine import RamanAcquisitionWriter
from napari_raman_widget.dataset import load_experiment


_MODULE_PATH = (
    Path(__file__).parents[1]
    / "napari_raman_widget"
    / "large_dataset_viewing.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "streaming_large_dataset_viewing", _MODULE_PATH
)
_VIEWING = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _VIEWING
_SPEC.loader.exec_module(_VIEWING)


class _FakeMDA:
    def __init__(self):
        self.events = MDASignaler()
        self.engine = object()

    @staticmethod
    def status():
        return SimpleNamespace(finish_reason="completed")


class _FakeCore:
    def __init__(self):
        self.mda = _FakeMDA()

    @staticmethod
    def getPixelSizeUm():
        return 0.25

    @staticmethod
    def getXYPosition():
        return (10.5, 20.5)

    @staticmethod
    def getZPosition():
        return 30.5


def test_streaming_run_roundtrips_through_lazy_index(tmp_path):
    core = _FakeCore()
    writer = RamanAcquisitionWriter(
        tmp_path / "run",
        core=core,
        wavenumbers=np.array([100.0, 200.0, 300.0]),
        image_positions=[0, 2],
        batch=True,
    )
    sequence = MDASequence(
        axis_order="tpzc",
        time_plan={"interval": 0, "loops": 2},
        stage_positions=[(0, 0, 0), (1, 0, 0), (2, 0, 0)],
        channels=["BF", "RM", "GFP"],
        z_plan={"relative": [-1, 1]},
        metadata={"raman": {"channel": "RM", "z": [0]}},
    )

    core.mda.events.sequenceStarted.emit(sequence, {})
    expected = {}
    value = 1
    for event in sequence.iter_events():
        if (
            event.channel is not None
            and event.channel.config != "RM"
            and event.index["p"] in {0, 2}
        ):
            key = (
                event.index["t"],
                event.index["p"],
                0 if event.channel.config == "BF" else 1,
                event.index["z"],
            )
            expected[key] = value
            core.mda.events.frameReady.emit(
                np.full((4, 5), value, dtype=np.uint16),
                event,
                {"position": {"x": value, "y": 2, "z": 3}},
            )
            value += 1

    writer._save_raman(
        MDAEvent(index={"t": 0, "p": 1, "z": 0}),
        np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint16),
        np.array([[0.25, 0.5]]),
        ["cell"],
        125.0,
    )
    writer._save_raman(
        MDAEvent(index={"t": 1, "p": 2, "z": 1}),
        np.array([7, 8, 9], dtype=np.uint16),
        np.array([[0.1, 0.2], [0.3, 0.4]]),
        ["cell", "background"],
        250.0,
    )
    core.mda.events.sequenceFinished.emit(sequence)

    acquisition = _VIEWING.AcquisitionIndex.build(writer.path)
    assert set(key.p for key in acquisition.image_keys) == {0, 2}
    assert acquisition.image_channel_names == ("BF", "GFP")
    assert acquisition.raman_fov_source_positions == (0, 2)
    assert acquisition.summary()["image_files"] == len(expected)
    for key, expected_value in expected.items():
        np.testing.assert_array_equal(
            acquisition.load_image(_VIEWING.ImageKey(*key)),
            expected_value,
        )

    first_key = _VIEWING.RamanKey(0, 1, 0)
    np.testing.assert_array_equal(
        acquisition.load_spectra(first_key),
        [[1, 2, 3], [4, 5, 6]],
    )
    np.testing.assert_allclose(
        acquisition.load_locations(first_key), [[0.25, 0.5]]
    )
    assert acquisition.load_designations(first_key) == ("cell",)
    first_fov_cells = acquisition.raman_cells(0)
    assert len(first_fov_cells) == 1
    assert first_fov_cells[0].repeat_count == 2
    np.testing.assert_allclose(first_fov_cells[0].point_yx, [0.25, 0.5])
    second_fov_cells = acquisition.raman_cells(1)
    assert len(second_fov_cells) == 1
    assert second_fov_cells[0].designation == "integrated batch"
    metadata = acquisition.load_raman_metadata(first_key)
    assert metadata["exposure_ms"] == 125.0
    np.testing.assert_allclose(metadata["stage_xyz_um"], [10.5, 20.5, 30.5])
    axis, _ = acquisition.spectral_axis(3)
    np.testing.assert_allclose(axis, [100, 200, 300])

    dataframe, locations, images = load_experiment(
        writer.path,
        zarr_output=tmp_path / "legacy-export.zarr",
        batch=None,
    )
    assert len(dataframe) == 3
    assert locations["designation"].tolist() == [
        "cell",
        "cell",
        "background",
    ]
    assert images.dims == ("t", "p", "c", "z", "y", "x")
    assert images.shape == (2, 3, 2, 2, 4, 5)


def test_same_point_in_different_cell_layers_remains_distinct(tmp_path):
    core = _FakeCore()
    writer = RamanAcquisitionWriter(
        tmp_path / "layered-run",
        core=core,
        wavenumbers=np.array([100.0, 200.0, 300.0]),
    )
    sequence = MDASequence(
        stage_positions=[(0, 0, 0)],
        channels=["BF", "RM"],
        metadata={
            "raman": {
                "channel": "RM",
                "z": [0],
                "cell_layers": [
                    {
                        "id": "a",
                        "name": "Type A",
                        "source_name": "cell:Type A",
                        "color": "#aa0000ff",
                    },
                    {
                        "id": "b",
                        "name": "Type B",
                        "source_name": "cell:Type B",
                        "color": "#0066ccff",
                    },
                ],
            }
        },
    )

    core.mda.events.sequenceStarted.emit(sequence, {})
    bf_event = next(
        event
        for event in sequence.iter_events()
        if event.channel is not None and event.channel.config == "BF"
    )
    core.mda.events.frameReady.emit(
        np.ones((4, 5), dtype=np.uint16),
        bf_event,
        {},
    )
    writer._save_raman(
        MDAEvent(index={"t": 0, "p": 0, "z": 0}),
        np.array(
            [[1, 2, 3], [7, 8, 9], [4, 5, 6]],
            dtype=np.uint16,
        ),
        np.array(
            [[0.25, 0.5], [0.75, 0.5], [0.25, 0.5]],
        ),
        ["cell:Type A", "cell:Type A", "cell:Type B"],
        125.0,
    )
    core.mda.events.sequenceFinished.emit(sequence)

    acquisition = _VIEWING.AcquisitionIndex.build(writer.path)
    cells = acquisition.raman_cells(0)
    assert len(cells) == 3
    assert [cell.designation for cell in cells] == [
        "cell:Type A",
        "cell:Type A",
        "cell:Type B",
    ]
    assert [cell.layer_index for cell in cells] == [0, 0, 1]
    assert [cell.layer_cell_index for cell in cells] == [0, 1, 0]
    assert [cell.repeat_count for cell in cells] == [1, 1, 1]
    layers = acquisition.raman_cell_layers(0)
    assert [
        (layer.layer_index, layer.name, layer.designation, layer.cell_count)
        for layer in layers
    ] == [
        (0, "Type A", "cell:Type A", 2),
        (1, "Type B", "cell:Type B", 1),
    ]
    type_b_cells = acquisition.raman_cells(
        0,
        cell_layer="cell:Type B",
    )
    assert len(type_b_cells) == 1
    assert type_b_cells[0].cell_index == 2
    selected = acquisition.raman_cell(
        0,
        0,
        cell_layer="cell:Type B",
    )
    assert selected.designation == "cell:Type B"
    type_b_spectrum = _VIEWING._mean_raman_spectrum(
        acquisition,
        0,
        0,
        cell_layer="cell:Type B",
    )
    np.testing.assert_array_equal(type_b_spectrum[1], [4, 5, 6])
    assert "layer 1 (Type B), cell 0" in type_b_spectrum[-1]
    assert acquisition.sequence["metadata"]["raman"]["cell_layers"][1]["name"] == (
        "Type B"
    )
