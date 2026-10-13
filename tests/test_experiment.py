from types import SimpleNamespace

import numpy as np
import pytest
from useq import MDAEvent, MDASequence, Position

from napari_raman_widget.experiment import (
    ExperimentExtensions, METADATA_KEY, capture_targets, identify_positions, indices,
    prepare_acquisition,
)
from napari_raman_widget.mda_timing import estimate_mda_time


def targets(sequence):
    source = SimpleNamespace(
        name="cell:A", role="cell", source_id="a", _img_shape=(100, 200),
        get_mda_points=lambda event, transform=True: np.array([[0.2 + event.index['p'] / 10, 0.4]]),
    )
    return capture_targets(sequence, [source])


def test_camera_only_needs_no_raman_targets_or_z_plan():
    sequence = MDASequence(channels=[{"config": "BF", "exposure": 10}])
    plan = prepare_acquisition(sequence, ExperimentExtensions(), [])
    assert not plan.raman_enabled
    assert plan.sources == ()
    assert plan.sequence.z_plan is None


def test_target_reordering_preserves_identity_and_detaches_points():
    sequence = identify_positions(MDASequence(
        stage_positions=[Position(x=1, y=2), Position(x=3, y=4)],
        channels=[{"config": "RM", "exposure": 100}],
    ))
    records = targets(sequence)
    reordered = sequence.replace(stage_positions=sequence.stage_positions[::-1])
    plan = prepare_acquisition(reordered, ExperimentExtensions(), records)
    points = plan.sources[0].get_mda_points(MDAEvent(index={"p": 0}))
    assert points[0, 0] == pytest.approx(0.3)
    records[0]["points"]["1"][0][0] = 0.9
    assert plan.sources[0].get_mda_points(MDAEvent(index={"p": 0}))[0, 0] == pytest.approx(0.3)


def test_deleted_or_duplicated_position_blocks_run():
    sequence = identify_positions(MDASequence(stage_positions=[Position(x=1), Position(x=2)]))
    records = targets(sequence)
    with pytest.raises(ValueError, match="deleted"):
        prepare_acquisition(sequence.replace(stage_positions=sequence.stage_positions[:1]), ExperimentExtensions(), records)
    with pytest.raises(ValueError, match="identities"):
        prepare_acquisition(sequence.replace(stage_positions=[sequence.stage_positions[0]] * 2), ExperimentExtensions(), records)


def test_raman_exposure_is_channel_exposure_and_sparse_camera_is_optional():
    sequence = identify_positions(MDASequence(
        stage_positions=[Position(x=1)], channels=[{"config": "RM", "exposure": 150}],
        z_plan={"relative": [0, 2]}, metadata={"other": {"preserved": True}},
    ))
    ext = ExperimentExtensions()
    ext.raman.camera_positions = "none"
    ext.raman.planes = "middle"
    plan = prepare_acquisition(sequence, ext, targets(sequence))
    assert plan.raman_exposure == 150
    assert plan.raman_planes == (1,)
    assert plan.image_positions == ()
    assert plan.sequence.metadata["other"] == {"preserved": True}
    assert plan.sequence.channels == sequence.channels
    assert plan.sequence.z_plan == sequence.z_plan


@pytest.mark.parametrize("nested", [False, True])
def test_unconfigured_hardware_autofocus_is_stripped(nested):
    from useq import HardwareAutofocus

    ext = ExperimentExtensions()
    ext.autofocus.mode = "laser"
    ext.autofocus.axes = ()
    ext.autofocus.positions = "invalid custom positions"
    af = {"axes": ("p",)}
    sequence = MDASequence(
        channels=["BF"], z_plan={"relative": [0, 2]},
        stage_positions=[Position(x=1, sequence=MDASequence(autofocus_plan=af) if nested else None)],
        autofocus_plan=None if nested else af,
    )
    plan = prepare_acquisition(sequence, ext, [])
    assert plan.extensions.autofocus.mode == "off"
    assert plan.sequence.autofocus_plan is None
    assert not any(isinstance(event.action, HardwareAutofocus) for event in plan.sequence)
    assert ext.autofocus.mode == "laser"
    assert ext.autofocus.positions == "invalid custom positions"



def test_custom_autofocus_rejects_absolute_z_without_hardware_autofocus():
    ext = ExperimentExtensions()
    ext.autofocus.mode = "software"
    with pytest.raises(ValueError, match="relative"):
        prepare_acquisition(MDASequence(z_plan={"absolute": [1, 2]}), ext, [])


def test_extensions_round_trip_and_snapshot_is_not_editor_state():
    ext = ExperimentExtensions()
    ext.tracking.every = 3
    ext.autofocus.every = 2
    restored = ExperimentExtensions.from_dict(ext.to_dict())
    assert restored == ext
    plan = prepare_acquisition(MDASequence(channels=["BF"]), ext, [])
    ext.tracking.every = 8
    assert plan.extensions.tracking.every == 3


def test_indices_validate_and_batch_timing_counts_integrated_collection():
    assert indices("middle", 4, middle=True) == [2]
    assert indices("none", 4) == []
    with pytest.raises(ValueError):
        indices("4", 4)
    sequence = MDASequence(channels=[{"config": "RM", "exposure": 100}])
    source = SimpleNamespace(name="cell", get_mda_points=lambda event: np.zeros((5, 2)))
    estimate = estimate_mda_time(sequence, [source], 100, [0], [], batch=True)
    assert estimate.raman_points == 1
    assert estimate.imaging_frames == 0
    assert estimate.active_seconds == pytest.approx(0.7)


def test_hardware_autofocus_is_not_counted_as_camera_frame():
    from useq import HardwareAutofocus
    sequence = MDASequence(stage_positions=[Position(x=1)], channels=["BF"],
                           autofocus_plan={"axes": ("p",)})
    estimate = estimate_mda_time(sequence, [], 0, [], [0])
    assert estimate.imaging_frames == 1
    assert any(isinstance(event.action, HardwareAutofocus) for event in sequence)


def test_dense_grid_eta_and_target_snapshot_do_not_expand_positions():
    from raman_mda_engine.grid_plan import GRID_KEY
    sequence = MDASequence(channels=[{"config": "RM", "exposure": 100}],
                           time_plan={"loops": 2, "interval": 0},
                           metadata={GRID_KEY: {"version": 1, "regions": [
                               {"x": [0, 100, 1000], "y": [0, 100, 1000],
                                "z": 0, "tilt": None, "snake_axis": "x"}]}})
    source = SimpleNamespace(name="cell", role="cell", _img_shape=(100, 100),
                             get_mda_points=lambda event, transform=True: np.array([[0.5, 0.5], [0.5, 0.5]]))
    records = capture_targets(MDASequence(stage_positions=[Position()]), [source])
    records[0].update(fixed=True, position_ids=[])
    ext = ExperimentExtensions()
    ext.raman.camera_positions = "none"
    plan = prepare_acquisition(sequence, ext, records)
    assert plan.autofocus_positions == ()
    assert len(plan.sources[0]._points_by_position) == 1
    estimate = estimate_mda_time(plan.sequence, plan.sources, 100, [0], [])
    assert estimate.raman_points == 4_000_000
    assert estimate.raman_events == 2_000_000
    assert estimate.active_seconds == pytest.approx(2_800_000)
    plan.sources[0].set_mda_points(0, [[60, 60], [60, 60]])
    np.testing.assert_allclose(plan.sources[0].get_mda_points(MDAEvent(index={"p": 999999})),
                               [[0.5, 0.5], [0.5, 0.5]])
