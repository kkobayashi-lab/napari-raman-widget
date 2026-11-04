from types import SimpleNamespace
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from napari_raman_widget.mda_timing import (
    estimate_mda_time,
    estimate_live_mda_end,
    format_duration,
    mda_schedule_window,
    AcquisitionOverheadLearner,
    remembered_overhead_seconds,
)


def _event(
    channel, exposure=None, *, p=0, z=0, t=0, min_start_time=0
):
    return SimpleNamespace(
        channel=SimpleNamespace(config=channel),
        exposure=exposure,
        index={"p": p, "z": z, "t": t},
        min_start_time=min_start_time,
    )


def _sequence(*events):
    return SimpleNamespace(iter_events=lambda: iter(events))


def _two_point_source():
    return SimpleNamespace(
        name="cells",
        get_mda_points=lambda _event: np.zeros((2, 2)),
    )


def test_two_one_second_raman_frames_take_three_point_two_seconds():
    estimate = estimate_mda_time(
        _sequence(_event("RM")),
        [_two_point_source()],
        raman_exposure_ms=1000,
        raman_z_indices=[0],
        image_positions=[0],
    )

    assert estimate.duration_seconds == 3.2
    assert estimate.raman_points == 2
    assert estimate.raman_events == 1
    assert estimate.acquisition_count == 1
    assert estimate.imaging_frames == 0
    assert format_duration(estimate.duration_seconds) == "3.2 seconds"


def test_selected_point_repeats_are_included_in_raman_time():
    estimate = estimate_mda_time(
        _sequence(_event("RM")),
        [_two_point_source()],
        raman_exposure_ms=1000,
        raman_z_indices=[0],
        image_positions=[0],
        point_repeats=3,
    )

    assert estimate.duration_seconds == pytest.approx(9.6)
    assert estimate.raman_points == 6
    assert estimate.raman_events == 1


def test_imaging_adds_exposure_and_overhead_per_frame():
    estimate = estimate_mda_time(
        _sequence(
            _event("BF", exposure=50),
            _event("GFP", exposure=200),
        ),
        [],
        raman_exposure_ms=1000,
        raman_z_indices=[],
        image_positions=[0],
    )

    assert estimate.duration_seconds == pytest.approx(0.45)
    assert estimate.imaging_frames == 2
    assert estimate.acquisition_count == 2


def test_scheduled_interval_is_included_in_expected_end():
    estimate = estimate_mda_time(
        _sequence(
            _event("RM", t=0),
            _event("RM", t=1, min_start_time=10),
        ),
        [_two_point_source()],
        raman_exposure_ms=1000,
        raman_z_indices=[0],
        image_positions=[0],
    )

    assert estimate.duration_seconds == 13.2
    assert estimate.scheduled_wait_seconds == pytest.approx(6.8)
    assert estimate.raman_points == 4


def test_non_imaging_positions_and_unselected_raman_z_are_skipped():
    estimate = estimate_mda_time(
        _sequence(
            _event("RM", z=1),
            _event("BF", exposure=100, p=1),
        ),
        [_two_point_source()],
        raman_exposure_ms=1000,
        raman_z_indices=[0],
        image_positions=[0],
    )

    assert estimate.duration_seconds == 0
    assert estimate.raman_points == 0
    assert estimate.imaging_frames == 0


def test_schedule_window_adds_delay_before_acquisition_duration():
    starting_time = datetime(2026, 7, 29, 8, 30, tzinfo=timezone.utc)

    scheduled_start, expected_end = mda_schedule_window(
        starting_time,
        delay_seconds=90,
        duration_seconds=3.2,
    )

    assert scheduled_start.strftime("%Y-%m-%d %H:%M:%S") == (
        "2026-07-29 08:31:30"
    )
    assert expected_end.strftime("%Y-%m-%d %H:%M:%S.%f") == (
        "2026-07-29 08:31:33.200000"
    )


def test_live_eta_uses_only_the_previous_ten_measurements():
    current_time = datetime(2026, 7, 29, 9, 0, tzinfo=timezone.utc)

    live = estimate_live_mda_end(
        completion_intervals=range(1, 13),
        completed=12,
        total=20,
        current_time=current_time,
    )

    assert live.sample_count == 10
    assert live.average_seconds == 7.5
    assert live.expected_end == current_time + timedelta(seconds=60)


def test_live_eta_is_unavailable_before_first_measurement():
    assert estimate_live_mda_end([], 0, 20, datetime.now()) is None


def _learn(kind="raman", exposure=100, count=1, overhead=0.6, wait=0):
    learner = AcquisitionOverheadLearner()
    for _ in range(10):
        learner.record(kind, (exposure / 1000 + overhead) * count + wait,
                       exposure, count, wait)
    return learner


def test_previous_run_100ms_predicts_500ms_with_same_overhead():
    _learn()
    # The learner is discarded: a new QSettings instance loads persisted overhead.
    assert remembered_overhead_seconds("raman") == pytest.approx(0.6)
    estimate = estimate_mda_time(
        _sequence(*[_event("RM") for _ in range(5)]),
        [_two_point_source()], 500, [0], [],
    )
    assert estimate.raman_points == 10
    assert estimate.duration_seconds == pytest.approx(11)
    assert dict(estimate.learned_overheads)["raman"] == pytest.approx(0.6)


def test_variable_exposures_counts_and_repeats_are_normalized():
    learner = AcquisitionOverheadLearner()
    for i in range(1, 11):
        learner.record("raman", (i / 10 + 0.2) * i, i * 100, i)
    estimate = estimate_mda_time(
        _sequence(_event("RM")), [_two_point_source()], 500, [0], [],
        point_repeats=3,
    )
    assert estimate.duration_seconds == pytest.approx(6 * 0.7)


def test_modes_are_learned_independently():
    _learn("raman", overhead=0.2)
    _learn("imaging", overhead=0.05)
    _learn("raman_batch", overhead=0.3, count=3)
    estimate = estimate_mda_time(
        _sequence(_event("RM"), _event("BF", exposure=200)),
        [_two_point_source()], 500, [0], [0], batch=True, point_repeats=3,
    )
    assert estimate.duration_seconds == pytest.approx(3 * 0.8 + 0.25)


def test_short_run_preserves_previous_overhead_and_window_rolls():
    learner = _learn(overhead=0.2)
    for _ in range(10):
        learner.record("raman", 0.5, 100)
    assert remembered_overhead_seconds("raman") == pytest.approx(0.4)
    learner = AcquisitionOverheadLearner()
    for _ in range(9):
        learner.record("raman", 100, 100)
    assert remembered_overhead_seconds("raman") == pytest.approx(0.4)


def test_waits_are_not_saved_as_overhead_or_counted_twice():
    _learn(overhead=0.2, wait=10)
    estimate = estimate_mda_time(
        _sequence(_event("RM"), _event("RM", min_start_time=10)),
        [_two_point_source()], 500, [0], [],
    )
    assert estimate.active_seconds == pytest.approx(2.8)
    assert estimate.duration_seconds == pytest.approx(11.4)


@pytest.mark.parametrize("value", ["broken", "nan", "inf", -1])
def test_invalid_saved_overhead_falls_back(value):
    from napari_raman_widget.mda_timing import _timing_settings
    _timing_settings().setValue("mda/overhead_v2/raman", value)
    estimate = estimate_mda_time(
        _sequence(_event("RM")), [_two_point_source()], 1000, [0], [0],
    )
    assert estimate.duration_seconds == 3.2
    assert not estimate.learned_overheads


def test_legacy_total_is_ignored_and_zero_overhead_is_valid():
    from napari_raman_widget.mda_timing import _timing_settings
    _timing_settings().setValue("mda/average_acquisition_seconds", 99)
    assert remembered_overhead_seconds("raman") is None
    _learn(overhead=0)
    estimate = estimate_mda_time(
        _sequence(_event("RM")), [_two_point_source()], 500, [0], [],
    )
    assert estimate.duration_seconds == 1
