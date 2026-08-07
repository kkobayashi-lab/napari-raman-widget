from types import SimpleNamespace
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from napari_raman_widget.mda_timing import (
    estimate_mda_time,
    estimate_live_mda_end,
    format_duration,
    mda_schedule_window,
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
