"""Timing estimates for Raman multidimensional acquisitions."""
from dataclasses import dataclass
from datetime import datetime, timedelta
from math import isfinite


RAMAN_POINT_OVERHEAD_S = 0.6
IMAGING_FRAME_OVERHEAD_S = 0.1


def _timing_settings():
    from qtpy.QtCore import QSettings

    return QSettings("CNS", "napari-raman-widget")


def remembered_overhead_seconds(kind):
    """Read exposure-independent overhead; legacy total-time values are ignored."""
    try:
        value = float(_timing_settings().value(f"mda/overhead_v2/{kind}"))
    except (TypeError, ValueError, OverflowError):
        return None
    return value if isfinite(value) and value >= 0 else None


class AcquisitionOverheadLearner:
    """Retain ten completion samples per acquisition mode, normalized per spectrum."""

    def __init__(self):
        self.samples = {}

    def record(self, kind, interval, exposure_ms, count=1, scheduled_wait=0):
        if kind not in ("raman", "raman_batch", "imaging"):
            return
        if exposure_ms is None:
            return
        exposure = float(exposure_ms) / 1000.0
        if (count < 1 or not all(isfinite(v) for v in
                (interval, exposure, scheduled_wait))
                or min(interval, exposure, scheduled_wait) < 0):
            return
        overhead = max(0.0, interval - scheduled_wait - exposure * count)
        samples = self.samples.setdefault(kind, [])
        samples.append((overhead, count))
        del samples[:-10]
        if len(samples) == 10:
            _timing_settings().setValue(
                f"mda/overhead_v2/{kind}",
                sum(value for value, _ in samples) / sum(n for _, n in samples),
            )


def _is_cell_source(source):
    role = getattr(source, "role", None)
    return role == "cell" if role is not None else "cell" in source.name.lower()


@dataclass(frozen=True)
class MdaTimeEstimate:
    """Acquisition-only MDA timing estimate."""

    duration_seconds: float
    active_seconds: float
    raman_points: int
    raman_events: int
    imaging_frames: int
    learned_overheads: tuple = ()

    @property
    def scheduled_wait_seconds(self):
        return max(0.0, self.duration_seconds - self.active_seconds)

    @property
    def acquisition_count(self):
        return self.raman_events + self.imaging_frames


@dataclass(frozen=True)
class LiveMdaEstimate:
    """Rolling estimate derived from recently completed acquisitions."""

    expected_end: datetime
    average_seconds: float
    sample_count: int
    completed: int
    total: int


def _event_min_start_seconds(event):
    """Return a useq event's scheduled start as seconds from MDA start."""
    value = getattr(event, "min_start_time", None)
    if value is None:
        return 0.0
    if hasattr(value, "total_seconds"):
        return float(value.total_seconds())
    return float(value)


def estimate_mda_time(
    sequence,
    aiming_sources,
    raman_exposure_ms,
    raman_z_indices,
    image_positions,
    raman_channel="RM",
    point_repeats=1,
    batch=False,
):
    """Estimate from remembered live timing, falling back to fixed overhead.

    Raman events use the same cell-source filtering as ``RamanEngine``.
    Learned overhead is stored per spectrum/frame and acquisition mode.
    Current exposures and repeats are added independently of historical exposure.
    Explicit scheduled starts remain lower bounds on event start times.
    """
    elapsed = 0.0
    active = 0.0
    raman_points = 0
    raman_events = 0
    imaging_frames = 0
    overheads = {kind: remembered_overhead_seconds(kind)
                 for kind in ("raman", "raman_batch", "imaging")}
    used_overheads = {}
    raman_z = set(int(index) for index in raman_z_indices)
    image_p = image_positions if isinstance(image_positions, range) else set(int(index) for index in image_positions)
    raman_exposure_s = float(raman_exposure_ms) / 1000.0
    point_repeats = max(1, int(point_repeats))

    # useq autofocus plans retain their previous event while iterating. A preview
    # must not consume the autofocus state of the sequence that will be executed.
    from useq import MDASequence
    preview_sequence = (
        MDASequence.model_validate_json(sequence.model_dump_json())
        if isinstance(sequence, MDASequence) else sequence
    )
    virtual_count = 1
    dense = isinstance(preview_sequence, MDASequence) and "raman_grid" in preview_sequence.metadata
    if dense:
        from copy import deepcopy
        from raman_mda_engine.grid_plan import grid_count, iter_grid_events
        virtual_count = grid_count(preview_sequence)
        metadata = deepcopy(preview_sequence.metadata)
        region = metadata["raman_grid"]["regions"][0]
        region["x"][2] = region["y"][2] = 1
        metadata["raman_grid"]["regions"] = [region]
        preview_sequence = preview_sequence.replace(metadata=metadata)
        events = iter_grid_events(preview_sequence)
    else:
        events = preview_sequence.iter_events()
    for event in events:
        from useq import AcquireImage
        if not isinstance(getattr(event, "action", None), (AcquireImage, type(None))):
            continue
        elapsed = max(elapsed, _event_min_start_seconds(event))
        channel = (
            event.channel.config
            if getattr(event, "channel", None) is not None
            else None
        )
        event_seconds = 0.0

        if channel == raman_channel:
            if event.index.get("z", 0) in raman_z:
                point_count = sum(
                    len(source.get_mda_points(event))
                    for source in aiming_sources
                    if _is_cell_source(source)
                ) * point_repeats
                policy = getattr(preview_sequence, "metadata", {}).get("raman_positions", {}).get(str(event.index.get("p", 0)), {})
                event_batch = policy.get("mode") == "integrated_batch" if policy else batch
                if event_batch and point_count:
                    point_count = point_repeats
                point_count *= virtual_count
                raman_points += point_count
                if point_count:
                    raman_events += virtual_count
                kind = "raman_batch" if event_batch else "raman"
                overhead = overheads[kind]
                if overhead is not None and point_count:
                    used_overheads[kind] = overhead
                event_seconds = point_count * (
                    raman_exposure_s + (RAMAN_POINT_OVERHEAD_S
                                        if overhead is None else overhead)
                )
        elif (dense and bool(image_p)) or event.index.get("p", 0) in image_p:
            exposure_ms = getattr(event, "exposure", None) or 0.0
            frame_count = len(image_p) if dense else 1
            imaging_frames += frame_count
            overhead = overheads["imaging"]
            if overhead is not None and frame_count:
                used_overheads["imaging"] = overhead
            event_seconds = (
                float(exposure_ms) / 1000.0 + (IMAGING_FRAME_OVERHEAD_S
                                             if overhead is None else overhead)
            ) * frame_count
        active += event_seconds
        elapsed += event_seconds

    return MdaTimeEstimate(
        duration_seconds=elapsed,
        active_seconds=active,
        raman_points=raman_points,
        raman_events=raman_events,
        imaging_frames=imaging_frames,
        learned_overheads=tuple(used_overheads.items()),
    )


def format_duration(seconds):
    """Format a duration compactly while preserving sub-minute precision."""
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:.1f} seconds"

    rounded = int(round(seconds))
    days, remainder = divmod(rounded, 86_400)
    hours, remainder = divmod(remainder, 3_600)
    minutes, secs = divmod(remainder, 60)
    parts = []
    if days:
        parts.append(f"{days} day{'s' if days != 1 else ''}")
    if hours:
        parts.append(f"{hours} hour{'s' if hours != 1 else ''}")
    if minutes:
        parts.append(f"{minutes} minute{'s' if minutes != 1 else ''}")
    if secs or not parts:
        parts.append(f"{secs} second{'s' if secs != 1 else ''}")
    return ", ".join(parts)


def mda_schedule_window(starting_time, delay_seconds, duration_seconds):
    """Return the delayed MDA start and expected completion timestamps."""
    delay_seconds = float(delay_seconds)
    duration_seconds = float(duration_seconds)
    if delay_seconds < 0:
        raise ValueError("MDA start delay cannot be negative")
    if duration_seconds < 0:
        raise ValueError("MDA duration cannot be negative")

    scheduled_start = starting_time + timedelta(seconds=delay_seconds)
    expected_end = scheduled_start + timedelta(seconds=duration_seconds)
    return scheduled_start, expected_end


def estimate_live_mda_end(
    completion_intervals,
    completed,
    total,
    current_time,
    window_size=10,
):
    """Predict completion from the latest acquisition completion intervals."""
    if window_size < 1:
        raise ValueError("Live ETA window size must be at least 1")
    recent = tuple(float(value) for value in completion_intervals)[
        -window_size:
    ]
    if not recent:
        return None
    if any(value < 0 for value in recent):
        raise ValueError("Acquisition intervals cannot be negative")

    completed = max(0, int(completed))
    total = max(completed, int(total))
    average_seconds = sum(recent) / len(recent)
    remaining = total - completed
    expected_end = current_time + timedelta(
        seconds=average_seconds * remaining
    )
    return LiveMdaEstimate(
        expected_end=expected_end,
        average_seconds=average_seconds,
        sample_count=len(recent),
        completed=completed,
        total=total,
    )
