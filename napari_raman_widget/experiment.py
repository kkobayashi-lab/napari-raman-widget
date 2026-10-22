"""Serializable acquisition extensions and detached execution snapshots."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
import math
import uuid

import numpy as np
from useq import MDAEvent

METADATA_KEY = "confocal_control"


@dataclass
class RamanSettings:
    planes: str = "all"
    camera_positions: str = "all"
    batch: bool = False
    pre_acq: bool = True
    repeats: int = 1


@dataclass
class AutofocusSettings:
    mode: str = "off"
    positions: str = "all"
    every: int = 0
    search_range: float = 6.0
    search_points: int = 8
    fine_range: float = 1.5
    fine_points: int = 8
    axes: tuple[str, ...] = ("p", "t")


@dataclass
class TrackingSettings:
    enabled: bool = False
    every: int = 0
    channel: str = "BF"
    scale: float = 2.0
    model: str = "cyto2"
    diameter: float = 15.0
    crop: bool = True
    config: str = "particle_config.json"


@dataclass
class ExperimentExtensions:
    raman: RamanSettings = field(default_factory=RamanSettings)
    autofocus: AutofocusSettings = field(default_factory=AutofocusSettings)
    tracking: TrackingSettings = field(default_factory=TrackingSettings)
    delay: float = 0.0
    overwrite: bool = False

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        return cls(
            raman=RamanSettings(**value.get("raman", {})),
            autofocus=AutofocusSettings(**value.get("autofocus", {})),
            tracking=TrackingSettings(**value.get("tracking", {})),
            delay=value.get("delay", 0.0), overwrite=value.get("overwrite", False),
        )


def indices(spec, count, *, middle=False):
    text = str(spec).strip().lower()
    if text in ("all", ""):
        return range(count)
    if text in ("none", "off"):
        return []
    if middle and text == "middle":
        return [count // 2]
    try:
        result = sorted(set(int(x.strip()) for x in text.split(",")))
    except ValueError as error:
        raise ValueError("Use all, none, or comma-separated position/plane indices") from error
    if any(x < 0 or x >= count for x in result):
        raise ValueError(f"Indices must be between 0 and {count - 1}")
    return result


def identify_positions(sequence):
    """Keep stable identities in position metadata, including duplicate XY positions."""
    from useq import MDASequence
    positions = []
    for position in sequence.stage_positions:
        sub = position.sequence or MDASequence()
        metadata = {**sub.metadata, "position_identity": position_id(position) or uuid.uuid4().hex}
        positions.append(position.replace(sequence=sub.replace(metadata=metadata)))
    return sequence.replace(stage_positions=positions)


def position_id(position):
    if position.sequence and position.sequence.metadata.get("position_identity"):
        return position.sequence.metadata["position_identity"]
    if " [id:" not in (position.name or ""):
        return None
    return position.name.rsplit(" [id:", 1)[1].rstrip("]")


class SnapshotSource:
    """Detached target points; tracking mutates this run, never the editable layers."""
    def __init__(self, record):
        self.record = deepcopy(record)
        self.name = record["name"]
        self.role = record["role"]
        self.source_id = record["id"]
        self.display_name = record.get("display_name", self.name)
        self.color = record.get("color")
        self._points_by_position = {
            int(p): np.asarray(points, dtype=float).reshape((-1, 2)).copy()
            for p, points in record["points"].items()
        }
        self._anchors = {
            int(p): np.asarray(points, dtype=float).reshape((-1, 2)).copy()
            for p, points in record.get("anchors", record["points"]).items()
        }
        self._img_shape = tuple(record["image_shape"])
        self.fixed = bool(record.get("fixed"))
        self._fixed_points = self._points_by_position.get(0, np.empty((0, 2))).copy()
        self._fixed_anchors = self._anchors.get(0, np.empty((0, 2))).copy()

    def get_mda_points(self, event, transform=True):
        table = self._points_by_position if transform else self._anchors
        default = (self._fixed_points if transform else self._fixed_anchors) if self.fixed else np.empty((0, 2))
        return table.get(event.index.get("p", 0), default).copy()

    def set_mda_points(self, position, points):
        anchors = np.asarray(points, dtype=float) / self._img_shape
        old = self._anchors.get(position, self._fixed_anchors if self.fixed else anchors)
        expanded = self._points_by_position.get(position, self._fixed_points if self.fixed else anchors)
        if len(old) and len(expanded) % len(old) == 0:
            delta = np.repeat(anchors - old, len(expanded) // len(old), axis=0)
            self._points_by_position[position] = expanded + delta
        else:
            self._points_by_position[position] = anchors.copy()
        self._anchors[position] = anchors.copy()


def capture_targets(sequence, sources):
    records = []
    for source in sources:
        record = {
            "id": str(getattr(source, "source_id", source.name)),
            "name": source.name,
            "role": getattr(source, "role", None) or (
                "autofocus" if "autofocus" in source.name.lower() else "cell"
            ),
            "display_name": getattr(source, "display_name", source.name),
            "color": getattr(source, "color", None),
            "image_shape": list(getattr(source, "_img_shape", (1024, 1024))),
            "points": {}, "anchors": {}, "position_ids": [],
        }
        for p, position in enumerate(sequence.stage_positions):
            event = MDAEvent(index={"p": p, "t": 0})
            for key, transform in (("points", True), ("anchors", False)):
                try:
                    points = source.get_mda_points(event, transform=transform)
                except TypeError:
                    points = source.get_mda_points(event)
                points = np.asarray(points, dtype=float).reshape((-1, 2))
                if not np.isfinite(points).all():
                    raise ValueError(f"Target {source.name} has non-finite coordinates")
                record[key][str(p)] = points.tolist()
            record["position_ids"].append(position_id(position))
        records.append(record)
    return records


def remap_targets(sequence, records):
    ids = [position_id(p) for p in sequence.stage_positions]
    if records and (None in ids or len(set(ids)) != len(ids)):
        raise ValueError("Prepared position identities changed; apply targets again")
    result = []
    for record in records:
        old_ids = record["position_ids"]
        if any(identity not in ids for identity in old_ids):
            raise ValueError("A prepared position was deleted; apply targets again")
        mapped = deepcopy(record)
        for key in ("points", "anchors"):
            mapped[key] = {
                str(ids.index(old_ids[int(p)])): points
                for p, points in record[key].items()
            }
        result.append(SnapshotSource(mapped))
    return result


@dataclass(frozen=True)
class PreparedAcquisition:
    sequence: object
    extensions: ExperimentExtensions
    sources: tuple
    raman_enabled: bool
    raman_exposure: float
    raman_planes: tuple
    image_positions: tuple
    autofocus_positions: tuple


def prepare_acquisition(sequence, extensions, targets, image_shape=None):
    """Validate and detach a run without touching hardware or creating output."""
    from .raman_selection import restore_position_targets
    sequence = restore_position_targets(sequence)
    from .autofocus_references import reference_records, REFERENCE_KEY
    if any(p.sequence and REFERENCE_KEY in p.sequence.metadata for p in sequence.stage_positions):
        targets = [r for r in targets if r.get("role") != "autofocus"] + reference_records(sequence)
    ext = deepcopy(extensions)
    from raman_mda_engine.grid_plan import GRID_KEY, grid_count
    compact_grid = GRID_KEY in sequence.metadata
    if compact_grid and sequence.metadata.get(METADATA_KEY, {}).get("grid_mode") == "raman":
        ext.raman.batch = False
        ext.raman.pre_acq = False
    if compact_grid and (sequence.stage_positions or sequence.grid_plan):
        raise ValueError("Detach the Raman grid before using native stage positions or grids")
    if compact_grid and any(not record.get("fixed") for record in targets):
        raise ValueError("Raman grids require shared fixed targets")
    if not math.isfinite(ext.delay) or ext.delay < 0:
        raise ValueError("Start delay must be finite and non-negative")
    channels = [channel for channel in sequence.channels if channel.config == "RM"]
    if len(channels) > 1:
        raise ValueError("Only one RM channel is supported")
    enabled = bool(channels)
    exposure = float(channels[0].exposure or 0) if enabled else 0.0
    if enabled and (not math.isfinite(exposure) or exposure < 73.8):
        raise ValueError("RM exposure must be at least 73.8 ms per collection")
    # Hardware autofocus is not configured on this rig. Never dispatch native
    # autofocus actions, including plans inherited from imported settings.
    sequence = sequence.replace(autofocus_plan=None, stage_positions=[
        p.replace(sequence=p.sequence.replace(autofocus_plan=None)) if p.sequence else p
        for p in sequence.stage_positions
    ])
    if not ext.autofocus.axes or ext.autofocus.mode == "hardware":
        ext.autofocus.mode = "off"
    if any(axis not in ("p", "t", "g") for axis in ext.autofocus.axes):
        raise ValueError("Autofocus axes must be p, t, or g")
    if ext.raman.repeats < 1 or ext.autofocus.every < 0 or ext.tracking.every < 0:
        raise ValueError("Repeats must be positive and update cadences non-negative")
    custom_af = ext.autofocus.mode not in ("off", "hardware")
    if custom_af and sequence.z_plan and not sequence.z_plan.is_relative:
        raise ValueError("Custom autofocus requires a relative Z plan")
    # Target tables refer to stage positions, not nested grid/subsequence expansions.
    expanded_subsequences = any(
        p.sequence and p.sequence.model_dump(
            exclude={"uid", "autofocus_plan", "metadata"}, exclude_defaults=True
        ) for p in sequence.stage_positions
    )
    if targets and (sequence.grid_plan or expanded_subsequences):
        raise ValueError("Prepared targets require explicit positions; expand the grid before applying targets")
    from .raman_selection import target, position_records
    policies = None
    position_owned = any(target(p) for p in sequence.stage_positions)
    if enabled and not compact_grid and (position_owned or not any(r.get("role") == "cell" for r in targets)):
        shape = image_shape or next((target(p)["image_shape"] for p in sequence.stage_positions if target(p)), (1024, 1024))
        record, policies = position_records(sequence, shape)
        focus = [r for r in targets if r.get("role") == "autofocus"]
        sources = (*remap_targets(sequence, focus), SnapshotSource(record))
    else:
        sources = tuple(SnapshotSource(record) for record in targets) if compact_grid else tuple(remap_targets(sequence, targets))
    if enabled and not any(s.role == "cell" and any(len(x) for x in s._points_by_position.values()) for s in sources):
        raise ValueError("RM acquisition needs prepared Raman targets")
    if policies is not None and ext.tracking.enabled:
        raise ValueError("Disable tracking for position-owned Raman targets; reselect positions from updated images instead")
    if custom_af and compact_grid and not any(
        s.role == "autofocus" and len(s._fixed_anchors) for s in sources
    ):
        raise ValueError("Dense grid autofocus needs a fixed autofocus target")
    if ext.tracking.enabled and not any(s.role == "cell" for s in sources):
        raise ValueError("Tracking needs prepared cell targets")
    count = max(1, grid_count(sequence))
    nz = sequence.z_plan.num_positions() if sequence.z_plan else 1
    planes = list(indices(ext.raman.planes, nz, middle=True)) if enabled else []
    if enabled and not planes:
        raise ValueError("RM is selected but no Raman planes are selected")
    image_p = indices(ext.raman.camera_positions, count)
    af_p = indices(ext.autofocus.positions, count) if custom_af else []
    if custom_af and not compact_grid:
        af_p = [p for p in af_p if any(
            s.role == "autofocus" and len(s.get_mda_points(MDAEvent(index={"p": p}), transform=False))
            for s in sources)]
        if not af_p:
            ext.autofocus.mode = "off"
    metadata = deepcopy(sequence.metadata)
    from .autofocus_references import reference
    metadata["autofocus_stage_xy"] = {
        str(p): reference(pos)["stage_xy_um"] for p, pos in enumerate(sequence.stage_positions)
        if reference(pos) and "stage_xy_um" in reference(pos)
    }
    metadata["autofocus_position_references"] = {
        str(p): deepcopy(reference(pos)) for p, pos in enumerate(sequence.stage_positions) if reference(pos)
    }
    for xy in metadata["autofocus_stage_xy"].values():
        if len(xy) != 2 or not all(math.isfinite(v) for v in xy):
            raise ValueError("Autofocus reference needs finite stage XY coordinates")
    if policies is not None:
        metadata["raman_positions"] = policies
    else:
        metadata.pop("raman_positions", None)
    metadata[METADATA_KEY] = {
        **metadata.get(METADATA_KEY, {}),
        "schema_version": 1, "extensions": ext.to_dict(), "targets": deepcopy(targets),
    }
    if enabled:
        metadata["raman"] = {
            **metadata.get("raman", {}), "channel": "RM", "z": planes,
            "frames_per_point": ext.raman.repeats,
            "cell_layers": [{"id": s.source_id, "name": s.display_name,
                             "source_name": s.name, "color": s.color}
                            for s in sources if s.role == "cell"],
        }
    else:
        metadata.pop("raman", None)
    # Native useq channel offsets are authoritative; migrate legacy values only on import.
    metadata.pop("channel_z_offsets_um", None)
    # A metadata-only sub-sequence changes useq channel traversal. Strip it only
    # from the detached execution copy, retaining identity for acquired images.
    sequence = execution_positions(sequence)
    return PreparedAcquisition(
        sequence.replace(metadata=metadata), ext, sources, enabled, exposure,
        tuple(planes), image_p if isinstance(image_p, range) else tuple(image_p),
        af_p if isinstance(af_p, range) else tuple(af_p),
    )


def execution_positions(sequence):
    positions = []
    for pos in sequence.stage_positions:
        if pos.sequence and not pos.sequence.model_dump(
                exclude={"uid", "metadata", "autofocus_plan"}, exclude_defaults=True):
            identity = position_id(pos)
            name = (pos.name or "Position").split(" [id:")[0]
            pos = pos.replace(sequence=None, name=f"{name} [id:{identity}]" if identity else name)
        positions.append(pos)
    return sequence.replace(stage_positions=positions)
