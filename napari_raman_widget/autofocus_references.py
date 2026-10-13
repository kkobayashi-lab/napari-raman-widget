"""Select autofocus references without creating or changing Raman cell targets."""
from copy import deepcopy

import numpy as np
from useq import MDASequence, Position

from .experiment import METADATA_KEY, capture_targets, identify_positions, position_id


def preserve_references(old_sequence, new_sequence, records, source_positions=None):
    """Carry references to unchanged FOVs, including repeated batch positions.

    Shifted, cell-centered FOVs need a new reference; pixel coordinates from
    their parent image must not silently be used in a different FOV.
    """
    old_positions = list(old_sequence.stage_positions)
    result = []
    for record in records:
        if record.get("role") != "autofocus":
            continue
        if record.get("fixed"):
            continue  # Grid anchors belong to that grid, not a different draft.
        mapped = deepcopy(record)
        mapped.update(position_ids=[], points={}, anchors={})
        for p, destination_position in enumerate(new_sequence.stage_positions):
            position = source_positions[p] if source_positions is not None else destination_position
            matches = [i for i, old in enumerate(old_positions)
                       if (old.x, old.y) == (position.x, position.y)
                       and position_id(old) == position_id(position)]
            if not matches:
                matches = [i for i, old in enumerate(old_positions)
                           if (old.x, old.y) == (position.x, position.y)
                           and (old.name or "").split(" [id:")[0] == (position.name or "").split(" [id:")[0]]
            if len(matches) != 1:
                continue
            identity = position_id(old_positions[matches[0]])
            if identity not in record["position_ids"]:
                continue
            source_p = str(record["position_ids"].index(identity))
            destination = str(len(mapped["position_ids"]))
            mapped["position_ids"].append(position_id(destination_position))
            for key in ("points", "anchors"):
                mapped[key][destination] = deepcopy(record.get(key, record["points"]).get(source_p, []))
        if mapped["position_ids"]:
            result.append(mapped)
    return result


def _require_idle(owner):
    if owner._unified_mda is None:
        raise ValueError("Connect the microscope first")
    if owner._acquisition_controller.busy or owner.core.mda.is_running():
        raise ValueError("Stop the acquisition before editing autofocus references")


def choose_references(owner):
    """Acquire a camera preview at the existing positions and open an AF-only layer."""
    _require_idle(owner)
    from cns_control.utils import create_point_sources
    from raman_mda_engine.aiming.transformers import Identity
    from .experiment import SnapshotSource

    panel = owner._unified_mda
    sequence = panel.value()
    if "raman_grid" in sequence.metadata or sequence.grid_plan:
        raise ValueError("Grid autofocus uses its existing grid anchor; select explicit FOVs for custom references")
    sequence = identify_positions(sequence)
    if not sequence.stage_positions:
        sequence = identify_positions(sequence.replace(stage_positions=[Position(
            x=owner.core.getXPosition(), y=owner.core.getYPosition(), z=owner.core.getZPosition())]))
    panel.setValue(sequence)
    channels = [c for c in sequence.channels if c.config != "RM"]
    channel = channels[0] if channels else {"config": "BF", "exposure": 100}
    preview = MDASequence(stage_positions=[p.replace(sequence=None) for p in sequence.stage_positions],
                          channels=[channel], axis_order="tpcz")
    previous_engine = owner.core.mda.engine
    group = (channel.group or "Channel") if channels else "Channel"
    preset = owner.core.getCurrentConfig(group)
    auto_shutter = owner.core.getAutoShutter()
    shutter_open = owner.core.getShutterOpen()
    try:
        owner._prepare_for_selection()
        owner.core.register_mda_engine(owner.default_engine)
        owner.core.run_mda(preview)
    finally:
        owner.core.register_mda_engine(previous_engine)
        if preset:
            owner.core.setConfig(group, preset)
            owner.core.waitForConfig(group, preset)
        owner.core.setAutoShutter(auto_shutter)
        owner.core.setShutterOpen(shutter_open)
    sources = create_point_sources(owner.viewer, Identity(), names=["autofocus references"],
                                   colors=["springgreen"], roles=["autofocus"])
    source = sources[0]
    source._img_shape = (owner.core.getImageWidth(), owner.core.getImageHeight())
    references = preserve_references(sequence, sequence, panel.targets)
    for record in references:
        snapshot = SnapshotSource(record)
        for i, identity in enumerate(record["position_ids"]):
            p = next(j for j, pos in enumerate(sequence.stage_positions) if position_id(pos) == identity)
            points = snapshot._anchors.get(i, np.empty((0, 2))) * source._img_shape
            for y, x in points:
                source._points.add([0, p, 0, 0, y, x])
    owner._autofocus_reference_selection = (sequence, sources)
    owner.viewer.layers.selection.active = source._points
    source._points.mode = "add"
    owner.status.setText("Status: click one autofocus reference per FOV, then Apply autofocus references")


def apply_references(owner):
    _require_idle(owner)
    selection = getattr(owner, "_autofocus_reference_selection", None)
    if selection is None:
        raise ValueError("Choose autofocus reference points first")
    sequence, sources = selection
    panel = owner._unified_mda
    current = panel.value()
    records = capture_targets(sequence, sources)
    for record in records:
        if any(len(points) > 1 for points in record["anchors"].values()):
            raise ValueError("Keep one autofocus reference per FOV; remove extra points before applying")
    if {(position_id(p), p.x, p.y) for p in sequence.stage_positions} != {
            (position_id(p), p.x, p.y) for p in current.stage_positions}:
        raise ValueError("MDA positions changed; choose autofocus references again")
    metadata = deepcopy(current.metadata)
    metadata[METADATA_KEY]["targets"] = [r for r in panel.targets if r.get("role") != "autofocus"] + records
    panel.setValue(current.replace(metadata=metadata))
    panel.update_summary()
    owner.status.setText("Status: autofocus references applied; camera and Raman settings preserved")


def reference_action(owner, action):
    try:
        action(owner)
    except Exception as error:
        owner.status.setText(f"Status: autofocus references — {error}")
