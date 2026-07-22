"""LLM-backed chat panel for the CNS Raman HardwareWidget.

The assistant does exactly one thing: it maps a plain-English message to one
of the widget's existing GUI actions (the same methods the buttons call),
optionally filling a few fields first. It never touches hardware directly --
it drives the GUI, so it reuses every existing range check and validation.

Design
------
* ACTIONS is a small registry. Each entry names a widget method, the fields it
  may set (attribute + kind), and whether it is read-only (safe) or moves
  hardware (gated by a confirm dialog).
* Tool schemas for the Anthropic API are generated from ACTIONS, so adding a
  new capability means adding one registry entry -- no schema by hand.
* The API call runs on a worker thread (never blocks napari). When the model
  asks to run a tool, execution is marshaled back to the Qt main thread via a
  BlockingQueuedConnection signal, because Qt and MMCore are not thread-safe.

Requirements
------------
    pip install anthropic
    set ANTHROPIC_API_KEY in the environment before launching napari.

Integration (in widget.py, near the end of __init__, before the scroll wrap):
    from .chat_panel import ChatPanel
    self.chat_panel = ChatPanel(self)
    outer.addWidget(self.chat_panel)
"""

import threading

import numpy as np
from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import (
    QLabel, QLineEdit, QMessageBox, QPushButton, QTextEdit, QVBoxLayout,
    QWidget,
)

# Model to use. Change this to whatever your Anthropic account can access.
MODEL = "claude-sonnet-4-5"
MAX_TOKENS = 1024

# A relative stage move larger than this (in um, any axis) is refused outright,
# as a guard against a hallucinated or fat-fingered value crashing the
# objective. Raise it only if you know your travel is safe.
MAX_STAGE_STEP_UM = 500.0

# Field "kinds" tell the executor how to set a widget value.
#   text  -> QLineEdit.setText(str)
#   int   -> QSpinBox.setValue(int)
#   float -> QDoubleSpinBox.setValue(float)
#   combo -> QComboBox.setCurrentText(str)
#   check -> QCheckBox.setChecked(bool)

_AF_OBJECTS = ["None", "laser", "software", "quartz", "glass", "cell"]
_BATCH = ["False", "True"]


def _p(name, attr, kind, description, enum=None, schema=None):
    """Shorthand for a settable-parameter spec.

    attr/kind drive how a *button* action sets a widget field. For handler
    actions the value is read straight from tool_input, so attr may be None.
    Pass ``schema`` to override the generated JSON schema (e.g. arrays).
    """
    d = {"name": name, "attr": attr, "kind": kind, "description": description}
    if enum is not None:
        d["enum"] = enum
    if schema is not None:
        d["schema"] = schema
    return d


# ---------------------------------------------------------------------------
# Handlers for actions that don't map to an existing widget button:
# napari layers, camera exposure/channel, stage moves, and the MDA widget.
# Each takes (hw, tool_input) and returns a short result string.
# ---------------------------------------------------------------------------

def _h_list_layers(hw, inp):
    names = [ly.name for ly in hw.viewer.layers]
    return "Layers: " + (", ".join(names) if names else "(none)")


def _h_create_points(hw, inp):
    name = str(inp.get("name") or "points")
    pts = inp.get("points")
    data = np.asarray(pts, dtype=float) if pts else np.empty((0, 2))
    hw.viewer.add_points(data, name=name)
    return f"Added points layer '{name}' with {len(data)} point(s)."


def _h_create_shapes(hw, inp):
    name = str(inp.get("name") or "shapes")
    rect = inp.get("rectangle")
    if rect and len(rect) == 4:
        y0, x0, y1, x1 = [float(v) for v in rect]
        corners = np.array([[y0, x0], [y0, x1], [y1, x1], [y1, x0]])
        hw.viewer.add_shapes([corners], shape_type="rectangle", name=name)
        return f"Added shapes layer '{name}' with a rectangle."
    hw.viewer.add_shapes(name=name)
    return f"Added empty shapes layer '{name}'."


def _h_list_channels(hw, inp):
    if hw.core is None:
        return "Not connected."
    try:
        chans = list(hw.core.getAvailableConfigs("Channel"))
    except Exception as e:
        return f"Couldn't read channels: {e}"
    return "Channels: " + (", ".join(chans) if chans else "(none)")


def _h_set_exposure(hw, inp):
    if hw.core is None:
        return "Not connected."
    ms = float(inp["exposure_ms"])
    hw.core.setExposure(ms)
    return f"Camera exposure set to {ms:g} ms."


def _h_set_channel(hw, inp):
    if hw.core is None:
        return "Not connected."
    ch = str(inp["channel"])
    try:
        avail = list(hw.core.getAvailableConfigs("Channel"))
    except Exception:
        avail = []
    if avail and ch not in avail:
        return f"Channel '{ch}' not available. Options: {', '.join(avail)}"
    hw.core.setConfig("Channel", ch)
    return f"Channel set to '{ch}'."


def _h_snap(hw, inp):
    if hw.core is None:
        return "Not connected."
    try:
        # a live/sequence acquisition blocks a single snap -- stop it first
        try:
            hw.core.stopSequenceAcquisition()
        except Exception:
            pass
        img = hw.core.snap()
    except Exception as e:
        return f"Snap failed: {e}"
    try:
        hw.viewer.add_image(np.asarray(img), name="snap")
    except Exception:
        pass  # napari-micromanager may already be showing the preview
    return "Snapped an image."


def _h_start_live(hw, inp):
    if hw.core is None:
        return "Not connected."
    try:
        exp = inp.get("exposure_ms")
        if exp is not None:
            hw.core.setExposure(float(exp))
        hw.core.startContinuousSequenceAcquisition(0)
    except Exception as e:
        return f"Couldn't start live: {e}"
    return "Live (continuous) acquisition started."


def _h_stop_live(hw, inp):
    if hw.core is None:
        return "Not connected."
    try:
        hw.core.stopSequenceAcquisition()
    except Exception as e:
        return f"Couldn't stop live: {e}"
    return "Live/sequence acquisition stopped."


def _h_get_image_size(hw, inp):
    x_size, y_size = hw._get_image_xy()
    return (
        f"Image width={x_size}, height={y_size}. "
        f"Center pixel (x,y)=({x_size // 2},{y_size // 2}); "
        f"in (y,x) order that is ({y_size // 2},{x_size // 2})."
    )


def _h_center_on_pixel(hw, inp):
    if hw.core is None:
        return "Not connected."
    x_size, y_size = hw._get_image_xy()
    x = inp.get("x")
    y = inp.get("y")
    x = x_size / 2.0 if x is None else float(x)
    y = y_size / 2.0 if y is None else float(y)
    try:
        hw._move_clicked_to_center(np.array([y, x], dtype=float))
    except Exception as e:
        return f"Centering failed: {e}"
    try:
        st = hw.status.text()
    except Exception:
        st = ""
    return f"Moved so pixel (y={y:.0f}, x={x:.0f}) -> mask center. {st}"


def _reveal_dock(dw):
    """Show, un-hide and raise a (possibly tabified/hidden) dock widget."""
    try:
        dw.setVisible(True)
    except Exception:
        pass
    try:
        dw.show()
    except Exception:
        pass
    try:
        dw.raise_()
    except Exception:
        pass


def _h_open_mm_widget(hw, inp):
    """Reveal a napari-micromanager sub-dock (MDA, Stages, ...).

    In current napari-micromanager everything lives inside one 'Main Window';
    the individual tools are registered in main_window._dock_widgets. We
    reveal the matching one directly rather than trying to open a separate
    plugin widget.
    """
    which = str(inp.get("widget") or "MDA").strip()
    mw = hw.main_window

    # 1) reveal a registered sub-dock by fuzzy name match
    dws = getattr(mw, "_dock_widgets", None) if mw is not None else None
    if isinstance(dws, dict) and dws:
        key = None
        for k in dws:
            if which.lower() == k.lower():
                key = k
                break
        if key is None:
            for k in dws:
                if which.lower() in k.lower():
                    key = k
                    break
        if key is not None:
            _reveal_dock(dws[key])
            return f"Revealed napari-micromanager '{key}'."
        return (f"No sub-dock matching '{which}'. "
                f"Available: {', '.join(dws.keys())}.")

    # 2) fall back to opening a top-level plugin widget by name
    try:
        hw.viewer.window.add_plugin_dock_widget(
            "napari-micromanager", widget_name=which
        )
        return f"Opened napari-micromanager '{which}'."
    except Exception as e:
        return (f"Couldn't open '{which}': {e}. "
                "Use inspect_mm to list the real widget names.")


def _h_inspect_mm(hw, inp):
    """Report how the napari-micromanager main window exposes its tools, so
    the exact dock names / methods can be wired precisely."""
    mw = hw.main_window
    if mw is None:
        return ("napari-micromanager main window not captured -- "
                "reconnect hardware first.")
    parts = [f"main_window type: {type(mw).__name__}"]
    dws = getattr(mw, "_dock_widgets", None)
    if isinstance(dws, dict):
        parts.append("sub-docks: " + (", ".join(dws.keys()) or "(empty)"))
    else:
        parts.append("no _dock_widgets dict")
    meths = [
        m for m in dir(mw)
        if any(k in m.lower() for k in ("dock", "widget", "show", "tool"))
        and not m.startswith("__")
    ]
    parts.append("methods: " + (", ".join(meths[:25]) or "(none)"))
    return " | ".join(parts)


def _h_get_stage(hw, inp):
    if hw.core is None:
        return "Not connected."
    try:
        x, y = hw.core.getXYPosition()
        z = hw.core.getPosition()
        return f"Stage X={x:.2f} Y={y:.2f} Z={z:.2f} um"
    except Exception as e:
        return f"Couldn't read stage: {e}"


def _h_move_stage_relative(hw, inp):
    if hw.core is None:
        return "Not connected."
    dx = float(inp.get("dx", 0.0))
    dy = float(inp.get("dy", 0.0))
    dz = float(inp.get("dz", 0.0))
    for v, nm in ((dx, "dx"), (dy, "dy"), (dz, "dz")):
        if abs(v) > MAX_STAGE_STEP_UM:
            return (f"Refused: {nm}={v} um exceeds the "
                    f"{MAX_STAGE_STEP_UM:g} um safety limit.")
    x, y = hw.core.getXYPosition()
    hw.core.setXYPosition(x + dx, y + dy)
    if dz:
        hw.core.setPosition(hw.core.getPosition() + dz)
    hw.core.waitForSystem()
    nx, ny = hw.core.getXYPosition()
    nz = hw.core.getPosition()
    return f"Moved to X={nx:.2f} Y={ny:.2f} Z={nz:.2f} um."


def _mda_settings(hw):
    """Locate the napari-micromanager MDA settings object (the one exposing
    value()/setValue() for a useq MDASequence)."""
    mw = hw.main_window
    if mw is None:
        return None
    try:
        dock = mw._dock_widgets["MDA"]
    except Exception:
        return None
    # the known layout index first, then a defensive search
    try:
        cand = dock.children()[4]
        if hasattr(cand, "value") and hasattr(cand, "setValue"):
            return cand
    except Exception:
        pass
    try:
        for c in dock.findChildren(QWidget):
            if hasattr(c, "value") and hasattr(c, "setValue"):
                try:
                    v = c.value()
                except Exception:
                    continue
                if hasattr(v, "replace") and hasattr(v, "stage_positions"):
                    return c
    except Exception:
        pass
    return None


def _h_start_mda(hw, inp):
    if hw.core is None:
        return "Not connected."
    settings = _mda_settings(hw)
    if settings is None:
        return ("Couldn't find the MDA widget. Open it first ('open MDA') "
                "and make sure hardware is connected.")
    try:
        seq = settings.value()
    except Exception as e:
        return f"Couldn't read the MDA sequence: {e}"
    hw.core.run_mda(seq)
    return ("Started the standard napari-micromanager MDA (not the Raman "
            "engine). Use 'stop MDA' to cancel.")


def _h_build_mda(hw, inp):
    """Build a useq MDASequence from simple parameters and load it into the
    napari-micromanager MDA widget (does not run it)."""
    import datetime as dt
    try:
        from useq import MDASequence
    except Exception as e:
        return f"useq not importable: {e}"
    settings = _mda_settings(hw)
    if settings is None:
        return ("Couldn't find the MDA widget. Open it first ('open MDA') "
                "and reconnect if needed.")

    kwargs = {"axis_order": "tpcz"}

    channels = inp.get("channels")
    if channels:
        exp = (float(inp["exposure_ms"])
               if inp.get("exposure_ms") is not None else 100.0)
        kwargs["channels"] = [
            {"config": str(c), "exposure": exp} for c in channels
        ]

    loops = inp.get("loops")
    if loops:
        interval = float(inp.get("interval_s") or 0.0)
        kwargs["time_plan"] = {
            "interval": dt.timedelta(seconds=interval),
            "loops": int(loops),
        }

    z_range = inp.get("z_range_um")
    if z_range is not None:
        z_step = float(inp.get("z_step_um") or 1.0)
        kwargs["z_plan"] = {"range": float(z_range), "step": z_step}

    pos_list = []
    if inp.get("add_current_position") and hw.core is not None:
        try:
            x, y = hw.core.getXYPosition()
            z = hw.core.getPosition()
            pos_list.append({"x": x, "y": y, "z": z})
        except Exception:
            pass
    for p in (inp.get("positions") or []):
        try:
            if len(p) >= 3:
                pos_list.append({"x": float(p[0]), "y": float(p[1]),
                                 "z": float(p[2])})
            elif len(p) == 2:
                pos_list.append({"x": float(p[0]), "y": float(p[1])})
        except Exception:
            continue
    if pos_list:
        kwargs["stage_positions"] = pos_list

    try:
        seq = MDASequence(**kwargs)
    except Exception as e:
        return f"Couldn't build the sequence: {e}"
    try:
        settings.setValue(seq)
    except Exception as e:
        return f"Built it but couldn't load into the MDA widget: {e}"
    return (
        f"MDA sequence loaded: channels={channels or 'unchanged'}, "
        f"loops={loops or 1}, interval={inp.get('interval_s') or 0}s, "
        f"z_range={z_range}, positions={len(pos_list)}. "
        "Use 'start MDA' to run it."
    )


def _h_add_current_position(hw, inp):
    """Append the current stage position to the MDA widget's sequence."""
    if hw.core is None:
        return "Not connected."
    settings = _mda_settings(hw)
    if settings is None:
        return "Couldn't find the MDA widget. Open it first ('open MDA')."
    try:
        x, y = hw.core.getXYPosition()
        z = hw.core.getPosition()
    except Exception as e:
        return f"Couldn't read the stage: {e}"
    try:
        seq = settings.value()
        pos = list(seq.stage_positions) + [{"x": x, "y": y, "z": z}]
        settings.setValue(seq.replace(stage_positions=pos))
    except Exception as e:
        return f"Couldn't add the position: {e}"
    return (f"Added position X={x:.1f} Y={y:.1f} Z={z:.1f}. "
            f"Total positions: {len(pos)}.")


# The registry. Order here is the order the model sees them.
ACTIONS = [
    # ---- read-only queries (run immediately, never gated) ----
    {
        "name": "get_state",
        "label": "Read current state",
        "readonly": True,
        "method": None,          # handled specially
        "params": [],
        "description": (
            "Report connection status, the status-bar text, the current "
            "center wavelength label and the selected grating. Use this when "
            "unsure of the rig state before proposing an action."
        ),
    },

    # ---- loading ----
    {
        "name": "connect_hardware",
        "label": "Connect hardware",
        "method": "connect",
        "params": [],
        "description": "Connect the rig: load config, devices, models.",
    },
    {
        "name": "disconnect_hardware",
        "label": "Disconnect hardware",
        "method": "disconnect",
        "params": [],
        "description": "Unload all devices and clear session state.",
    },
    {
        "name": "set_wavelength",
        "label": "Update center wavelength",
        "method": "update_wavelength",
        "params": [_p("wavelength", "wl_input", "float",
                      "Center wavelength in nm (0-2000).")],
        "description": "Move the spectrometer to a center wavelength.",
    },
    {
        "name": "set_grating",
        "label": "Update grating",
        "method": "update_grating",
        "params": [_p("grating", "grating_combo", "combo",
                      "Grating number as a string, e.g. '1'.")],
        "description": "Rotate the turret to the given grating.",
    },

    # ---- collect spectra ----
    {
        "name": "collect_spectra",
        "label": "Collect spectra at last point",
        "method": "collect_raman",
        "params": [
            _p("exposure_ms", "exposure_input", "float", "Exposure in ms."),
            _p("repeats", "n_input", "int", "Repeat spectra (>=2)."),
            _p("save_as", "collect_save_input", "text",
               "Filename without extension; blank = don't save."),
        ],
        "description": (
            "Aim at the last point in the top layer and collect spectra."
        ),
    },

    # ---- calibration ----
    {
        "name": "run_calibration",
        "label": "Run laser aiming calibration",
        "method": "run_calibration",
        "params": [
            _p("n", "cal_n_input", "int", "Repeats per target."),
            _p("exposure_ms", "cal_exp_input", "float", "Exposure in ms."),
            _p("max_volts", "cal_volts_input", "float", "Max galvo volts."),
            _p("grid_size", "cal_grid_input", "int", "Calibration grid side."),
            _p("threshold", "cal_thres_input", "float", "Detection threshold."),
        ],
        "description": "Sweep the laser grid and fit a new transformer.",
    },

    # ---- axial background scan ----
    {
        "name": "collect_reference",
        "label": "Collect reference spectra",
        "method": "collect_reference",
        "params": [
            _p("name", "ref_name_input", "text", "Output name prefix."),
            _p("exposure_ms", "ref_exp_input", "float", "Exposure in ms."),
            _p("n_per_z", "ref_n_input", "int", "Spectra per z-plane."),
            _p("search_range", "ref_range_input", "float",
               "Axial half-range in um."),
            _p("search_pts", "ref_pts_input", "int", "Number of z-samples."),
        ],
        "description": "Run an axial background/autofocus scan at the point.",
    },

    # ---- spatial mapping ----
    {
        "name": "run_grid_scan",
        "label": "Run spatial map (grid scan)",
        "method": "run_grid_scan",
        "params": [
            _p("file_name", "scan_name_input", "text", "Output label."),
            _p("exposure_ms", "scan_exp_input", "float", "Raman exposure ms."),
            _p("grid_side", "scan_n_input", "int", "N x N grid side."),
            _p("z_offset", "scan_z_input", "float", "Z offset in um."),
        ],
        "description": (
            "Raman-map the rectangle in the last Shapes layer."
        ),
    },

    # ---- generate stage grid ----
    {
        "name": "generate_stage_grid",
        "label": "Generate stage grid",
        "method": "run_grid_selection",
        "params": [
            _p("autofocus_object", "grid_af_combo", "combo",
               "Autofocus mode.", enum=_AF_OBJECTS),
            _p("x_range", "grid_xrange_input", "float", "X half-range um."),
            _p("y_range", "grid_yrange_input", "float", "Y half-range um."),
            _p("x_step", "grid_xstep_input", "float", "X spacing um."),
            _p("y_step", "grid_ystep_input", "float", "Y spacing um."),
            _p("repeats", "grid_repeats_input", "int", "Points/position (>=2)."),
        ],
        "description": "Build a stage-position grid around the current XY.",
    },

    # ---- cell selection ----
    {
        "name": "add_mask",
        "label": "Add mask overlay",
        "readonly": True,        # viewer overlay only; safe
        "method": "add_mask",
        "params": [
            _p("center_y", "sel_cy_input", "int", "Mask center Y (px)."),
            _p("center_x", "sel_cx_input", "int", "Mask center X (px)."),
            _p("radius", "sel_r_input", "int", "Mask radius (px)."),
        ],
        "description": "Show the circular selection mask in the viewer.",
    },
    {
        "name": "run_automated_selection",
        "label": "Run automated cell selection",
        "method": "run_automated_selection",
        "params": [
            _p("center_y", "sel_cy_input", "int", "Mask center Y (px)."),
            _p("center_x", "sel_cx_input", "int", "Mask center X (px)."),
            _p("radius", "sel_r_input", "int", "Mask radius (px)."),
            _p("autofocus_object", "sel_af_combo", "combo",
               "Autofocus mode.", enum=_AF_OBJECTS),
            _p("n_per_fov", "sel_npf_input", "int", "Cells per FOV."),
            _p("batch", "sel_batch_combo", "combo",
               "Batch collection.", enum=_BATCH),
        ],
        "description": "Segment cells in the mask and prepare the MDA.",
    },
    {
        "name": "run_manual_selection",
        "label": "Set up manual selection",
        "method": "run_manual_selection",
        "params": [
            _p("autofocus_object", "sel_af_combo", "combo",
               "Autofocus mode.", enum=_AF_OBJECTS),
            _p("n_per_fov", "sel_npf_input", "int", "Cells per FOV."),
            _p("batch", "sel_batch_combo", "combo",
               "Batch collection.", enum=_BATCH),
        ],
        "description": "Create empty layers for hand-clicking cells.",
    },
    {
        "name": "center_clicked_cells",
        "label": "Center clicked cells",
        "method": "center_manual_cells",
        "params": [],
        "description": "Turn non-batch clicked cells into centered positions.",
    },

    # ---- run raman MDA ----
    {
        "name": "run_raman_mda",
        "label": "Run Raman MDA",
        "method": "run_raman_mda",
        "params": [
            _p("output_dir", "mda_dir_input", "text", "Writer output dir."),
            _p("exposure_per_cell_ms", "mda_exp_input", "float",
               "Exposure per cell in ms."),
            _p("loops", "mda_loops_input", "int", "Time points."),
            _p("interval_s", "mda_interval_input", "float",
               "Interval between time points in seconds."),
            _p("z_relative", "mda_zrel_input", "text",
               "Comma-separated relative z planes, e.g. '0, 4'."),
            _p("raman_z_indices", "mda_rz_input", "text",
               "Comma-separated z indices for Raman, e.g. '0'."),
        ],
        "description": (
            "Launch the time-lapse Raman acquisition. Requires a selection "
            "to have been prepared first."
        ),
    },
    {
        "name": "stop_mda",
        "label": "Stop MDA",
        "always_run": True,      # never gate a stop
        "method": "stop_raman_mda",
        "params": [],
        "description": "Request a clean stop of the running MDA.",
    },

    # ---- napari layers (safe: viewer only) ----
    {
        "name": "list_layers",
        "label": "List layers",
        "readonly": True,
        "handler": _h_list_layers,
        "params": [],
        "description": "List the names of the current napari layers.",
    },
    {
        "name": "create_points_layer",
        "label": "Create points layer",
        "readonly": True,
        "handler": _h_create_points,
        "params": [
            _p("name", None, "text", "Layer name."),
            _p("points", None, "list",
               "Optional list of [y, x] pixel coordinates to seed the layer.",
               schema={"type": "array",
                       "items": {"type": "array",
                                 "items": {"type": "number"}}}),
        ],
        "description": "Add a napari Points layer (optionally pre-seeded).",
    },
    {
        "name": "create_shapes_layer",
        "label": "Create shapes layer",
        "readonly": True,
        "handler": _h_create_shapes,
        "params": [
            _p("name", None, "text", "Layer name."),
            _p("rectangle", None, "list",
               "Optional [y0, x0, y1, x1] bounds to draw a rectangle.",
               schema={"type": "array", "items": {"type": "number"}}),
        ],
        "description": (
            "Add a napari Shapes layer, optionally with one rectangle "
            "(useful for the Spatial mapping section)."
        ),
    },

    # ---- camera (hardware: gated) ----
    {
        "name": "list_channels",
        "label": "List channels",
        "readonly": True,
        "handler": _h_list_channels,
        "params": [],
        "description": "List the available Micro-Manager channels.",
    },
    {
        "name": "set_camera_exposure",
        "label": "Set camera exposure",
        "handler": _h_set_exposure,
        "params": [_p("exposure_ms", None, "float", "Exposure in ms.")],
        "description": "Set the live camera exposure via the core.",
    },
    {
        "name": "set_channel",
        "label": "Set channel",
        "handler": _h_set_channel,
        "params": [_p("channel", None, "text",
                      "Channel name, e.g. 'BF', 'GFP'.")],
        "description": "Switch the Micro-Manager 'Channel' config group.",
    },
    {
        "name": "snap_image",
        "label": "Snap image",
        "handler": _h_snap,
        "params": [],
        "description": "Snap one camera image into a 'snap' layer.",
    },
    {
        "name": "start_live",
        "label": "Start live",
        "handler": _h_start_live,
        "params": [_p("exposure_ms", None, "float",
                      "Optional exposure to set first (ms).")],
        "description": "Start continuous (live) camera acquisition.",
    },
    {
        "name": "stop_live",
        "label": "Stop live",
        "readonly": True,        # stopping is safe
        "handler": _h_stop_live,
        "params": [],
        "description": "Stop live / sequence acquisition.",
    },

    # ---- image geometry (safe queries) ----
    {
        "name": "get_image_size",
        "label": "Read image size",
        "readonly": True,
        "handler": _h_get_image_size,
        "params": [],
        "description": (
            "Report the camera image width, height and center pixel. Use "
            "this to compute the true image center -- never guess it."
        ),
    },
    {
        "name": "center_on_pixel",
        "label": "Center a pixel (Vandermonde)",
        "handler": _h_center_on_pixel,
        "params": [
            _p("y", None, "float",
               "Pixel row to bring to the mask center; default = image center."),
            _p("x", None, "float",
               "Pixel column to bring to the mask center; default = image "
               "center."),
        ],
        "description": (
            "Move the stage (via the Vandermonde model) so a given pixel "
            "lands at the mask center. Omit y/x to use the true image "
            "center. Requires a loaded Vandermonde model."
        ),
    },

    # ---- open napari-micromanager docks (safe) ----
    {
        "name": "open_mm_widget",
        "label": "Open napari-micromanager widget",
        "readonly": True,
        "handler": _h_open_mm_widget,
        "params": [_p("widget", None, "text",
                      "Sub-dock name, e.g. 'MDA', 'Stages', 'Camera ROI'.")],
        "description": (
            "Reveal a napari-micromanager sub-dock such as the stage "
            "controller ('Stages') or 'MDA'. Matches names case-insensitively "
            "and lists the available ones if there is no match. If you are "
            "unsure of the names, call inspect_mm first."
        ),
    },
    {
        "name": "inspect_mm",
        "label": "Inspect napari-micromanager",
        "readonly": True,
        "handler": _h_inspect_mm,
        "params": [],
        "description": (
            "List the napari-micromanager sub-dock names and relevant methods "
            "so the right one can be opened."
        ),
    },

    # ---- stage (hardware: gated + clamped) ----
    {
        "name": "get_stage_position",
        "label": "Read stage position",
        "readonly": True,
        "handler": _h_get_stage,
        "params": [],
        "description": "Report the current stage X, Y and Z in um.",
    },
    {
        "name": "move_stage_relative",
        "label": "Move stage (relative)",
        "handler": _h_move_stage_relative,
        "params": [
            _p("dx", None, "float", "Relative X move in um."),
            _p("dy", None, "float", "Relative Y move in um."),
            _p("dz", None, "float", "Relative Z move in um."),
        ],
        "description": (
            "Move the stage by a relative offset in um. Each axis is capped "
            "by a safety limit; absolute moves are intentionally not offered."
        ),
    },

    # ---- MDA widget (hardware: gated) ----
    {
        "name": "start_mda",
        "label": "Start napari-micromanager MDA",
        "handler": _h_start_mda,
        "params": [],
        "description": (
            "Run the sequence currently set in the napari-micromanager MDA "
            "widget (the standard MDA, not the Raman engine)."
        ),
    },
    {
        "name": "build_mda_sequence",
        "label": "Build MDA sequence",
        "readonly": True,        # configures the widget; no hardware motion
        "handler": _h_build_mda,
        "params": [
            _p("channels", None, "list",
               "Channel config names, e.g. ['BF', 'GFP'].",
               schema={"type": "array", "items": {"type": "string"}}),
            _p("exposure_ms", None, "float",
               "Exposure for the channels above (ms)."),
            _p("loops", None, "int", "Number of time points."),
            _p("interval_s", None, "float", "Seconds between time points."),
            _p("z_range_um", None, "float",
               "Total Z range (um) for a centered z-stack."),
            _p("z_step_um", None, "float", "Z step (um)."),
            _p("add_current_position", None, "check",
               "Seed the sequence with the current stage position.",
               schema={"type": "boolean"}),
            _p("positions", None, "list",
               "Explicit stage positions as [x, y] or [x, y, z] lists.",
               schema={"type": "array",
                       "items": {"type": "array",
                                 "items": {"type": "number"}}}),
        ],
        "description": (
            "Create a useq MDA sequence (channels, z-stack, timelapse, "
            "positions) and load it into the napari-micromanager MDA widget. "
            "Does NOT run it -- use start_mda afterwards. Omitted parts are "
            "left at their defaults."
        ),
    },
    {
        "name": "add_current_position_to_mda",
        "label": "Add current position to MDA",
        "readonly": True,
        "handler": _h_add_current_position,
        "params": [],
        "description": (
            "Append the current stage XYZ to the MDA widget's position list. "
            "Build a multi-position run by moving the stage and adding."
        ),
    },
]

ACTIONS_BY_NAME = {a["name"]: a for a in ACTIONS}

_KIND_TO_JSON = {
    "text": "string",
    "int": "integer",
    "float": "number",
    "combo": "string",
    "check": "boolean",
}


def build_tools():
    """Generate the Anthropic tools list from ACTIONS."""
    tools = []
    for a in ACTIONS:
        props = {}
        for p in a["params"]:
            if "schema" in p:
                schema = dict(p["schema"])
                schema.setdefault("description", p["description"])
            else:
                schema = {
                    "type": _KIND_TO_JSON[p["kind"]],
                    "description": p["description"],
                }
                if "enum" in p:
                    schema["enum"] = p["enum"]
            props[p["name"]] = schema
        tools.append({
            "name": a["name"],
            "description": a["description"],
            "input_schema": {
                "type": "object",
                "properties": props,
                "required": [],
            },
        })
    return tools


SYSTEM_PROMPT = (
    "You are a control assistant embedded in a napari Raman-microscope panel. "
    "Your ONLY capability is to run the panel's existing GUI actions via the "
    "provided tools, optionally setting a few fields first. You cannot do "
    "anything the buttons cannot already do. "
    "When the user asks to run something, pick the single best matching tool "
    "and include only the fields they specified; leave the rest to their "
    "current GUI values. If the request is ambiguous or could damage the "
    "sample or hardware, ask a clarifying question instead of guessing. "
    "Use get_state to check status before acting when it helps. "
    "NEVER guess image dimensions or the image center: call get_image_size "
    "(or get_state) and compute the center from the real width/height. "
    "To bring the field to center, prefer center_on_pixel with no arguments "
    "(it uses the true image center). Keep replies short."
)


class ChatPanel(QWidget):
    """A small chat box that controls the HardwareWidget via Claude tool-use."""

    # worker-thread -> main-thread signals
    _tool_request = Signal(object)   # payload dict; BlockingQueued
    _post = Signal(str, str)         # (who, text); Queued
    _set_busy = Signal(bool)

    def __init__(self, hardware_widget, confirm=True):
        super().__init__()
        self.hw = hardware_widget
        self.confirm = confirm       # gate hardware actions with a dialog
        self._messages = []          # conversation history
        self._busy = False

        layout = QVBoxLayout(self)
        title = QLabel("Assistant")
        title.setStyleSheet("font-weight: bold;")
        layout.addWidget(title)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMinimumHeight(160)
        layout.addWidget(self.log)

        self.input = QLineEdit()
        self.input.setPlaceholderText(
            "e.g. connect, then set wavelength to 785"
        )
        self.input.returnPressed.connect(self._on_send)
        layout.addWidget(self.input)

        self.send_btn = QPushButton("Send")
        self.send_btn.clicked.connect(self._on_send)
        layout.addWidget(self.send_btn)

        # cross-thread wiring
        self._tool_request.connect(
            self._on_tool_request, Qt.BlockingQueuedConnection
        )
        self._post.connect(self._on_post)
        self._set_busy.connect(self._on_set_busy)

        self._append("system", "Ready. Type a command and press Send.")

    # ---------- UI helpers (main thread) ----------
    def _append(self, who, text):
        prefix = {
            "you": "You",
            "assistant": "Assistant",
            "tool": "•",
            "system": "—",
        }.get(who, who)
        self.log.append(f"<b>{prefix}:</b> {text}" if who != "tool"
                        else f"<i>{prefix} {text}</i>")

    def _on_post(self, who, text):
        self._append(who, text)

    def _on_set_busy(self, busy):
        self._busy = busy
        self.send_btn.setEnabled(not busy)
        self.input.setEnabled(not busy)

    # ---------- send ----------
    def _on_send(self):
        if self._busy:
            return
        text = self.input.text().strip()
        if not text:
            return
        self.input.clear()
        self._append("you", text)
        self._messages.append({"role": "user", "content": text})
        self._set_busy.emit(True)
        threading.Thread(target=self._run_conversation, daemon=True).start()

    # ---------- worker thread ----------
    def _run_conversation(self):
        try:
            import anthropic
        except Exception:
            self._post.emit("system",
                            "The 'anthropic' package is not installed. "
                            "Run: pip install anthropic")
            self._set_busy.emit(False)
            return
        try:
            client = anthropic.Anthropic()   # reads ANTHROPIC_API_KEY
            tools = build_tools()
            while True:
                resp = client.messages.create(
                    model=MODEL,
                    max_tokens=MAX_TOKENS,
                    system=SYSTEM_PROMPT,
                    tools=tools,
                    messages=self._messages,
                )
                # record assistant turn verbatim (needed for tool loop)
                self._messages.append(
                    {"role": "assistant", "content": resp.content}
                )
                # show any text the model produced
                for block in resp.content:
                    if getattr(block, "type", None) == "text" and block.text:
                        self._post.emit("assistant", block.text.strip())

                if resp.stop_reason != "tool_use":
                    break

                # run each requested tool on the main thread
                tool_results = []
                for block in resp.content:
                    if getattr(block, "type", None) != "tool_use":
                        continue
                    result_text = self._run_tool_blocking(
                        block.name, dict(block.input or {})
                    )
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result_text,
                    })
                self._messages.append(
                    {"role": "user", "content": tool_results}
                )
        except Exception as e:
            self._post.emit("system", f"Error: {e}")
        finally:
            self._set_busy.emit(False)

    def _run_tool_blocking(self, name, tool_input):
        """Emit to the main thread and block until the tool finishes."""
        holder = {"text": ""}
        payload = {"name": name, "input": tool_input, "result": holder}
        self._tool_request.emit(payload)   # blocks (BlockingQueuedConnection)
        return holder["text"]

    # ---------- tool execution (main thread) ----------
    def _on_tool_request(self, payload):
        payload["result"]["text"] = self._execute_tool(
            payload["name"], payload["input"]
        )

    def _execute_tool(self, name, tool_input):
        action = ACTIONS_BY_NAME.get(name)
        if action is None:
            return f"Unknown action '{name}'."
        hw = self.hw

        # special read-only query
        if name == "get_state":
            return self._read_state()

        handler = action.get("handler")

        if handler is None:
            # button action: set any provided fields (reusing the widgets'
            # own validation), then call the bound method.
            applied = []
            for p in action["params"]:
                if p["name"] not in tool_input:
                    continue
                val = tool_input[p["name"]]
                w = getattr(hw, p["attr"], None)
                if w is None:
                    continue
                try:
                    kind = p["kind"]
                    if kind == "text":
                        w.setText(str(val))
                    elif kind == "int":
                        w.setValue(int(round(float(val))))
                    elif kind == "float":
                        w.setValue(float(val))
                    elif kind == "combo":
                        w.setCurrentText(str(val))
                    elif kind == "check":
                        w.setChecked(bool(val))
                    applied.append(f"{p['name']}={val}")
                except Exception as e:
                    return f"Couldn't set {p['name']} to {val!r}: {e}"
        else:
            # handler action (layers / camera / stage / MDA): the handler
            # reads tool_input directly, so just summarize the inputs.
            applied = [f"{k}={v}" for k, v in tool_input.items()]

        self._post.emit("tool", f"{action['label']}"
                        + (f" ({', '.join(applied)})" if applied else ""))

        # gate anything that isn't read-only or an emergency stop
        if not (action.get("readonly") or action.get("always_run")):
            if self.confirm:
                summary = f"Run '{action['label']}'?"
                if applied:
                    summary += "\n\n" + "\n".join(applied)
                reply = QMessageBox.question(
                    self, "Confirm action", summary,
                    QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
                )
                if reply != QMessageBox.Yes:
                    return "User declined to run this action."

        if handler is not None:
            try:
                return handler(hw, tool_input)
            except Exception as e:
                return f"Action raised: {e}"
        return self._call_method(action)

    def _call_method(self, action):
        method = getattr(self.hw, action["method"], None)
        if method is None:
            return f"Widget has no method '{action['method']}'."
        try:
            method()
        except Exception as e:
            return f"Action raised: {e}"
        # feed the status bar back to the model as the result
        try:
            return f"Done. Status: {self.hw.status.text()}"
        except Exception:
            return "Done."

    def _read_state(self):
        hw = self.hw
        connected = hw.core is not None
        try:
            status = hw.status.text()
        except Exception:
            status = "?"
        try:
            wl = hw.wl_current_label.text()
        except Exception:
            wl = "?"
        try:
            grating = hw.grating_combo.currentText() or "?"
        except Exception:
            grating = "?"
        try:
            x_size, y_size = hw._get_image_xy()
            img = (f"image {x_size}x{y_size}, "
                   f"center (y,x)=({y_size // 2},{x_size // 2})")
        except Exception:
            img = "image size unknown"
        return (
            f"connected={connected}; status={status!r}; "
            f"wavelength={wl}; grating={grating}; {img}"
        )