# napari-raman-widget

A napari application for controlling the Raman microscopy rig. Version 1.0 unifies
camera and Raman experiments in the MDA dock. See [the migration guide](MIGRATION-1.0.md).

## What it does

Provides a single collapsible sidebar panel inside napari with sections for:

- Loading Micro-Manager config and transformer model
- Collecting Raman spectra at clicked points
- Running laser aiming calibration
- Manual recalibration via point selector
- Collecting reference spectra with autofocus
- Running spatial Raman mapping (grid scan) over a shape
- Automated cell selection inside a mask
- Preparing targets and grids for the shared MDA panel
- Configuring and running camera, Raman, and mixed acquisitions in MDA

Relative output paths (reference `.npy` files, `grid_scan_*.zarr`,
recalibrated models) are resolved beneath the
base folder in **Data Output**. Changing it applies to the next save
immediately and does not reconnect hardware or change the process working
directory.

## Raman selection in MDA

Open **MDA → Raman selection → Select Raman points / batches**. Use a completed acquisition already
in the viewer, or acquire the configured camera channels. Click targets on each
image plane, then choose **Replace acquisition positions** or **Append positions**.
Ordinary points become separate stage positions using the current pixel-to-stage
calibration; Raman and camera channels acquire at the center of each new FOV.
For **Integrated batch (galvo)**, draw rectangles, ellipses, or polygons in the
**Raman batches** layer using napari's native Shapes tools. Each shape is a
separate batch, including multiple shapes on the same image plane. Set **Points
per new batch**, then adjust individual counts in the batch table. Cyan points
preview the exact sampling locations; editing a shape updates its preview.
Click a table row to select that shape and return to its source plane. Lines,
paths, empty regions, and regions outside the image cannot be applied.

The stage moves to each batch's geometric center (area centroid for polygons),
using the current objective's pixel-to-stage calibration. Sampling coordinates
are shifted relative to the new FOV center, and the galvo scans within the batch
during **one RM exposure**. The RM channel exposure in Channels is the
total integration time per batch, not per sampling point. **Spectrum repeats** in
Raman selection repeats the whole batch. Raman planes, camera-position filters,
and galvo prepositioning also live in Raman selection; the old Raman acquisition
section is removed. Centered positions and batches can share a run using Append.

Batch boundaries, counts, identities, and exact sampled points are saved with
positions and experiment settings. Reopening selection on the same source
acquisition restores its saved shapes for editing. Acquisition uses saved points
without regenerating them. Optional source-image metadata can still be omitted.
Previously saved batches retain their saved stage positions; reopen selection
and apply them again to use stage centering.

Raman positions inherit a configured autofocus reference from their original FOV
at its calibrated physical stage XY. Autofocus visits that location and returns
to the Raman position for acquisition. This also applies to integrated batches
and does not require optional source-image metadata. If the parent has no
reference, autofocus stays unset. Acquire camera images at the new positions to
select missing references or replace inherited ones. Inherited references remain
set when other positions are edited, and survive position and run saves.

Selection preserves Channels. Add **RM** to acquire spectra; retain or add camera
channels only when fresh images are wanted. Directly entered stage positions also
support centered Raman without any source image. Position-owned targets currently
require tracking to be off; reselect from updated images when targets move.

Source-image metadata is optional. When enabled it records the selected pixels,
source `(t,p,g,c,z)` indices, calibration when used, and the dataset path when the
source was saved. No source image is required to acquire or load Raman data.
Saved run indices refer to the saved position table, independent of subsequent
edits. Raman HDF5 schema 2 records `(t,p,z,c)`, actual stage coordinates, per-event
batch mode, and a spectrum-aligned repeat index. Batch repeats remain measurements
of the whole sampling set. Older HDF5 runs remain readable.

Replacing positions does not overwrite files. The existing Save Acquisition
behavior remains: choose a fresh destination by default; replace the entire run
folder only when **Overwrite acquisition folder** is enabled and reviewed.

## Viewer stage controls

In **Hardware Control**, enable **Drag image to move stage** to move XY by
holding and dragging the image. While this mode is armed, scrolling over the
image moves the Z focus stage: up increases Z and down decreases Z. The
**Z scroll** slider just below the drag controls sets the distance per wheel
notch from 0.1 to 10 micrometers (default 1.0). Scrolling replaces image zoom
in this mode. Press Escape or turn the mode off to restore normal scrolling.
Stage motion is disabled during MDA; wheel input is ignored while Z is busy.

## Install

```bash
pip install -e .
```

This assumes `pymmcore-plus`, `raman-control`, `raman-mda-engine`, and
`cns-control` are already installed in the same environment (they are not
on PyPI and must be installed from your internal/local sources).

On Windows, follow [WINDOWS_ENVIRONMENT.md](WINDOWS_ENVIRONMENT.md) to avoid
mixing incompatible Conda and PyTorch OpenMP runtimes in the napari process.

## Run

From inside the repo, with your conda environment active:

```bash
python run_napari.py
```

### One-click launcher (Windows)

A `launch_napari.bat` script is included for convenience. Double-click it to:

1. Activate your conda environment
2. Change into the repo directory
3. Launch napari with the widget

Before using it, edit `launch_napari.bat` to match your setup:

- The `call ... activate.bat <env-name>` line: replace `<env-name>` with
  your own conda environment name.
- The `cd /d <repo-path>` line: replace with the path to your local clone
  of this repo.

You can also pin the launcher to the taskbar or Start menu:

1. Right-click `launch_napari.bat` -> Create shortcut.
2. Right-click the shortcut -> Properties, and prepend `cmd /c ` to the
   Target field so it becomes `cmd /c "<full path>\launch_napari.bat"`.
3. Optionally click Change Icon to give it a recognizable icon.
4. Right-click the shortcut -> Pin to taskbar (or drag it to the desktop).

## Structure

- `run_napari.py` - entry point; just launches napari with the widget.
- `launch_napari.bat` - Windows one-click launcher (activates env + runs script).
- `napari_raman_widget/widget.py` - the main `HardwareWidget` class.
- `napari_raman_widget/plot_windows.py` - matplotlib pop-up windows.
- `napari_raman_widget/log_window.py` - streaming stdout log window.
- `napari_raman_widget/ui_helpers.py` - small Qt helpers.

Hardware-library imports (`pymmcore_plus`, `cns_control`, etc.) are done
lazily inside methods, so the package imports cleanly on machines without
the rig.

## License

`napari-raman-widget` is distributed under the terms of the [BSD-3-Clause](https://spdx.org/licenses/BSD-3-Clause.html) license.
# napari-raman-widget

A napari application for controlling the Raman microscopy rig. Version 1.0 unifies
camera and Raman experiments in the MDA dock. See [the migration guide](MIGRATION-1.0.md).

## What it does

Provides a single collapsible sidebar panel inside napari with sections for:

- Loading Micro-Manager config and transformer model
- Collecting Raman spectra at clicked points
- Running laser aiming calibration
- Manual recalibration via point selector
- Collecting reference spectra with autofocus
- Running spatial Raman mapping (grid scan) over a shape
- Automated cell selection inside a mask
- Preparing targets and grids for the shared MDA panel
- Configuring and running camera, Raman, and mixed acquisitions in MDA

It also includes three usability features layered on top of the panel:

- **Inline help** - every field has a hover tooltip explaining what it does,
  and a **Help** link at the top of the panel opens the full PDF user manual.
- **AI assistant** - an optional chat box that maps plain-English commands to
  the panel's existing actions (see [AI assistant](#ai-assistant-chat-panel)).

Relative output paths (reference `.npy` files, `grid_scan_*.zarr`,
recalibrated models) are resolved beneath the
base folder in **Data Output**. Changing it applies to the next save
immediately and does not reconnect hardware or change the process working
directory.

## Install

```bash
pip install -e .
```

This assumes `pymmcore-plus`, `raman-control`, `raman-mda-engine`, and
`cns-control` are already installed in the same environment (they are not
on PyPI and must be installed from your internal/local sources).

The AI assistant is optional. To enable it, also install the Anthropic SDK
and set an API key (see [AI assistant](#ai-assistant-chat-panel)):

```bash
pip install anthropic
```

## Run

From inside the repo, with your conda environment active:

```bash
python run_napari.py
```

### One-click launcher (Windows)

A `launch_napari.bat` script is included for convenience. Double-click it to:

1. Activate your conda environment
2. Change into the repo directory
3. Launch napari with the widget

Before using it, edit `launch_napari.bat` to match your setup:

- The `call ... activate.bat <env-name>` line: replace `<env-name>` with
  your own conda environment name.
- The `cd /d <repo-path>` line: replace with the path to your local clone
  of this repo.

You can also pin the launcher to the taskbar or Start menu:

1. Right-click `launch_napari.bat` -> Create shortcut.
2. Right-click the shortcut -> Properties, and prepend `cmd /c ` to the
   Target field so it becomes `cmd /c "<full path>\launch_napari.bat"`.
3. Optionally click Change Icon to give it a recognizable icon.
4. Right-click the shortcut -> Pin to taskbar (or drag it to the desktop).

## Inline help and manual

Two forms of built-in documentation ship with the panel:

- **Hover tooltips.** Every editable field and button carries a short
  description shown on mouse-over. The text lives in one central dictionary
  in `napari_raman_widget/field_help.py` (keyed by widget attribute name) and
  is attached in one call, `apply_tooltips(self)`, at the end of the widget's
  constructor. To reword a tooltip, edit that dictionary only - no other file
  changes are needed. The wording is kept consistent with the PDF manual.
- **User manual.** A **Help** link at the top-right of the panel opens
  `napari_raman_widget/resources/napari-raman-widget-manual.pdf`, a detailed
  guide to every section, the layer each workflow consumes, what must be
  prepared first, and what output is produced. The manual's LaTeX source is
  kept alongside it so it can be regenerated.

## AI assistant (chat panel)

`napari_raman_widget/chat_panel.py` adds an optional chat box that turns
plain-English requests into the panel's existing GUI actions. It never
touches hardware directly - it drives the same methods the buttons call (and
a few napari / Micro-Manager operations), so it reuses every existing range
check and validation. Anything that moves the stage, laser, or shutters pops
a confirmation dialog first; read-only queries run automatically.

**What it can do:**

- Run any panel action (connect, calibrate, collect spectra, run selection,
  run the Raman MDA, generate a dataset, ...).
- Read state: connection, status, wavelength, grating, and image size.
- Create napari Points/Shapes layers.
- Set camera exposure and channel; snap; start/stop live.
- Move the stage by a relative offset (clamped by a safety limit), and read
  the current stage position.
- Open napari-micromanager sub-docks (stage controller, MDA, ...).
- Build a `useq` MDA sequence (channels, z-stack, timelapse, positions) and
  load it into the MDA widget, add the current stage position to it, then
  start it.

**Setup:**

1. `pip install anthropic`
2. Set an Anthropic API key in the environment **before** launching napari,
   e.g. on Windows: `setx ANTHROPIC_API_KEY "sk-ant-..."` (open a new
   terminal afterwards), or per-session `$env:ANTHROPIC_API_KEY = "sk-ant-..."`.
3. Set `MODEL` at the top of `chat_panel.py` to a model your account can
   access.

**Enable it in the widget** by adding, near the end of `HardwareWidget.__init__`
(before the scroll wrapper):

```python
from .chat_panel import ChatPanel
self.chat_panel = ChatPanel(self)
outer.addWidget(self.chat_panel)
```

Adding a new capability is a one-entry change to the `ACTIONS` registry in
`chat_panel.py`; the API tool schemas are generated from it automatically.
Pass `ChatPanel(self, confirm=False)` to run recognized commands without the
confirmation dialog (not recommended on live hardware).

## Structure

- `run_napari.py` - entry point; just launches napari with the widget.
- `launch_napari.bat` - Windows one-click launcher (activates env + runs script).
- `napari_raman_widget/widget.py` - the main `HardwareWidget` class.
- `napari_raman_widget/field_help.py` - central hover-tooltip text and the
  `apply_tooltips()` helper.
- `napari_raman_widget/chat_panel.py` - optional LLM-backed assistant that
  drives the panel's actions.
- `napari_raman_widget/plot_windows.py` - matplotlib pop-up windows.
- `napari_raman_widget/log_window.py` - streaming stdout log window.
- `napari_raman_widget/ui_helpers.py` - small Qt helpers.
- `napari_raman_widget/resources/napari-raman-widget-manual.pdf` - the user
  manual opened by the Help link (LaTeX source kept alongside).

Hardware-library imports (`pymmcore_plus`, `cns_control`, etc.) are done
lazily inside methods, so the package imports cleanly on machines without
the rig. The `anthropic` import used by the chat panel is likewise lazy, so
the package still imports (and the assistant simply reports it is unavailable)
when the SDK or API key is missing.

## License

`napari-raman-widget` is distributed under the terms of the [BSD-3-Clause](https://spdx.org/licenses/BSD-3-Clause.html) license.
