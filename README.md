# napari-raman-widget

A napari dock widget for controlling the Raman microscopy rig.

## What it does

Provides a single collapsible sidebar panel inside napari with sections for:

- Loading Micro-Manager config and transformer model
- Collecting Raman spectra at clicked points
- Running laser aiming calibration
- Manual recalibration via point selector
- Collecting reference spectra with autofocus
- Running spatial Raman mapping (grid scan) over a shape
- Automated cell selection inside a mask
- Running a Raman MDA with fluorescence channels and Z stacks

Relative output paths (reference `.npy` files, `grid_scan_*.zarr`,
recalibrated models, and the MDA writer directory) are resolved beneath the
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

A napari dock widget for controlling the Raman microscopy rig.

## What it does

Provides a single collapsible sidebar panel inside napari with sections for:

- Loading Micro-Manager config and transformer model
- Collecting Raman spectra at clicked points
- Running laser aiming calibration
- Manual recalibration via point selector
- Collecting reference spectra with autofocus
- Running spatial Raman mapping (grid scan) over a shape
- Automated cell selection inside a mask
- Running a Raman MDA with fluorescence channels and Z stacks

It also includes three usability features layered on top of the panel:

- **Inline help** - every field has a hover tooltip explaining what it does,
  and a **Help** link at the top of the panel opens the full PDF user manual.
- **AI assistant** - an optional chat box that maps plain-English commands to
  the panel's existing actions (see [AI assistant](#ai-assistant-chat-panel)).

Relative output paths (reference `.npy` files, `grid_scan_*.zarr`,
recalibrated models, and the MDA writer directory) are resolved beneath the
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
