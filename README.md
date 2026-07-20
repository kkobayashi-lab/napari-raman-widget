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

All outputs (reference `.npy` files, `grid_scan_*.zarr`, recalibrated models,
the MDA writer directory) are written relative to the current working
directory - or an output folder you set in the Loading section, which is
switched to on connect.

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
