# Unified MDA in 1.0.0

The MDA dock now owns experiment channels, exposures, positions, grids, time plans,
Z plans, axis order, output settings, and Run/Pause/Resume/Cancel. Its extension
sections configure Raman collection, custom autofocus, and segmentation/tracking.
The Raman dock retains connection, live hardware control, calibration, scans, and
target preparation. The old Run Raman MDA section has been removed.

## Preparing an experiment

1. Connect the microscope. Raman hardware initialization is optional for camera MDA.
2. Prepare cell targets or Raman grids in Raman, if needed.
3. In Generate stage grid, **Dense grid** is checked by default. Click **Use grids
   in MDA** to attach its compact definition. MDA shows the virtual position count;
   its position table stays empty. Uncheck Dense grid before adding grids to use
   explicit MDA positions instead. All regions in a run share this mode.
   **Apply prepared positions to MDA** in Automated cell selection still transfers
   individual cell positions. Grid preparation creates one aiming point per position;
   **Frames per point** in MDA controls spectral repeats and is preserved when attaching
   grids. A single frame uses on-demand galvo output without extra spectra.
   Configure autofocus only in MDA. Grids prepare a fixed reference at the aiming
   point, and attaching a grid preserves MDA's autofocus mode and position policy.
   Both handoffs preserve channel, time, Z, and axis-order
   settings. Selection uses a separate survey sequence and does not reset them.
4. Configure the experiment in MDA. Include **RM** only when recording spectra.
   RM exposure is **milliseconds per collection**: per aiming point in individual
   mode, or per integrated spectrum in batch mode. It is not a total cell budget.
5. Choose Raman planes and camera positions independently. A generated Raman grid
   defaults to no saved camera frames. Autofocus/tracking can still acquire internal
   images; the review identifies these additional operations.
6. Enable saving, choose a destination, review the experiment, and run. Delayed runs
   retain a detached snapshot of settings and target coordinates. Cancel before
   their start creates no acquisition folder.

Autofocus and segmentation have independent update cadences. Zero means the initial
timepoint only. Hardware autofocus uses the native MDA autofocus axes and per-position
offsets. The Custom autofocus mode controls only the extension; `off` leaves native
hardware autofocus enabled when its axis checkboxes are selected. Custom autofocus
cannot be combined with native autofocus or absolute Z.
BF target preparation generates stage positions; BF tracking does not dynamically
recenter the camera. Prepared positions include identity suffixes in their names;
preserve those suffixes when renaming or reordering positions.

## Saving settings and measurements

**Save Settings** writes a useq experiment definition containing extension settings,
prepared target coordinates/identities, and calibration file references. It does not
save measurement data or copy calibration files. **Load Settings** restores that setup.
Dense grids save bounds/counts, traversal order, constant Z or fitted surface, and
shared fixed targets. They do not save an expanded position table or duplicate target
coordinates at every grid point. The acquisition engine generates positions lazily;
changing the grid in Raman requires clicking **Use grids in MDA** again. Rebuild
and reattach grids from older saved settings to replace duplicated aiming points;
loading those settings preserves their existing target coordinates. Use **Detach
dense Raman grid** in MDA to return to native position/grid editing.

Camera-only runs without tracking or dense grids use the native MDA output formats. Raman, dense-grid, and tracked
runs use the existing combined folder writer, regardless of the native format menu:

```text
run/
  useq-sequence.json      executed experiment, extension settings and prepared targets
  raman.h5               streamed spectra, spectral axis, points and event metadata
  images/*.ome.tiff     streamed OME-TIFF camera data (only when camera frames exist)
  aiming/*.npz           segmentation masks and old/new target coordinates, if enabled
```

The native and combined writers are never enabled simultaneously. Internal autofocus
or segmentation images are not saved as camera measurements. Tracking mask/coordinate
records omit the internal image. With no camera channels or no selected imaging
positions, no OME-TIFF camera data is produced. The saved acquisition viewer opens
Raman-only scans without an image or camera calibration model. Existing acquisitions
remain readable.
Existing output paths receive a unique sibling unless explicit combined-folder
overwrite is selected and confirmed during review.

Legacy channel-offset metadata is imported into native channel Z offsets. Legacy
Raman plane metadata is recognized. Legacy acquisitions that lack prepared target
coordinates can be opened for analysis, but targets must be prepared again before
repeating Raman acquisition. `HardwareWidget`, `load_experiment`, and deprecated
`run_raman_mda`/`stop_raman_mda` adapters remain available.

## Release verification

The tested integration versions are napari-micromanager 0.1.3, pymmcore-widgets 0.9.1,
pymmcore-plus 0.13.4, and useq-schema 0.9.2. Use a Micro-Manager adapter distribution
compatible with the installed pymmcore device interface. No dependency files in
site-packages need modification.

Before publishing 1.0.0, perform supervised hardware smoke runs for camera-only,
Raman grid, mixed acquisition, each used autofocus mode, tracking, and cancellation.
Simulator tests do not establish hardware calibration accuracy or instrument safety.
