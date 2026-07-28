"""Central hover-help text for HardwareWidget fields.

Every entry in HELP is keyed by the *attribute name* of the widget on
HardwareWidget (e.g. "sel_af_combo" -> self.sel_af_combo). Call
``apply_tooltips(self)`` once at the end of HardwareWidget.__init__ to
attach all of these as hover tooltips.

The wording is drawn from the napari-raman-widget user manual, so the
tooltips and the PDF stay consistent. To edit help text, change the dict
below -- no changes to widget.py are needed.
"""

# NOTE: Qt wraps long tooltips on its own; use an embedded "\n" to force a
# line break. Keep entries to a sentence or two.
HELP = {
    # ================= LOADING =================
    "cfg_path": (
        "Micro-Manager device configuration (.cfg) loaded on Connect. "
    ),
    "tf_path": (
        "Coordinate transformer (.json) mapping bright-field pixels to Raman "
        "galvo volts. Required for point collection, calibration, reference, "
        "mapping and Raman MDA."
    ),
    "sel_vdm_path": (
        "Multi-objective pixel-to-stage Vandermonde model (.json). Used when "
        "cells are physically centered."
    ),
    "objective_combo": (
        "Read-only numeric state from Micro-Manager's Objective device. This "
        "index selects the matching JSON calibration; the widget never "
        "commands the objective turret."
    ),
    "out_path": (
        "Working directory applied on Connect (created if needed). Relative "
        "result paths resolve from here. Editing after connection has no "
        "effect."
    ),
    "connect_btn": (
        "Unload old devices, load the CFG, open the napari-micromanager dock, "
        "create the collector/DAQ, load both models, and refresh channels."
    ),
    "disconnect_btn": (
        "Unload devices and clear all calibration, selection, writer and "
        "model state."
    ),
    "reload_tf_btn": (
        "Reload the Raman transformer AND Vandermonde model from disk without "
        "reconnecting hardware."
    ),

    # ============ COLLECT SPECTRA (points layer) ============
    "exposure_input": "Integration time used by the spectrum collector (ms).",
    "n_input": (
        "Number of identical copies of the transformed galvo coordinate "
        "acquired. Minimum of 2 is enforced (DAQ needs >= 2 samples)."
    ),
    "collect_save_input": (
        "Optional filename; the .npy suffix is added if missing. Relative "
        "names save under the working directory. Blank = display only."
    ),
    "collect_btn": (
        "Restart the galvo, transform the last layer's first point, collect "
        "N spectra, optionally save, and plot."
    ),

    # ============ LASER AIMING CALIBRATION ============
    "cal_n_input": (
        "Spectra acquired per calibration target. More repeats increase time "
        "but can stabilize detection."
    ),
    "cal_exp_input": "Raman exposure used by the calibrator (ms).",
    "cal_volts_input": (
        "Galvanometer-voltage extent of the calibration. Keep within the "
        "confirmed operating range of the rig."
    ),
    "cal_grid_input": (
        "Calibration sampling density (grid side). Larger grids need "
        "substantially more measurements."
    ),
    "cal_thres_input": (
        "Detection threshold used to accept/localize calibration responses "
        "(interpreted by cns-control.Calibrator)."
    ),
    "calibrate_btn": (
        "Acquire a new calibration dataset with the active transformer, then "
        "open a log and calibration plot."
    ),
    "recal_check": (
        "Reveal the controls for manually correcting the current calibration."
    ),
    "model_name_input": (
        "Base filename for the corrected transformer. Save writes "
        "<name>.json, loads it immediately, and updates the Loading path."
    ),
    "open_selector_btn": (
        "Open the last calibration dataset for correction. Click a point; "
        "Enter=advance, Backspace=back, R=reset, N=mark frame NaN."
    ),
    "save_model_btn": (
        "Save the corrected transformer as <Model name>.json and immediately "
        "activate it."
    ),

    # ============ AXIAL BACKGROUND SCAN ============
    "ref_name_input": "Human-readable prefix for the saved output.",
    "ref_exp_input": "Exposure for each Raman spectrum (ms).",
    "ref_n_input": "Replicate spectra acquired at each axial (Z) position.",
    "ref_range_input": (
        "Half-range (um) around the starting Z; the scan spans -range..+range. "
        "Verify objective clearance."
    ),
    "ref_pts_input": "Number of axial samples across the full search interval.",
    "ref_collect_btn": (
        "Run the autofocus/background scan, move to the found focus Z, plot "
        "all spectra, and save reference/<name>_<uuid>.zarr."
    ),

    # ============ SPATIAL MAPPING ============
    "scan_name_input": "Label inserted into the saved Zarr filename.",
    "scan_exp_input": "Exposure at every Raman grid coordinate (ms).",
    "scan_n_input": (
        "Grid side: an N x N grid = N^2 Raman points per Z plane. Doubling N "
        "roughly quadruples the point count."
    ),
    "scan_z_input": (
        "Base Raman Z = current Z minus this offset (um). The stage returns to "
        "its original Z after a successful scan."
    ),
    "scan_zscan_check": (
        "Collect the full Raman grid at multiple Z planes instead of one."
    ),
    "scan_zrange_input": (
        "Half-range (+/- um) around the base Raman Z. Z-scan only."
    ),
    "scan_zsteps_input": "Evenly spaced planes across the full Z range.",
    "add_channel_btn": (
        "Add a Micro-Manager channel + exposure snapped once per scan. "
        "Duplicates are ignored; BF is excluded (always captured before/after)."
    ),
    "scan_btn": (
        "Snap BF and extra channels, then collect the Raman grid over the "
        "rectangle in the last Shapes layer (using its bounding box)."
    ),

    # ============ GENERATE STAGE GRID ============
    "grid_af_combo": (
        "Autofocus mode attached to the generated positions. Also controls "
        "which MDA autofocus fields are visible."
    ),
    "grid_channel_combo": (
        "Choose a hardware channel for a per-position preview, or choose "
        "Raman for a full pre-scan or a compact procedural grid without a "
        "pre-scan or per-position napari layers."
    ),
    "grid_definition_combo": (
        "Use the current stage XY as the grid center, or define the bounds "
        "with top-left and bottom-right stage positions."
    ),
    "grid_sampling_combo": (
        "Define each axis by a maximum stage spacing or by an exact number "
        "of positions including the two endpoints."
    ),
    "grid_scan_order_combo": (
        "Raster preserves the original Y-fast ordering. Snake X-fast reverses "
        "X on alternating Y rows; Snake Y-fast reverses Y on alternating X "
        "columns. The position order applies to every acquisition channel."
    ),
    "grid_capture_tl_btn": "Copy the current stage XY into the top-left fields.",
    "grid_capture_br_btn": (
        "Copy the current stage XY into the bottom-right fields."
    ),
    "grid_fovx_input": (
        "Fixed image X pixel used at every field of view (paired with FOV y)."
    ),
    "grid_fovy_input": (
        "Fixed image Y pixel used at every field of view (paired with FOV x)."
    ),
    "grid_xrange_input": (
        "Half-width (+/- um) of the stage grid about the current X. Total "
        "span is twice this."
    ),
    "grid_yrange_input": (
        "Half-width (+/- um) of the stage grid about the current Y. Total "
        "span is twice this."
    ),
    "grid_xstep_input": (
        "Maximum stage spacing in X (um). The actual uniform spacing may be "
        "smaller so both grid boundaries are included."
    ),
    "grid_ystep_input": (
        "Maximum stage spacing in Y (um). The actual uniform spacing may be "
        "smaller so both grid boundaries are included."
    ),
    "grid_xcount_input": "Exact number of X positions, including endpoints.",
    "grid_ycount_input": "Exact number of Y positions, including endpoints.",
    "grid_size_btn": (
        "Show grid dimensions, resulting spacing, stage positions, and total "
        "acquisition points including repeats."
    ),
    "grid_tilt_check": (
        "Replace each generated position's default Z with a centered/scaled "
        "2D Vandermonde surface fitted to focused XYZ references."
    ),
    "grid_tilt_degree_input": (
        "Total degree of the centered/scaled 2D Vandermonde Z fit. Degree 1 "
        "requires at least 3 references, degree 2 requires 6, and degree 3 "
        "requires 10. Use the lowest degree that removes systematic residuals."
    ),
    "grid_tilt_capture_btn": (
        "After moving to a grid location and focusing it, capture the current "
        "stage X, Y, and Z into the editable reference table."
    ),
    "grid_tilt_add_btn": "Add a blank, manually editable XYZ reference row.",
    "grid_tilt_remove_btn": "Remove the selected tilt-reference rows.",
    "grid_tilt_clear_btn": "Remove all tilt-reference rows.",
    "grid_repeats_input": (
        "Identical points at each stage position. Minimum of 2 required by "
        "the DAQ."
    ),
    "run_grid_sel_btn": (
        "Stop live mode, prepare the MDA sequence, build the stage grid, and "
        "prepare sources / autofocus_p / new_seq for Run Raman MDA."
    ),

    # ============ HARDWARE CONTROL ============
    "click_laser_btn": (
        "Arm a one-shot viewer click that aims the calibrated laser at the "
        "clicked pixel. Needs the DAQ and transformer."
    ),
    "drag_stage_btn": (
        "Enable stage movement: hold the left mouse button and drag in the "
        "viewer. The stage follows the drag direction; distance controls speed. "
        "Release the mouse to stop."
    ),
    "stage_drag_speed_input": (
        "Maximum click-and-drag stage speed in micrometers per second. The "
        "cursor reaches this speed 100 pixels from the press point."
    ),
    "open_shutter_btn": (
        "Open the laser shutter by switching the Channel config to RM."
    ),
    "close_shutter_btn": (
        "Close the laser shutter by restoring the previous imaging channel."
    ),
    "open_filter_btn": (
        "Open the ND-filter path by removing the autofocus filter with the "
        "Micro-Manager DigitalIO device."
    ),
    "close_filter_btn": (
        "Close the ND-filter path by inserting the autofocus filter with the "
        "Micro-Manager DigitalIO device."
    ),
    "click_center_btn": (
        "Arm a one-shot viewer click that moves the stage so the clicked "
        "feature reaches Center Y/X. Needs a Vandermonde model + hardware; "
        "commands stage motion."
    ),

    # ============ AUTOMATED CELL SELECTION ============
    "sel_cy_input": (
        "Center of the circular permitted region (Y pixel). Same value is "
        "passed to the Raman MDA engine."
    ),
    "sel_cx_input": (
        "Center of the circular permitted region (X pixel). Same value is "
        "passed to the Raman MDA engine."
    ),
    "sel_r_input": (
        "Radius of the permitted circular region (px). Also passed to the "
        "Raman MDA engine."
    ),
    "add_mask_btn": (
        "Add a red masked overlay with a green center marker to inspect the "
        "permitted region before selection."
    ),
    "sel_af_combo": (
        "Autofocus strategy saved with the selection. None disables autofocus "
        "in the later MDA."
    ),
    "sel_npf_input": (
        "Requested cell count per FOV. Automated selection passes N+1 to its "
        "helper; in manual batch mode it is the exact clicks per FOV."
    ),
    "sel_center_cell_check": (
        "Split detections into one new stage position per cell, each shifted "
        "so the cell sits at the center. Requires the Vandermonde model."
    ),
    "sel_shape_combo": (
        "Shape of the Raman subpoint pattern placed around each selected cell."
    ),
    "sel_sqsize_input": "Spatial extent of the aiming point pattern (px).",
    "sel_sqn_input": (
        "Pattern sampling parameter. Batch MDA requires the resulting pattern "
        "multiplier to be at least 2."
    ),
    "sel_bkd_input": (
        "Distance threshold (px) used by automated selection for background "
        "placement."
    ),
    "sel_batch_combo": (
        "Batch vs individual point handling. Must agree with how the dataset "
        "is later generated."
    ),
    "sel_cellpose_combo": (
        "Segmentation model used to identify candidate cells (defaults to "
        "cyto2 if available)."
    ),
    "run_selection_btn": (
        "Prepare the MDA widget, run Cellpose-based selection, create source "
        "layers and a new sequence, and store them for Raman MDA."
    ),
    "run_manual_btn": (
        "Create empty point-source layers to hand-click cells. Batch = click "
        "exactly N per FOV; non-batch = click freely. Finish clicking before "
        "running the MDA."
    ),
    "center_manual_btn": (
        "Turn each clicked cell (non-batch) into a centered stage position via "
        "the Vandermonde model, replacing the selection results."
    ),

    # ============ RUN RAMAN MDA ============
    "mda_dir_input": (
        "Directory for Raman TIFF/NumPy outputs. Blank uses data/run."
    ),
    "mda_afp_input": (
        "Optional comma-separated 0-based position indices to autofocus, e.g. "
        "0,2,5. Blank or 'None' = use the selection. Out-of-range is rejected."
    ),
    "mda_imgp_input": (
        "Optional comma-separated 0-based indices to image. Initialized from "
        "the selection's autofocus list; set explicitly when overriding "
        "autofocus differently."
    ),
    "mda_af_range_input": "Coarse autofocus range. Hidden if autofocus is None.",
    "mda_search_pts_input": "Coarse autofocus sample count.",
    "mda_fine_range_input": (
        "Fine autofocus half-range (+/- um). Shown only for laser autofocus."
    ),
    "mda_fine_pts_input": "Fine laser-autofocus sample count.",
    "mda_seg_track_check": (
        "Re-segment images and update aiming during the time series."
    ),
    "mda_seg_ch_combo": "Micro-Manager channel used for segmentation.",
    "mda_seg_scale_input": "Image scale used during segmentation.",
    "mda_seg_model_combo": (
        "Cellpose model for time-series re-segmentation (cyto2 preferred)."
    ),
    "mda_seg_crop_combo": (
        "Whether segmentation is cropped to the circular mask region."
    ),
    "mda_track_cfg_input": "Particle-tracking configuration file (.json).",
    "mda_exp_input": (
        "Total exposure while building the Raman sequence (ms). In non-batch "
        "mode it is multiplied by the aiming-pattern multiplier."
    ),
    "mda_loops_input": "Number of temporal repetitions (time points).",
    "mda_interval_input": "Requested interval between time points (s).",
    "mda_refocus_input": (
        "Cadence in time points for focus and tracking updates; 1 = every "
        "time point."
    ),
    "mda_zrel_input": (
        "Relative Z planes (comma-separated um) that replace the sequence Z "
        "plan, e.g. '0, 4'. Must parse as floats."
    ),
    "mda_rz_input": (
        "0-based indices into the Z list where Raman is requested. Every "
        "index must exist (two Z values -> valid indices 0 and 1). Enter "
        "None or Off to skip Raman spectra; use Software or None autofocus."
    ),
    "mda_add_channel_btn": (
        "Add any Micro-Manager channel (incl. BF) to the sequence. Duplicates "
        "ignored; added channels inherit the first channel as a template."
    ),
    "run_mda_btn": (
        "Build and launch the final acquisition using the MDA widget's axis "
        "order, selection sources, replaced time/Z plans and Raman metadata."
    ),
    "stop_mda_btn": (
        "Request MDA cancellation and stop sequence acquisition; the current "
        "hardware event may finish before exit."
    ),
    "view_acquisition_btn": (
        "Open the indexed, on-demand viewer. Imaging and Raman folders can be "
        "selected independently; only the requested spectrum and image tiles "
        "are loaded."
    ),

    # ---- pixel-to-stage calibration (inside Run Raman MDA) ----
    "px2stage_check": "Reveal the Vandermonde pixel-to-stage calibration workflow.",
    "px2stage_ds_path": (
        "Generated dataset (.zarr) with a JSON useq_sequence attribute and one "
        "image per stage position."
    ),
    "px2stage_degree_input": (
        "Polynomial degree. Degree d needs >= (d+1)(d+2)/2 valid points; "
        "prefer the lowest degree with adequate residuals."
    ),
    "px2stage_pick_btn": (
        "Open images position by position; click the same feature in each "
        "frame. Skipped/NaN frames are excluded."
    ),
    "px2stage_name_input": (
        "Suggested save name; a save dialog still asks for the final location."
    ),
    "px2stage_save_btn": (
        "Center coordinates, report degree 1-3 RMSE, fit the selected degree, "
        "and update the active objective in the shared JSON without replacing "
        "other objective calibrations."
    ),
}


def apply_tooltips(widget):
    """Attach every HELP entry as a hover tooltip on the matching attribute
    of ``widget`` (a HardwareWidget). Missing attributes are skipped, so it's
    safe to call even if some fields are renamed or removed.

    Returns the number of tooltips actually applied (handy as a coverage
    check during development).
    """
    applied = 0
    for attr, text in HELP.items():
        w = getattr(widget, attr, None)
        if w is not None and hasattr(w, "setToolTip"):
            w.setToolTip(text)
            applied += 1
    return applied
