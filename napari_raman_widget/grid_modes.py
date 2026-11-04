"""Imaging tiles and Raman point grids in the shared MDA Grid editor."""
from copy import deepcopy

from qtpy.QtWidgets import QComboBox, QFormLayout, QLabel, QSpinBox, QTabBar, QVBoxLayout, QWidget

from raman_mda_engine.grid_plan import GRID_KEY
from .experiment import METADATA_KEY


class GridModes:
    def __init__(self, panel, owner):
        self.panel = panel
        self.owner = owner
        self.mode = "imaging"
        self.drafts = {}
        self.loaded_grid = None
        self.imaging = panel.grid_plan.takeWidget()
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.addWidget(QLabel("Grid acquisition type"))
        self.selector = QComboBox()
        self.selector.addItem("Imaging grid — camera fields", "imaging")
        self.selector.addItem("Raman grid — center points", "raman")
        layout.addWidget(self.selector)
        layout.addWidget(self.imaging)
        self.raman = QWidget()
        self.raman_layout = QVBoxLayout(self.raman)
        self.raman_layout.setContentsMargins(0, 0, 0, 0)
        repeats_form = QFormLayout()
        self.repeats = QSpinBox()
        self.repeats.setRange(owner.mda_repeats_input.minimum(), owner.mda_repeats_input.maximum())
        self.repeats.setValue(owner.mda_repeats_input.value())
        self.repeats.setToolTip("Consecutive spectra at each grid point, independent of MDA time points.")
        self.repeats.valueChanged.connect(owner.mda_repeats_input.setValue)
        owner.mda_repeats_input.valueChanged.connect(self.repeats.setValue)
        repeats_form.addRow("Frames per Raman grid point", self.repeats)
        self.raman_layout.addLayout(repeats_form)
        note = QLabel("Add grids below, then click Run MDA. These are absolute stage grids; "
                      "the Positions list is kept separately and is not repeated for each grid.")
        note.setWordWrap(True)
        self.raman_layout.addWidget(note)
        from .raman_grid_editor import RamanGridEditor
        self.editor = RamanGridEditor(self)
        layout.addWidget(self.raman)
        panel.grid_plan.setWidget(container)
        self.selector.currentIndexChanged.connect(self.change_mode)
        self.refresh()

    def compile_regions(self, metadata):
        if self.mode != "raman" or not hasattr(self.owner, "_grid_regions"):
            return
        regions = self.owner._grid_regions
        metadata.pop(GRID_KEY, None)
        if not regions:
            self.panel.targets = [record for record in self.panel.targets if not record.get("fixed")]
            metadata[METADATA_KEY]["targets"] = deepcopy(self.panel.targets)
            return
        from useq import MDASequence, Position
        from .experiment import capture_targets
        from cns_control.utils import combine_grid_sequences

        sequence = combine_grid_sequences(region["sequence"] for region in regions)
        metadata[GRID_KEY] = deepcopy(sequence.metadata[GRID_KEY])
        records = capture_targets(MDASequence(stage_positions=[Position()]), regions[0]["sources"])
        for record in records:
            record["fixed"] = True
            record["position_ids"] = []
        metadata[METADATA_KEY]["targets"] = records
        self.panel.targets = deepcopy(records)
        metadata[METADATA_KEY]["extensions"]["raman"].update(batch=False, pre_acq=False)

    def restore_regions(self, sequence):
        plan = sequence.metadata.get(GRID_KEY)
        if not hasattr(self.owner, "_grid_regions") or plan == self.loaded_grid:
            return
        self.loaded_grid = deepcopy(plan)
        if plan is None:
            if self.mode == "raman":
                self.owner._grid_regions = []
                self.owner._refresh_grid_region_list()
            return
        from useq import MDASequence
        from .experiment import SnapshotSource

        records = sequence.metadata.get(METADATA_KEY, {}).get("targets", [])
        sources = [SnapshotSource(record) for record in records]
        shape = next((record["image_shape"] for record in records if record["role"] == "cell"),
                     (self.owner.core.getImageWidth(), self.owner.core.getImageHeight()))
        settings = dict(fov_x=shape[0] / 2, fov_y=shape[1] / 2, dense=True)
        self.owner.grid_dense_check.setChecked(True)
        self.owner._grid_regions = [dict(
            sequence=MDASequence(metadata={GRID_KEY: {"version": 1, "regions": [deepcopy(region)]}}),
            sources=sources, source_settings=settings.copy(),
            tilt_reference_points=None,
            tilt_degree=(region.get("tilt") or {}).get("degree", 1),
            snake_axis=region.get("snake_axis"),
            summary=f"{region['x'][2]} × {region['y'][2]} points; "
                    f"X {region['x'][0]:g}..{region['x'][1]:g}; "
                    f"Y {region['y'][0]:g}..{region['y'][1]:g}",
        ) for region in plan["regions"]]
        self.owner._refresh_grid_region_list()

    def restore(self, sequence):
        metadata = sequence.metadata
        self.mode = ("raman" if GRID_KEY in metadata else
                     metadata.get(METADATA_KEY, {}).get("grid_mode", "imaging"))
        if self.mode not in ("imaging", "raman"):
            self.mode = "imaging"
        self.selector.blockSignals(True)
        self.selector.setCurrentIndex(self.selector.findData(self.mode))
        self.selector.blockSignals(False)
        self.refresh()

    def refresh(self):
        raman = self.mode == "raman"
        self.imaging.setVisible(True)
        self.editor.set_visible(raman)
        self.raman.setVisible(raman)
        index = self.panel.tab_wdg.indexOf(self.panel.grid_plan)
        checkbox = self.panel.tab_wdg.tabBar().tabButton(index, QTabBar.LeftSide)
        checkbox.setVisible(not raman)

    def change_mode(self):
        panel = self.panel
        if not panel._ready:
            return
        controller = getattr(self.owner, "_acquisition_controller", None)
        if controller and controller.busy:
            self.restore(panel.value())
            return
        desired = self.selector.currentData()
        current = panel.value()
        self.drafts[self.mode] = current
        draft = self.drafts.get(desired)
        metadata = deepcopy(current.metadata)
        metadata.pop(GRID_KEY, None)
        meta = metadata.setdefault(METADATA_KEY, {})
        meta["grid_mode"] = desired
        meta["targets"] = []
        meta.pop("wellplate_plan", None)
        if draft is not None:
            if GRID_KEY in draft.metadata:
                metadata[GRID_KEY] = deepcopy(draft.metadata[GRID_KEY])
            previous = draft.metadata.get(METADATA_KEY, {})
            meta["targets"] = deepcopy(previous.get("targets", []))
            if "raman_grid_editor" in previous:
                meta["raman_grid_editor"] = deepcopy(previous["raman_grid_editor"])
            if "wellplate_plan" in previous:
                meta["wellplate_plan"] = deepcopy(previous["wellplate_plan"])
        positions = draft.stage_positions if draft is not None else current.stage_positions
        grid = draft.grid_plan if draft is not None else None
        shutter_axes = tuple("g" if axis == "p" else axis
                             for axis in current.keep_shutter_open_across) if GRID_KEY in current.metadata else current.keep_shutter_open_across
        panel.setValue(current.replace(metadata=metadata, stage_positions=positions,
                                       grid_plan=grid, keep_shutter_open_across=shutter_axes))
        panel.tab_wdg.setCurrentWidget(panel.grid_plan)
