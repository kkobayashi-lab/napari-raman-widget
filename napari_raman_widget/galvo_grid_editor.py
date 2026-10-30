"""MDA voltage-grid controls, relative to the saved hardware center."""
from qtpy.QtWidgets import QWidget, QFormLayout, QLabel, QDoubleSpinBox, QSpinBox, QComboBox


class GalvoGridEditor(QWidget):
    def __init__(self, modes):
        super().__init__()
        self.modes = modes
        form = QFormLayout(self)
        self.center = QLabel()
        self.center.setWordWrap(True)
        form.addRow("Saved center", self.center)
        self.ranges, self.counts = [], []
        for axis in ("X", "Y"):
            span = QDoubleSpinBox()
            span.setDecimals(4)
            span.setRange(0, 1.6)
            span.setSingleStep(.01)
            span.setValue(.1)
            span.setPrefix("± ")
            span.setSuffix(" V")
            count = QSpinBox()
            count.setRange(1, 1000)
            count.setValue(20)
            form.addRow(f"{axis} voltage range", span)
            form.addRow(f"{axis} points", count)
            self.ranges.append(span)
            self.counts.append(count)
        self.order = QComboBox()
        for label, value in (("Row-wise", "row_wise"), ("Row-wise snake", "row_wise_snake"),
                             ("Column-wise", "column_wise"), ("Column-wise snake", "column_wise_snake"),
                             ("Spiral", "spiral")):
            self.order.addItem(label, value)
        self.order.setCurrentIndex(1)
        form.addRow("Traversal", self.order)
        self.repeats = QSpinBox()
        original = modes.owner.mda_repeats_input
        self.repeats.setRange(original.minimum(), original.maximum())
        self.repeats.setValue(original.value())
        self.repeats.valueChanged.connect(original.setValue)
        original.valueChanged.connect(self.repeats.setValue)
        form.addRow("Spectra per point", self.repeats)
        self.shift = QDoubleSpinBox()
        self.shift.setRange(-10000, 10000)
        self.shift.setValue(1003)
        self.shift.setSuffix(" cm⁻¹")
        form.addRow("Map Raman shift", self.shift)
        self.summary = QLabel()
        self.summary.setWordWrap(True)
        form.addRow(self.summary)
        note = QLabel("Ranges are offsets from Hardware Control → Save galvo center. "
                      "Stage XY stays fixed; the Positions list is not scanned. "
                      "Autofocus and tracking are ignored in this mode. No stage-grid Z setting is needed; "
                      "current Z is used unless you enable a Z stack. "
                      "Use the RM channel and Run MDA. A one-point axis samples its center.")
        note.setWordWrap(True)
        form.addRow(note)
        for control in (*self.ranges, *self.counts, self.shift):
            control.valueChanged.connect(self.changed)
        self.order.currentIndexChanged.connect(self.changed)
        button = getattr(modes.owner, "save_galvo_center_btn", None)
        if button is not None:
            button.clicked.connect(self.changed)
        self.update_summary()

    def settings(self):
        return dict(half_ranges=[v.value() for v in self.ranges],
                    counts=[v.value() for v in self.counts], order=self.order.currentData(),
                    shift_cm=self.shift.value())

    def restore(self, settings):
        if not settings:
            return
        for controls, key in ((self.ranges, "half_ranges"), (self.counts, "counts")):
            for control, value in zip(controls, settings[key]):
                control.blockSignals(True)
                control.setValue(value)
                control.blockSignals(False)
        self.order.blockSignals(True)
        self.order.setCurrentIndex(max(0, self.order.findData(settings.get("order"))))
        self.order.blockSignals(False)
        self.shift.blockSignals(True)
        self.shift.setValue(settings.get("shift_cm", 1003))
        self.shift.blockSignals(False)
        self.update_summary()

    def plan(self):
        center = getattr(self.modes.owner, "_saved_galvo_center", None)
        center = list(map(float, center)) if center is not None else None
        region = {axis: [c - span.value(), c + span.value(), count.value()]
                  for axis, c, span, count in zip(("x", "y"), center or (0, 0), self.ranges, self.counts)}
        region.update(order=self.order.currentData(), z=None, tilt=None)
        return dict(version=1, backend="galvo", center_volts=center, regions=[region])

    def update_summary(self):
        plan = self.plan()
        center = plan["center_volts"]
        self.center.setText("Not saved — set and save it in Hardware Control" if center is None else
                            f"X {center[0]:.4f} V; Y {center[1]:.4f} V")
        region = plan["regions"][0]
        from raman_mda_engine.grid_plan import validate_grid
        try:
            validate_grid(plan)
            self.summary.setText(f"{self.counts[0].value() * self.counts[1].value():,} points. " +
                "; ".join(f"{axis.upper()} {region[axis][0]:.4f} to {region[axis][1]:.4f} V" for axis in ("x", "y")))
        except ValueError as error:
            self.summary.setText(str(error))

    def changed(self, *_):
        self.update_summary()
        self.modes.panel.valueChanged.emit()
