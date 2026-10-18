"""Shared laser autofocus controls and the latest measured focus curve."""
import numpy as np
from qtpy.QtCore import Slot
from qtpy.QtWidgets import QDoubleSpinBox, QFormLayout, QLabel, QSpinBox, QVBoxLayout, QWidget
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg


class AutofocusPanel(QWidget):
    def __init__(self, owner):
        super().__init__(owner)
        layout = QVBoxLayout(self)
        note = QLabel("Search settings are shared with MDA autofocus.")
        note.setWordWrap(True)
        layout.addWidget(note)
        form = QFormLayout()
        layout.addLayout(form)
        self.controls = {}
        for name, label in (
            ("mda_af_range_input", "Initial coarse range (± µm)"),
            ("mda_search_pts_input", "Coarse samples"),
            ("mda_fine_range_input", "Automatic fine range (± µm)"),
            ("mda_fine_pts_input", "Automatic fine samples (0.5 µm step)"),
        ):
            source = getattr(owner, name)
            control = QDoubleSpinBox() if isinstance(source, QDoubleSpinBox) else QSpinBox()
            control.setRange(source.minimum(), source.maximum())
            if isinstance(control, QDoubleSpinBox):
                control.setDecimals(source.decimals())
            control.setReadOnly(source.isReadOnly())
            control.setValue(source.value())
            if not source.isReadOnly():
                control.valueChanged.connect(source.setValue)
            source.valueChanged.connect(control.setValue)
            form.addRow(label, control)
            self.controls[name] = control
        self.limit_label = QLabel()
        self.limit_label.setWordWrap(True)
        def update_limit(value):
            self.limit_label.setText("Search extends in both directions up to ±100 µm from starting Z. Requires a bracketed peak with score > 800,000. Fine spacing is 0.5 µm; range covers ±one coarse step, rounded up.")
        owner.mda_af_range_input.valueChanged.connect(update_limit)
        update_limit(owner.mda_af_range_input.value())
        layout.addWidget(self.limit_label)
        layout.addWidget(owner.laser_autofocus_btn)
        self.result_label = QLabel("No autofocus run yet.")
        self.result_label.setWordWrap(True)
        layout.addWidget(self.result_label)
        self.figure = Figure(figsize=(5, 2.8), layout="constrained")
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setMinimumHeight(230)
        layout.addWidget(self.canvas)
        self.axes = self.figure.add_subplot(111)
        self.last_result = None
        self._label_axes()

    def _label_axes(self):
        self.axes.set_xlabel("Stage Z (µm)")
        self.axes.set_ylabel("Focus score (camera counts)")
        self.axes.grid(True, alpha=0.25)

    @Slot(object)
    def show_result(self, result):
        self.last_result = result
        self.axes.clear()
        for phase, color in (("coarse", "tab:blue"), ("fine", "tab:orange")):
            samples = [s for s in result["samples"] if s["phase"] == phase]
            if samples:
                samples.sort(key=lambda s: s["z"])
                self.axes.plot([s["z"] for s in samples], [s["score"] for s in samples],
                               "o-", color=color, label=phase.capitalize(), markersize=4)
        self.axes.axvline(result["start_z"], color="gray", linestyle=":", label="Starting Z")
        best = result.get("best_z")
        if best is not None and np.isfinite(best):
            self.axes.axvline(best, color="tab:green", linestyle="--", label="Selected Z")
        self._label_axes()
        self.axes.legend(fontsize="small")
        if result.get("error"):
            text = f"Last run failed: {result['error']}"
        else:
            text = f"Last run: Z={best:.3f} µm (ΔZ={best - result['start_z']:+.3f} µm)"
        self.result_label.setText(text)
        self.canvas.draw_idle()
