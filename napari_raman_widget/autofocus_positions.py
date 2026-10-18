"""Position-owned references and compact status in the native position table."""
from copy import deepcopy
import uuid

from qtpy.QtCore import Qt
from qtpy.QtWidgets import QLabel, QTableWidgetItem
from useq import MDASequence, Position

from .autofocus_references import REFERENCE_KEY, reference, choose_references, reference_action
from .experiment import position_id


def install_reference_column(panel, owner):
    table = panel.stage_positions.table()
    panel._reference_column = table.columnCount()
    table.insertColumn(panel._reference_column)
    table.setHorizontalHeaderItem(panel._reference_column, QTableWidgetItem("Autofocus"))
    panel._reference_summary = QLabel()
    panel.stage_positions.layout().addWidget(panel._reference_summary)

    def clicked(row, col):
        if col != panel._reference_column:
            return
        selector = table._get_selector_col()
        if not table._is_row_checked(row, selector):
            owner.status.setText("Status: enable this position before selecting its reference")
            return
        index = sum(table._is_row_checked(r, selector) for r in range(row))
        reference_action(owner, lambda obj: choose_references(obj, position_index=index))

    table.cellClicked.connect(clicked)


def row_position(table, row):
    values = table.rowData(row, exclude_hidden_cols=True)
    values.pop("af", None)
    return Position(**values)


def synchronize_reference_positions(panel, sequence):
    table = panel.stage_positions.table()
    seq_col = table.indexOf(panel.stage_positions.SEQ)
    previous = table.blockSignals(True)
    try:
        for row in range(table.rowCount()):
            pos = row_position(table, row)
            sub = pos.sequence or MDASequence()
            metadata = deepcopy(sub.metadata)
            metadata.setdefault("position_identity", position_id(pos) or uuid.uuid4().hex)
            if REFERENCE_KEY in metadata and reference(pos) is None:
                # Imported/calibrated coordinates are rounded by the native
                # spinboxes. Bind the reference to that displayed position,
                # without treating a real table edit as the same location.
                stage = [pos.x, pos.y, pos.z]
                original = metadata[REFERENCE_KEY]["stage"]
                columns = (panel.stage_positions.X, panel.stage_positions.Y, panel.stage_positions.Z)
                tolerances = [0.5 * 10 ** -table.cellWidget(row, table.indexOf(col)).decimals()
                              for col in columns]
                if all(a == b or (a is not None and b is not None and abs(a - b) <= tol)
                       for a, b, tol in zip(original, stage, tolerances)):
                    metadata[REFERENCE_KEY]["stage"] = stage
                else:
                    metadata.pop(REFERENCE_KEY)
            if metadata != sub.metadata:
                widget = table.cellWidget(row, seq_col)
                blocked = widget.blockSignals(True)
                widget.setValue(sub.replace(metadata=metadata))
                widget.blockSignals(blocked)
    finally:
        table.blockSignals(previous)
    if table.rowCount() and panel.stage_positions._plate_plan is None:
        sequence = sequence.replace(stage_positions=panel.stage_positions.value())
    return sequence


def refresh_reference_column(panel, owner):
    table = panel.stage_positions.table()
    previous = table.blockSignals(True)
    count = total = 0
    try:
        for row in range(table.rowCount()):
            ref = reference(row_position(table, row))
            item = QTableWidgetItem("● Set" if ref else "— Skipped")
            item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            if ref:
                y, x = [a * b for a, b in zip(ref["point"], ref["image_shape"])]
                tip = f"Reference: x={x:.1f}, y={y:.1f} px\nMethod: {owner.mda_af_mode_combo.currentText()}\nClick to edit"
                if "stage_xy_um" in ref:
                    x, y = ref["stage_xy_um"]
                    tip = f"Inherited reference: stage X={x:.3f}, Y={y:.3f} um\nClick to replace using current-position images"
                if not any(box.isChecked() for box in owner.mda_af_axes.values()):
                    tip += "\nAutofocus is globally off; enable an axis in the Autofocus tab."
            else:
                tip = "No reference selected; autofocus will be skipped. Click to set one."
            item.setToolTip(tip)
            table.setItem(row, panel._reference_column, item)
            if table._is_row_checked(row, table._get_selector_col()):
                total += 1
                count += bool(ref)
    finally:
        table.blockSignals(previous)
    panel._reference_summary.setText(f"Autofocus: {count} set · {total - count} skipped")
