import pytest
from qtpy.QtWidgets import QApplication, QPushButton, QWidget

from napari_raman_widget.autofocus_panel import AutofocusPanel
from napari_raman_widget.unified_mda import build_extension_controls


@pytest.fixture
def panel():
    app = QApplication.instance() or QApplication([])
    owner = QWidget()
    owner.browse_cellpose_model = lambda: None
    owner.browse_tracking_cfg = lambda: None
    owner.open_acquisition_viewer = lambda: None
    owner._toggle_autofocus_fields = lambda *_: None
    owner._toggle_seg_track_fields = lambda *_: None
    owner.laser_autofocus_btn = QPushButton("Laser autofocus at FOV center")
    build_extension_controls(owner)
    panel = AutofocusPanel(owner)
    yield owner, panel, app
    owner.close()


def test_settings_sync_both_directions(panel):
    owner, widget, _ = panel
    widget.controls["mda_af_range_input"].setValue(8.5)
    assert owner.mda_af_range_input.value() == 8.5
    assert "100" in widget.limit_label.text()
    assert owner.mda_fine_range_input.value() == 2.5
    assert widget.controls["mda_fine_range_input"].isReadOnly()
    assert widget.controls["mda_fine_pts_input"].isReadOnly()
    assert widget.controls["mda_fine_pts_input"].value() == 11
    widget.controls["mda_search_pts_input"].setValue(2)
    assert owner.mda_search_pts_input.value() == 3


def test_curve_replaced_on_failure_and_sorted_by_z(panel):
    _, widget, app = panel
    result = dict(start_z=10, best_z=11, error=None, samples=[
        dict(z=12, score=3, phase="coarse"),
        dict(z=10, score=1, phase="coarse"),
        dict(z=11, score=5, phase="fine"),
    ])
    widget.show_result(result)
    assert list(widget.axes.lines[0].get_xdata()) == [10, 12]
    assert len(widget.axes.lines) == 4
    assert "Z=11.000" in widget.result_label.text()
    widget.show_result(dict(start_z=15, best_z=None, error="No peak", samples=[
        dict(z=16, score=2, phase="coarse"),
    ]))
    assert len(widget.axes.lines) == 2
    assert "failed: No peak" in widget.result_label.text()
    widget.canvas.draw()
    app.processEvents()
