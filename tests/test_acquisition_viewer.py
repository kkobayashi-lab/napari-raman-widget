import json
import unittest
from types import SimpleNamespace

from napari_raman_widget.acquisition_viewer import (
    LargeAcquisitionViewerWindow,
    vandermonde_objectives,
)


class _FakeSignal:
    def __init__(self):
        self.slots = []

    def connect(self, slot):
        self.slots.append(slot)

    def emit(self):
        for slot in self.slots:
            slot()


class _FakeTimer:
    def __init__(self):
        self.starts = 0

    def start(self):
        self.starts += 1


class _FakeCombo:
    def __init__(self):
        self.items = []
        self.current_index = -1

    def currentData(self):
        if 0 <= self.current_index < len(self.items):
            return self.items[self.current_index][1]
        return None

    def blockSignals(self, _blocked):
        pass

    def clear(self):
        self.items.clear()
        self.current_index = -1

    def addItem(self, text, data):
        self.items.append((text, data))
        if self.current_index < 0:
            self.current_index = 0

    def setCurrentIndex(self, index):
        self.current_index = index


def _control(**signals):
    return SimpleNamespace(**signals)


def test_vandermonde_objectives_are_read_without_hardware(tmp_path):
    model = tmp_path / "vandermonde.json"
    model.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "objectives": {
                    "1": {"degree": 1, "C": []},
                    "3": {"degree": 1, "C": []},
                },
            }
        ),
        encoding="utf-8",
    )

    assert vandermonde_objectives(model) == ["1", "3"]


class TestAcquisitionViewerAutoRefresh(unittest.TestCase):
    def test_every_view_control_schedules_a_refresh(self):
        signals = [_FakeSignal() for _ in range(8)]
        calls = []
        viewer = SimpleNamespace(
            raman_t_combo=_control(currentIndexChanged=signals[0]),
            raman_index=_control(valueChanged=signals[1]),
            raman_z_combo=_control(currentIndexChanged=signals[2]),
            cell_layer_combo=_control(currentIndexChanged=signals[3]),
            cell_index=_control(valueChanged=signals[4]),
            channel_combo=_control(currentIndexChanged=signals[5]),
            stitched_check=_control(toggled=signals[6]),
            preview_size=_control(currentIndexChanged=signals[7]),
            _schedule_auto_refresh=lambda *_args: calls.append(True),
        )

        LargeAcquisitionViewerWindow._connect_auto_refresh(viewer)
        for signal in signals:
            signal.emit()

        self.assertEqual(len(calls), len(signals))

    def test_refresh_is_scheduled_only_for_an_indexed_viewer(self):
        timer = _FakeTimer()
        viewer = SimpleNamespace(
            _auto_refresh_suspended=False,
            acquisition=object(),
            geometry=object(),
            _auto_refresh_timer=timer,
        )

        LargeAcquisitionViewerWindow._schedule_auto_refresh(viewer)
        self.assertEqual(timer.starts, 1)

        viewer._auto_refresh_suspended = True
        LargeAcquisitionViewerWindow._schedule_auto_refresh(viewer)
        self.assertEqual(timer.starts, 1)

    def test_cell_layer_selector_uses_saved_layer_indices(self):
        combo = _FakeCombo()
        range_updates = []
        layers = (
            SimpleNamespace(
                layer_index=0,
                name="Type A",
                designation="cell:Type A",
                cell_count=2,
            ),
            SimpleNamespace(
                layer_index=1,
                name="Type B",
                designation="cell:Type B",
                cell_count=1,
            ),
        )
        viewer = SimpleNamespace(
            cell_layer_combo=combo,
            acquisition=SimpleNamespace(
                raman_cell_layers=lambda *_args, **_kwargs: layers
            ),
            raman_index=SimpleNamespace(value=lambda: 0),
            _selected_raman_t=lambda: 0,
            _selected_raman_z=lambda: 0,
            _update_cell_range=lambda: range_updates.append(True),
        )

        LargeAcquisitionViewerWindow._update_cell_layers(viewer)

        self.assertEqual(
            combo.items,
            [
                ("0: Type A (2 cells)", "cell:Type A"),
                ("1: Type B (1 cell)", "cell:Type B"),
            ],
        )
        self.assertEqual(range_updates, [True])


if __name__ == "__main__":
    unittest.main()
