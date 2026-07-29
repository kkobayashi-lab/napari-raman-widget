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
        signals = [_FakeSignal() for _ in range(7)]
        calls = []
        viewer = SimpleNamespace(
            raman_t_combo=_control(currentIndexChanged=signals[0]),
            raman_index=_control(valueChanged=signals[1]),
            raman_z_combo=_control(currentIndexChanged=signals[2]),
            cell_index=_control(valueChanged=signals[3]),
            channel_combo=_control(currentIndexChanged=signals[4]),
            stitched_check=_control(toggled=signals[5]),
            preview_size=_control(currentIndexChanged=signals[6]),
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


if __name__ == "__main__":
    unittest.main()
