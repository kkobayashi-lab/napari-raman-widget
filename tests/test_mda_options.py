import unittest
from pathlib import Path
from types import MethodType, SimpleNamespace
from unittest.mock import patch

import numpy as np
from qtpy.QtCore import QEvent
from qtpy.QtWidgets import QMessageBox
from napari_raman_widget.widget import (
    DEFAULT_BEAM_CENTER_XY,
    HardwareWidget,
    _parse_raman_z_indices,
    _raman_free_autofocus_allowed,
    _spatial_yx,
)


class TestRamanZIndices(unittest.TestCase):
    def test_can_disable_spectra(self):
        for value in ("None", " none ", "OFF", "off"):
            with self.subTest(value=value):
                self.assertEqual(_parse_raman_z_indices(value), [])

    def test_numeric_values(self):
        self.assertEqual(_parse_raman_z_indices("0, 2"), [0, 2])

    def test_invalid_values(self):
        for value in ("", "  ", "0, nope"):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "Raman z indices"):
                    _parse_raman_z_indices(value)


class TestSpatialPointExtraction(unittest.TestCase):
    def test_uses_final_yx_coordinates_from_nd_point(self):
        point = np.arange(8)

        np.testing.assert_array_equal(_spatial_yx(point), [6, 7])

    def test_rejects_non_point_arrays(self):
        with self.assertRaisesRegex(ValueError, "one napari point"):
            _spatial_yx(np.zeros((2, 4)))


class TestRamanFreeAutofocus(unittest.TestCase):
    def test_disabled_autofocus_is_allowed(self):
        self.assertTrue(_raman_free_autofocus_allowed(False, "laser"))

    def test_software_autofocus_is_allowed(self):
        self.assertTrue(_raman_free_autofocus_allowed(True, "software"))

    def test_raman_assisted_autofocus_is_rejected(self):
        for autofocus_object in ("laser", "cell", "glass", "quartz"):
            with self.subTest(autofocus_object=autofocus_object):
                self.assertFalse(
                    _raman_free_autofocus_allowed(True, autofocus_object)
                )


class TestAutomaticHardwareDisconnect(unittest.TestCase):
    def test_disconnects_connected_hardware_on_shutdown(self):
        for core, collector in ((object(), None), (None, object())):
            with self.subTest(core=core, collector=collector):
                disconnected = []
                widget = SimpleNamespace(
                    core=core,
                    collector=collector,
                    disconnect=lambda: disconnected.append(True),
                )

                HardwareWidget._disconnect_on_shutdown(widget)

                self.assertEqual(disconnected, [True])

    def test_does_nothing_when_already_disconnected(self):
        disconnected = []
        widget = SimpleNamespace(
            core=None,
            collector=None,
            disconnect=lambda: disconnected.append(True),
        )

        HardwareWidget._disconnect_on_shutdown(widget)

        self.assertEqual(disconnected, [])

    def test_close_prompt_can_cancel_shutdown(self):
        window = object()
        disconnected = []
        event = SimpleNamespace(
            type=lambda: QEvent.Close,
            ignore=lambda: setattr(event, "ignored", True),
            ignored=False,
        )
        widget = SimpleNamespace(
            _napari_window=window,
            core=object(),
            collector=None,
            disconnect=lambda: disconnected.append(True),
        )

        with patch(
            "napari_raman_widget.widget.QMessageBox.question",
            return_value=QMessageBox.No,
        ):
            handled = HardwareWidget.eventFilter(widget, window, event)

        self.assertTrue(handled)
        self.assertTrue(event.ignored)
        self.assertEqual(disconnected, [])

    def test_confirming_close_disconnects_and_allows_shutdown(self):
        window = object()
        disconnected = []
        event = SimpleNamespace(type=lambda: QEvent.Close)
        widget = SimpleNamespace(
            _napari_window=window,
            core=None,
            collector=object(),
            disconnect=lambda: disconnected.append(True),
        )

        with patch(
            "napari_raman_widget.widget.QMessageBox.question",
            return_value=QMessageBox.Yes,
        ):
            handled = HardwareWidget.eventFilter(widget, window, event)

        self.assertFalse(handled)
        self.assertEqual(disconnected, [True])


class TestAutomaticBeamCentering(unittest.TestCase):
    def test_aims_at_default_center_and_holds_single_voltage_pair(self):
        class FakeGalvo:
            def __init__(self):
                self.stopped = False
                self.write_calls = []

            def stop(self):
                self.stopped = True

            def write(self, values, auto_start):
                self.write_calls.append((np.array(values), auto_start))

        galvo = FakeGalvo()
        requested_points = []
        widget = SimpleNamespace(
            daq=SimpleNamespace(galvo=galvo),
            transformer=object(),
            _pt_to_volts=lambda point: (
                requested_points.append(np.array(point))
                or np.array([[0.25, -0.5]])
            ),
        )

        volts = HardwareWidget._aim_beam_at_pixel(
            widget, *DEFAULT_BEAM_CENTER_XY
        )

        np.testing.assert_array_equal(requested_points[0], [512.0, 512.0])
        np.testing.assert_array_equal(volts, [0.25, -0.5])
        self.assertTrue(galvo.stopped)
        self.assertEqual(len(galvo.write_calls), 1)
        np.testing.assert_array_equal(
            galvo.write_calls[0][0], [0.25, -0.5]
        )
        self.assertTrue(galvo.write_calls[0][1])

    def test_rejects_invalid_transformer_output_without_writing(self):
        galvo = SimpleNamespace(
            stop=lambda: self.fail("galvo should not stop"),
            write=lambda *_args, **_kwargs: self.fail(
                "galvo should not write"
            ),
        )
        widget = SimpleNamespace(
            daq=SimpleNamespace(galvo=galvo),
            transformer=object(),
            _pt_to_volts=lambda _point: np.array([[np.nan, 0.0]]),
        )

        with self.assertRaisesRegex(ValueError, "finite X/Y voltage pair"):
            HardwareWidget._aim_beam_at_pixel(widget, 512, 512)


class _FakeStatusLabel:
    def __init__(self):
        self.text = ""

    def setText(self, text):
        self.text = text


class _FakeCheckBox:
    def __init__(self, checked):
        self._checked = checked

    def isChecked(self):
        return self._checked


class _FakeCombo:
    def __init__(self):
        self.items = []
        self.current = ""
        self.enabled = False

    def blockSignals(self, _blocked):
        pass

    def clear(self):
        self.items.clear()
        self.current = ""

    def addItem(self, item):
        self.items.append(item)
        if not self.current:
            self.current = item

    def addItems(self, items):
        for item in items:
            self.addItem(item)

    def setCurrentText(self, item):
        self.current = item

    def currentText(self):
        return self.current

    def setEnabled(self, enabled):
        self.enabled = enabled

    def count(self):
        return len(self.items)

    def itemText(self, index):
        return self.items[index]


class _FakeObjectiveCore:
    def __init__(self, available=("10X", "20X"), current="20X", state=3):
        self.available = list(available)
        self.current = current
        self.state = state
        self.set_calls = []
        self.read_calls = []
        self.waited = False

    def getAvailableConfigs(self, group):
        assert group == "Objective"
        self.read_calls.append("available")
        return self.available

    def getCurrentConfig(self, group):
        assert group == "Objective"
        self.read_calls.append("current")
        return self.current

    def getState(self, device):
        assert device == "Objective"
        self.read_calls.append("state")
        return self.state

    def setConfig(self, group, preset):
        assert group == "Objective"
        self.current = preset
        self.set_calls.append(preset)

    def waitForSystem(self):
        self.waited = True


class TestObjectiveSelection(unittest.TestCase):
    def _widget(self):
        combo = _FakeCombo()
        combo.addItem("(connect first)")
        widget = SimpleNamespace(
            core=_FakeObjectiveCore(),
            objective_combo=combo,
            vandermonde=(np.ones((3, 2)), 1),
            vandermonde_objective="20X",
            vandermonde_path="old.json",
            status=_FakeStatusLabel(),
        )
        for name in (
            "_set_objective_combo",
            "_refresh_objective_combo",
            "_current_objective",
        ):
            setattr(
                widget,
                name,
                MethodType(getattr(HardwareWidget, name), widget),
        )
        return widget

    def test_refresh_reads_numeric_objective_state_not_preset_label(self):
        widget = self._widget()

        current = widget._refresh_objective_combo()

        self.assertEqual(current, "3")
        self.assertEqual(widget.core.read_calls, ["state"])
        self.assertEqual(widget.objective_combo.currentText(), "3")
        self.assertFalse(widget.objective_combo.enabled)

    def test_index_refresh_never_commands_objective_turret(self):
        widget = self._widget()
        widget.core.state = 0

        current = widget._current_objective()

        self.assertEqual(current, "0")
        self.assertEqual(widget.objective_combo.currentText(), "0")
        self.assertEqual(widget.core.set_calls, [])
        self.assertFalse(widget.core.waited)


class TestAutomaticDatasetGeneration(unittest.TestCase):
    def _widget(
        self, *, reason="completed", checked=True, has_raman=True,
        status_api=True,
    ):
        generated = []
        mda = SimpleNamespace()
        if status_api:
            mda.status = lambda: SimpleNamespace(finish_reason=reason)
        widget = SimpleNamespace(
            _raman_mda_pending=True,
            _raman_mda_canceled=False,
            _raman_mda_writer=SimpleNamespace(_path=Path("data/run_1")),
            _raman_mda_batch=True,
            _raman_mda_has_raman=has_raman,
            auto_dataset_check=_FakeCheckBox(checked),
            status=_FakeStatusLabel(),
            core=SimpleNamespace(mda=mda),
            _generate_dataset=lambda run_dir, batch: generated.append(
                (run_dir, batch)
            ),
        )
        return widget, generated

    def test_successful_run_generates_dataset_when_enabled(self):
        widget, generated = self._widget()

        HardwareWidget._on_raman_mda_finished(widget, None)

        self.assertEqual(generated, [(str(Path("data/run_1")), True)])
        self.assertFalse(widget._raman_mda_pending)

    def test_disabled_option_does_not_generate(self):
        widget, generated = self._widget(checked=False)

        HardwareWidget._on_raman_mda_finished(widget, None)

        self.assertEqual(generated, [])
        self.assertEqual(widget.status.text, "Status: MDA finished OK")

    def test_canceled_or_errored_run_does_not_generate(self):
        for reason in ("canceled", "errored"):
            with self.subTest(reason=reason):
                widget, generated = self._widget(reason=reason)

                HardwareWidget._on_raman_mda_finished(widget, None)

                self.assertEqual(generated, [])
                self.assertIn(
                    "automatic dataset generation skipped", widget.status.text
                )

    def test_raman_free_run_does_not_generate(self):
        widget, generated = self._widget(has_raman=False)

        HardwareWidget._on_raman_mda_finished(widget, None)

        self.assertEqual(generated, [])
        self.assertIn("Raman-free MDA finished", widget.status.text)

    def test_runtime_without_status_api_uses_completion_signal(self):
        widget, generated = self._widget(status_api=False)

        HardwareWidget._on_raman_mda_finished(widget, None)

        self.assertEqual(generated, [(str(Path("data/run_1")), True)])

    def test_runtime_without_status_api_uses_cancellation_signal(self):
        widget, generated = self._widget(status_api=False)

        HardwareWidget._on_raman_mda_canceled(widget, None)
        HardwareWidget._on_raman_mda_finished(widget, None)

        self.assertEqual(generated, [])
        self.assertIn("MDA canceled", widget.status.text)


if __name__ == "__main__":
    unittest.main()
