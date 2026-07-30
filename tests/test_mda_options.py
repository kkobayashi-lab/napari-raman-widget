import unittest
from pathlib import Path
import tempfile
from types import MethodType, SimpleNamespace
from unittest.mock import patch

import numpy as np
from qtpy.QtCore import QEvent
from qtpy.QtWidgets import QMessageBox
from napari_raman_widget.widget import (
    DEFAULT_BEAM_CENTER_XY,
    HardwareWidget,
    _cell_layer_metadata,
    _cellpose_diameter,
    _parse_raman_z_indices,
    _raman_free_autofocus_allowed,
    _spatial_yx,
)


class TestCellLayerMetadata(unittest.TestCase):
    def test_legacy_and_named_cell_sources_are_serialized(self):
        sources = [
            SimpleNamespace(
                name="cells",
                role="cell",
                display_name="Type A",
                source_id="a",
                color="#aa0000ff",
            ),
            SimpleNamespace(
                name="cell:Type B",
                role="cell",
                display_name="Type B",
                source_id="b",
                color="#0066ccff",
            ),
            SimpleNamespace(name="autofocus", role="autofocus"),
        ]

        self.assertEqual(
            _cell_layer_metadata(sources),
            [
                {
                    "id": "a",
                    "name": "Type A",
                    "source_name": "cells",
                    "color": "#aa0000ff",
                },
                {
                    "id": "b",
                    "name": "Type B",
                    "source_name": "cell:Type B",
                    "color": "#0066ccff",
                },
            ],
        )


class TestCellposeDiameter(unittest.TestCase):
    def test_zero_uses_model_training_diameter(self):
        self.assertIsNone(_cellpose_diameter(0))

    def test_positive_diameter_is_preserved(self):
        self.assertEqual(_cellpose_diameter(15.5), 15.5)


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


class TestCellposeModelSelection(unittest.TestCase):
    def test_builtin_model_name_is_returned(self):
        combo = SimpleNamespace(
            currentData=lambda: "cpsam_v2",
            currentText=lambda: "cpsam_v2",
        )

        self.assertEqual(
            HardwareWidget._selected_cellpose_model(combo),
            "cpsam_v2",
        )

    def test_custom_model_path_is_validated_and_resolved(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "my-cellpose-model"
            model.write_bytes(b"weights")
            combo = SimpleNamespace(
                currentData=lambda: str(model),
                currentText=lambda: f"Custom: {model.name}",
            )

            selected = HardwareWidget._selected_cellpose_model(combo)

        self.assertEqual(selected, str(model.resolve()))

    def test_missing_custom_model_is_rejected(self):
        model = Path("missing-cellpose-model").resolve()
        combo = SimpleNamespace(
            currentData=lambda: str(model),
            currentText=lambda: f"Custom: {model.name}",
        )

        with self.assertRaisesRegex(FileNotFoundError, "model file not found"):
            HardwareWidget._selected_cellpose_model(combo)


class TestOutputPaths(unittest.TestCase):
    def test_relative_paths_follow_latest_output_folder(self):
        selected = {"path": "first"}
        widget = SimpleNamespace(
            out_path=SimpleNamespace(text=lambda: selected["path"])
        )

        first = HardwareWidget._output_path(widget, "reference/test.zarr")
        selected["path"] = "second"
        second = HardwareWidget._output_path(widget, "reference/test.zarr")

        self.assertEqual(first, Path("first/reference/test.zarr"))
        self.assertEqual(second, Path("second/reference/test.zarr"))

    def test_absolute_output_paths_are_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            absolute = Path(directory) / "run"
            widget = SimpleNamespace(
                out_path=SimpleNamespace(text=lambda: "ignored")
            )

            result = HardwareWidget._output_path(widget, absolute)

        self.assertEqual(result, absolute)


class TestMdaTimePreview(unittest.TestCase):
    @staticmethod
    def _estimate():
        return SimpleNamespace(
            duration_seconds=3.2,
            scheduled_wait_seconds=0,
            raman_points=2,
            imaging_frames=1,
        )

    def test_overwrite_preview_names_folder_and_warns_about_deletion(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            with patch(
                "napari_raman_widget.widget.QMessageBox.warning"
            ) as warning:
                HardwareWidget._show_mda_time_preview(
                    SimpleNamespace(),
                    self._estimate(),
                    raman_exposure_ms=1000,
                    delay_seconds=0,
                    output_dir=output,
                    overwrite=True,
                )

        warning.assert_called_once()
        message = warning.call_args.args[2]
        self.assertIn("folder will be overwritten", message)
        self.assertIn(str(output.resolve()), message)
        self.assertIn("permanently deleted", message)

    def test_normal_preview_remains_informational(self):
        with patch(
            "napari_raman_widget.widget.QMessageBox.information"
        ) as information:
            HardwareWidget._show_mda_time_preview(
                SimpleNamespace(),
                self._estimate(),
                raman_exposure_ms=1000,
                delay_seconds=0,
            )

        information.assert_called_once()
        self.assertNotIn(
            "folder will be overwritten",
            information.call_args.args[2],
        )


class TestMdaAxisOrder(unittest.TestCase):
    def test_selection_preparation_preserves_selected_axis_order(self):
        from useq import MDASequence

        original = MDASequence(
            axis_order="tpzc",
            channels=("BF", "GFP"),
            z_plan={"relative": (-1.0, 0.0, 1.0)},
            time_plan={"interval": 2.0, "loops": 3},
        )
        updated = []
        settings = SimpleNamespace(
            value=lambda: original,
            setValue=lambda sequence: updated.append(sequence),
        )
        widget = SimpleNamespace(
            core=SimpleNamespace(stopSequenceAcquisition=lambda: None),
            main_window=object(),
        )

        with patch(
            "raman_mda_engine.utils.get_mda_widget_from_napari",
            return_value=settings,
        ):
            HardwareWidget._prepare_for_selection(widget)

        self.assertEqual(len(updated), 1)
        self.assertEqual(updated[0].axis_order, original.axis_order)
        self.assertEqual(updated[0].time_plan.loops, 1)


class TestRamanFreeAutofocus(unittest.TestCase):
    def test_disabled_autofocus_is_allowed(self):
        self.assertTrue(_raman_free_autofocus_allowed(False, "laser"))

    def test_software_autofocus_is_allowed(self):
        self.assertTrue(_raman_free_autofocus_allowed(True, "software"))

    def test_raman_assisted_autofocus_is_allowed(self):
        for autofocus_object in (
            "laser", "software", "cell", "glass", "quartz"
        ):
            with self.subTest(autofocus_object=autofocus_object):
                self.assertTrue(
                    _raman_free_autofocus_allowed(True, autofocus_object)
                )

    def test_unknown_autofocus_is_rejected(self):
        self.assertFalse(_raman_free_autofocus_allowed(True, "unknown"))


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


class TestHardwareControls(unittest.TestCase):
    def test_click_to_center_stays_armed_after_each_click(self):
        class FakeButton:
            @staticmethod
            def setChecked(_checked):
                raise AssertionError("persistent mode should not disarm")

        moved = []
        widget = SimpleNamespace(
            click_center_btn=FakeButton(),
            _move_clicked_to_center=lambda yx: moved.append(yx),
        )
        event = SimpleNamespace(button=1, position=(4, 120, 240))

        HardwareWidget._click_center_cb(widget, None, event)

        np.testing.assert_array_equal(moved[0], [120.0, 240.0])

    def test_shutter_uses_rm_and_restores_previous_imaging_channel(self):
        class FakeCore:
            def __init__(self):
                self.current = "GFP"
                self.set_calls = []
                self.wait_calls = []

            def getCurrentConfig(self, group):
                self.assert_channel_group(group)
                return self.current

            def setConfig(self, group, config):
                self.assert_channel_group(group)
                self.current = config
                self.set_calls.append(config)

            def waitForConfig(self, group, config):
                self.assert_channel_group(group)
                self.wait_calls.append(config)

            @staticmethod
            def assert_channel_group(group):
                if group != "Channel":
                    raise AssertionError(group)

        core = FakeCore()
        widget = SimpleNamespace(
            core=core,
            status=_FakeStatusLabel(),
            _shutter_return_channel="BF",
        )

        HardwareWidget._set_laser_shutter(widget, True)
        HardwareWidget._set_laser_shutter(widget, False)

        self.assertEqual(core.set_calls, ["RM", "GFP"])
        self.assertEqual(core.wait_calls, ["RM", "GFP"])
        self.assertEqual(widget._shutter_return_channel, "GFP")
        self.assertIn("closed (GFP)", widget.status.text)

    def test_nd_filter_uses_autofocus_digital_io_and_preserves_shutter(self):
        class FakeCore:
            def __init__(self):
                self.state = 1
                self.set_calls = []
                self.wait_calls = []

            @staticmethod
            def getLoadedDevices():
                return ("Camera", "DigitalIO")

            def getState(self, device):
                self.assert_filter_device(device)
                return self.state

            def setState(self, device, state):
                self.assert_filter_device(device)
                self.state = state
                self.set_calls.append(state)

            def waitForDevice(self, device):
                self.assert_filter_device(device)
                self.wait_calls.append(device)

            @staticmethod
            def assert_filter_device(device):
                if device != "DigitalIO":
                    raise AssertionError(device)

        core = FakeCore()
        widget = SimpleNamespace(
            core=core,
            status=_FakeStatusLabel(),
        )

        HardwareWidget._set_nd_filter(widget, False)
        HardwareWidget._set_nd_filter(widget, True)

        self.assertEqual(core.set_calls, [3, 1])
        self.assertEqual(core.wait_calls, ["DigitalIO", "DigitalIO"])
        self.assertIn("open (removed)", widget.status.text)

    def test_viewer_click_points_laser_and_disarms(self):
        class FakeButton:
            def __init__(self):
                self.checked = True

            def setChecked(self, checked):
                self.checked = checked

        aimed = []
        widget = SimpleNamespace(
            click_laser_btn=FakeButton(),
            _aim_beam_at_pixel=lambda x, y: (
                aimed.append((x, y)) or np.array([0.25, -0.5])
            ),
            status=_FakeStatusLabel(),
        )
        event = SimpleNamespace(button=1, position=(4, 120, 240))

        HardwareWidget._click_laser_cb(widget, None, event)

        self.assertEqual(aimed, [(240.0, 120.0)])
        self.assertFalse(widget.click_laser_btn.checked)
        self.assertIn("laser pointed at (120,240)", widget.status.text)

    def test_stage_drag_follows_calibrated_stage_direction(self):
        moves = []
        core = SimpleNamespace(
            mda=SimpleNamespace(is_running=lambda: False),
            getXYStageDevice=lambda: "XYStage",
            deviceBusy=lambda _device: False,
            setRelativeXYPosition=lambda dx, dy: moves.append((dx, dy)),
        )
        widget = SimpleNamespace(
            _stage_drag_active=True,
            _stage_drag_anchor_yx=np.array([100.0, 100.0]),
            _stage_drag_current_yx=np.array([100.0, 200.0]),
            core=core,
            vandermonde=(object(), 1),
            stage_drag_speed_input=SimpleNamespace(value=lambda: 50.0),
            drag_stage_btn=SimpleNamespace(setChecked=lambda _checked: None),
            status=_FakeStatusLabel(),
        )

        with patch(
            "cns_control.utils.apply_vandermonde_model",
            side_effect=lambda offset, _coefficients, _degree: np.asarray(
                offset, dtype=float
            ),
        ):
            HardwareWidget._stage_drag_tick(widget)

        self.assertEqual(len(moves), 1)
        self.assertAlmostEqual(moves[0][0], 5.0)
        self.assertAlmostEqual(moves[0][1], 0.0)
        self.assertIn("dX=+5.00", widget.status.text)

    def test_stage_drag_dead_zone_does_not_move(self):
        widget = SimpleNamespace(
            _stage_drag_active=True,
            _stage_drag_anchor_yx=np.array([100.0, 100.0]),
            _stage_drag_current_yx=np.array([105.0, 105.0]),
            core=SimpleNamespace(
                mda=SimpleNamespace(is_running=lambda: False),
                getXYStageDevice=lambda: self.fail(
                    "dead-zone drag should not query the stage"
                ),
            ),
        )

        HardwareWidget._stage_drag_tick(widget)


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


class TestMdaCompletion(unittest.TestCase):
    def _widget(
        self, *, reason="completed", status_api=True,
        writer_status="completed",
    ):
        mda = SimpleNamespace()
        if status_api:
            mda.status = lambda: SimpleNamespace(finish_reason=reason)
        writer = SimpleNamespace(
            path=Path("data/run_1"),
            close=lambda status=None: None,
            completion_status=writer_status,
        )
        widget = SimpleNamespace(
            _raman_mda_pending=True,
            _raman_mda_canceled=False,
            _raman_mda_writer=writer,
            _mda_live_last_completion=0.0,
            mda_live_eta_label=_FakeStatusLabel(),
            status=_FakeStatusLabel(),
            core=SimpleNamespace(mda=mda),
        )
        widget._finish_live_mda_timing = MethodType(
            HardwareWidget._finish_live_mda_timing, widget
        )
        return widget

    def test_successful_run_reports_completion(self):
        widget = self._widget()

        HardwareWidget._on_raman_mda_finished(widget, None)

        self.assertEqual(widget.status.text, "Status: MDA finished OK")
        self.assertFalse(widget._raman_mda_pending)

    def test_canceled_or_errored_run_reports_reason(self):
        reasons = (("canceled", "canceled"), ("errored", "failed"))
        for reason, expected in reasons:
            with self.subTest(reason=reason):
                widget = self._widget(reason=reason)

                HardwareWidget._on_raman_mda_finished(widget, None)

                self.assertIn(f"MDA {expected}", widget.status.text)

    def test_writer_failure_overrides_core_completion(self):
        widget = self._widget(writer_status="failed")

        HardwareWidget._on_raman_mda_finished(widget, None)

        self.assertIn("MDA failed", widget.status.text)

    def test_runtime_without_status_api_uses_completion_signal(self):
        widget = self._widget(status_api=False)

        HardwareWidget._on_raman_mda_finished(widget, None)

        self.assertEqual(widget.status.text, "Status: MDA finished OK")

    def test_runtime_without_status_api_uses_cancellation_signal(self):
        widget = self._widget(status_api=False)

        HardwareWidget._on_raman_mda_canceled(widget, None)
        HardwareWidget._on_raman_mda_finished(widget, None)

        self.assertIn("MDA canceled", widget.status.text)


class TestSavedAcquisitionViewer(unittest.TestCase):
    def test_opens_offline_viewer_with_inferred_run_paths(self):
        opened = []

        class FakeWindow:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                self.shown = False
                opened.append(self)

            def show(self):
                self.shown = True

        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            (run / "raman").mkdir()
            (run / "useq-sequence.json").write_text(
                "{}", encoding="utf-8"
            )
            objective_combo = _FakeCombo()
            objective_combo.addItem("3")
            widget = SimpleNamespace(
                mda_dir_input=SimpleNamespace(text=lambda: str(run)),
                out_path=SimpleNamespace(text=lambda: ""),
                objective_combo=objective_combo,
                vandermonde_objective=None,
                sel_vdm_path=SimpleNamespace(text=lambda: "model.json"),
                _plot_windows=[],
            )
            widget._output_path = MethodType(
                HardwareWidget._output_path, widget
            )

            with patch(
                "napari_raman_widget.widget.LargeAcquisitionViewerWindow",
                FakeWindow,
            ):
                HardwareWidget.open_acquisition_viewer(widget)

        self.assertEqual(len(opened), 1)
        self.assertTrue(opened[0].shown)
        self.assertEqual(opened[0].kwargs["imaging_folder"], str(run))
        self.assertEqual(
            opened[0].kwargs["raman_folder"], str(run / "raman")
        )
        self.assertEqual(opened[0].kwargs["objective"], "3")
        self.assertEqual(widget._plot_windows, opened)


if __name__ == "__main__":
    unittest.main()
