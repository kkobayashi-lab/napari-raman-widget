import unittest
from pathlib import Path
from types import SimpleNamespace

from napari_raman_widget.widget import (
    HardwareWidget,
    _parse_raman_z_indices,
    _raman_free_autofocus_allowed,
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
