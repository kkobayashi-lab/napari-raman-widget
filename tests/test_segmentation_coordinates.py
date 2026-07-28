from types import SimpleNamespace
from unittest.mock import patch

from napari_raman_widget.widget import HardwareWidget


class _Value:
    def __init__(self, value):
        self._value = value

    def value(self):
        return self._value


class _Status:
    def setText(self, text):
        self.text = text


def test_mask_preview_passes_napari_yx_center():
    widget = SimpleNamespace(
        viewer=object(),
        sel_cy_input=_Value(700),
        sel_cx_input=_Value(300),
        sel_r_input=_Value(50),
        status=_Status(),
        _get_image_xy=lambda: (1344, 1024),
    )

    with patch("cns_control.utils.add_mask_with_hole") as add_mask:
        HardwareWidget.add_mask(widget)

    assert add_mask.call_args.kwargs["circle_center"] == (700, 300)
