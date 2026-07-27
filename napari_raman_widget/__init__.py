"""napari widget for controlling the CNS Raman microscopy rig."""
from .dataset import load_experiment

__all__ = ["HardwareWidget", "load_experiment"]
__version__ = "0.1.0"


def __getattr__(name):
    """Avoid importing napari/Qt when only data helpers are requested."""
    if name == "HardwareWidget":
        from .widget import HardwareWidget

        return HardwareWidget
    raise AttributeError(name)
