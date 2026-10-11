import numpy as np
import pytest

from napari_raman_widget.grid_calibration import vandermonde_fov


def test_rotated_scale_and_polynomial_center_derivative():
    model = np.array([[99, -70], [-0.3, 0], [0, 0.2], [1, 2], [3, 4], [5, 6]])
    assert vandermonde_fov((model, 2), 100, 200) == pytest.approx((20, 60))


@pytest.mark.parametrize("model", [
    (np.zeros((3, 2)), 1),
    (np.array([[0, 0], [0, 1], [0, 2]]), 1),
    (np.full((3, 2), np.nan), 1),
    (np.ones((2, 2)), 1),
    (np.ones((1, 2)), 0),
])
def test_invalid_or_degenerate_models_are_rejected(model):
    with pytest.raises(ValueError):
        vandermonde_fov(model, 100, 200)
