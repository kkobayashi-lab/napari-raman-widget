"""Measured plate focus surfaces in stage micrometers."""
from dataclasses import dataclass

import numpy as np
from pydantic import BaseModel, field_validator


def vandermonde(xy, degree):
    x, y = np.asarray(xy).T
    return np.column_stack([x**i * y**(order-i)
                            for order in range(degree+1) for i in range(order+1)])


@dataclass
class FocusFit:
    degree: int
    coefficients: np.ndarray
    center: np.ndarray
    scale: np.ndarray
    rms: float
    validation_rms: float

    def predict(self, xy):
        return vandermonde((np.asarray(xy)-self.center)/self.scale, self.degree) @ self.coefficients


class PlateFocusCalibration(BaseModel):
    samples: tuple[tuple[float, float, float], ...] = ()
    enabled: bool = False

    @field_validator("samples")
    @classmethod
    def finite_samples(cls, samples):
        if not np.isfinite(samples).all():
            raise ValueError("Focus coordinates must be finite")
        return samples

    def fit(self):
        data = np.asarray(self.samples)
        if len(data) < 4:
            raise ValueError("Record at least four focus points spread across the plate")
        xy, z = data[:, :2], data[:, 2]
        if len(np.unique(xy, axis=0)) != len(xy):
            raise ValueError("Focus points must have distinct XY coordinates")
        center = xy.mean(axis=0)
        scale = np.ptp(xy, axis=0)
        scale[scale == 0] = 1
        normalized = (xy-center)/scale
        if np.linalg.matrix_rank(vandermonde(normalized, 1)) < 3:
            raise ValueError("Focus points must not lie on a single line")
        best = None
        # Total-degree polynomials, capped at cubic. LOO rejects interpolation
        # and geometries that lose rank when a single observation is removed.
        for degree in range(4):
            matrix = vandermonde(normalized, degree)
            if len(z) <= matrix.shape[1]:
                continue
            u, singular, _ = np.linalg.svd(matrix, full_matrices=False)
            if singular[-1] < singular[0]*1e-8:
                continue
            leverage = (u*u).sum(axis=1)
            if np.any(1-leverage < 1e-8):
                continue
            coefficients = np.linalg.lstsq(matrix, z, rcond=None)[0]
            residual = z-matrix @ coefficients
            score = float(np.sqrt(np.mean((residual/(1-leverage))**2)))
            fit = FocusFit(degree, coefficients, center, scale,
                           float(np.sqrt(np.mean(residual**2))), score)
            # Keep the simpler model when scores are effectively tied.
            if best is None or score < best.validation_rms - max(1e-7, best.validation_rms*.01):
                best = fit
        if best is None:
            raise ValueError("Add more spatially distributed focus points")
        return best

    def heights(self, xy):
        xy = np.asarray(xy, dtype=float).reshape(-1, 2)
        if not np.isfinite(xy).all():
            raise ValueError("Plate positions must be finite")
        samples = np.asarray(self.samples)
        fit = self.fit()
        # Avoid unconstrained polynomial extrapolation outside measured coverage.
        from scipy.spatial import Delaunay
        hull = Delaunay((samples[:, :2]-fit.center)/fit.scale)
        if len(xy) and np.any(hull.find_simplex((xy-fit.center)/fit.scale, tol=1e-8) < 0):
            raise ValueError("Record focus points surrounding all acquisition positions (outside measured coverage)")
        heights = fit.predict(xy)
        if not np.isfinite(heights).all():
            raise ValueError("Predicted focus heights must be finite")
        return heights
