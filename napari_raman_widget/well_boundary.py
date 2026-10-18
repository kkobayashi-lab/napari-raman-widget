"""Image boundary measurements in calibrated stage coordinates (micrometers)."""
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter, label
from scipy.optimize import least_squares
from skimage.filters import threshold_otsu
from skimage.measure import find_contours


@dataclass
class BoundaryCapture:
    image: np.ndarray
    xyz: tuple
    metadata: dict
    model: tuple
    reference_xy: np.ndarray
    pixels_yx: np.ndarray | None = None
    stage_xy: np.ndarray | None = None


def detect_boundary(image, roi=None):
    """Trace the interface of the dark wall and bright well; review is mandatory."""
    image = np.asarray(image, dtype=float)
    if image.ndim != 2 or min(image.shape) < 16 or not np.isfinite(image).all():
        raise ValueError("Boundary detection needs a finite 2D grayscale image")
    y0, x0, y1, x1 = roi or (0, 0, *image.shape)
    crop = gaussian_filter(image[y0:y1, x0:x1], 1)
    if min(crop.shape) < 16 or np.ptp(crop) <= 0:
        raise ValueError("No boundary contrast; refocus or select a larger region")
    mask = crop < threshold_otsu(crop)
    components, count = label(mask)
    sizes = np.bincount(components.ravel())
    sizes[0] = 0
    if not count or sizes.max() < 20:
        raise ValueError("No continuous dark boundary found")
    band = components == sizes.argmax()
    if band.mean() > 0.65:
        raise ValueError("Dark region is too broad; select the boundary or adjust illumination")
    # A wall can fill the entire outside of the well. Its skeleton would run
    # through the black area rather than along the optical boundary.
    # Split contours at the frame margin to exclude screenshot borders and
    # keep one continuous interface, not disconnected cell outlines.
    margin = max(5, int(min(crop.shape)*0.025))
    segments = []
    for contour in find_contours(band.astype(float), 0.5):
        valid = ((contour >= margin).all(axis=1)
                 & (contour < np.asarray(crop.shape)-margin).all(axis=1))
        cuts = np.flatnonzero(np.diff(np.r_[False, valid, False]))
        segments.extend(contour[a:b] for a, b in cuts.reshape(-1, 2))
    points = max(segments, key=len) if segments else np.empty((0, 2))
    if len(points) < 20 or np.linalg.norm(np.ptp(points, axis=0)) < 20:
        raise ValueError("Boundary is too short; capture a longer visible section")
    return points + [y0, x0]


def boundary_to_stage(pixels_yx, xyz, reference_xy, model):
    """Match click-to-center's negative displacement and polynomial ordering."""
    coefficients, degree = model
    xy = np.asarray(pixels_yx)[:, ::-1] - reference_xy
    x, y = xy.T
    terms = np.column_stack([x**i * y**(d-i)
                             for d in range(degree + 1) for i in range(d + 1)])
    # Constant fit residual is not a displacement from the optical reference.
    delta = terms @ coefficients - coefficients[0]
    return np.asarray(xyz[:2]) - delta


def fit_well_boundary(captures, expected_radius):
    """Fit equally weighted views and reject inadequate geometry or agreement."""
    if len(captures) < 3:
        raise ValueError("Accept at least 3 views from different sides of the well")
    views = [c.stage_xy for c in captures]
    points = np.concatenate([p[np.linspace(0, len(p)-1, 128).astype(int)] for p in views])
    origin = points.mean(axis=0)
    q = points - origin
    design = np.column_stack([2*q, np.ones(len(q))])
    solution, _, rank, _ = np.linalg.lstsq(design, (q*q).sum(axis=1), rcond=None)
    if rank < 3:
        raise ValueError("Boundary views do not span a circle; collect another side")
    initial = [*solution[:2], np.linalg.norm(q - solution[:2], axis=1).mean()]
    result = least_squares(lambda v: np.linalg.norm(q-v[:2], axis=1)-v[2], initial,
                           loss="soft_l1", f_scale=5)
    center, radius = result.x[:2] + origin, result.x[2]
    if not result.success or not np.isfinite(result.x).all() or radius <= 0:
        raise ValueError("Could not fit a stable well center")
    directions = np.asarray([p.mean(axis=0)-center for p in views])
    angles = np.sort(np.mod(np.arctan2(directions[:, 1], directions[:, 0]), 2*np.pi))
    coverage = 2*np.pi - np.diff(np.r_[angles, angles[0]+2*np.pi]).max()
    if coverage < np.deg2rad(120) or np.linalg.cond(result.jac) > 30:
        raise ValueError("Views are clustered; capture a different side of the well")
    if abs(radius-expected_radius) > expected_radius*0.2:
        raise ValueError("Fitted diameter disagrees with the selected plate; check boundaries")
    residual = np.linalg.norm(points-center, axis=1)-radius
    rms = float(np.sqrt(np.mean(residual**2)))
    if rms > max(5, radius*0.01):
        raise ValueError("Boundaries disagree; review overlays and remove incorrect views")
    return tuple(center), float(radius), rms
