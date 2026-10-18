"""Deterministic sampling of native napari Shapes in image pixel coordinates."""
import numpy as np
from matplotlib.path import Path


def sample_region(vertices, shape_type, count, image_shape):
    """Return exactly count interior points, retaining the source t,p,g,c,z plane.

    A progressively refined grid gives reproducible area coverage. Ellipses and
    rectangles are sampled in their own axes so rotation and narrow regions do
    not inflate the candidate grid. No image-sized mask is allocated.
    """
    vertices = np.asarray(vertices, dtype=float)
    if vertices.ndim != 2 or vertices.shape[1] != 7 or len(vertices) < 3:
        raise ValueError("Draw a closed rectangle, ellipse, or polygon on an image plane")
    if shape_type not in ("rectangle", "ellipse", "polygon"):
        raise ValueError("Batches require a rectangle, ellipse, or polygon; lines and paths are not areas")
    if not np.isfinite(vertices).all():
        raise ValueError("Batch vertices must be finite")
    plane = vertices[0, :5]
    if np.any(vertices[:, :5] != plane) or np.any(plane != np.round(plane)) or np.any(plane < 0):
        raise ValueError("Each batch must lie on one acquired image plane")
    pixels = vertices[:, -2:]
    if np.any(pixels < 0) or np.any(pixels > np.asarray(image_shape)):
        raise ValueError("Keep the entire batch shape inside the source image")
    if int(count) != count or not 1 <= count <= 10000:
        raise ValueError("Points per batch must be between 1 and 10,000")
    count = int(count)
    if shape_type in ("rectangle", "ellipse"):
        if len(pixels) != 4:
            raise ValueError("Rectangles and ellipses need four bounding vertices")
        origin = pixels[0]
        basis = np.array([pixels[1] - origin, pixels[3] - origin])
        if abs(np.linalg.det(basis)) < 1e-10:
            raise ValueError("Batch shape must have nonzero area")
        low, span = np.zeros(2), np.ones(2)
        lengths = np.linalg.norm(basis, axis=1)
        inside = (lambda pts: np.sum((2 * pts - 1)**2, axis=1) < 1) if shape_type == "ellipse" else (
            lambda pts: np.ones(len(pts), dtype=bool))
        transform = lambda pts: origin + pts @ basis
    else:
        area = abs(np.sum(pixels[:, 0] * np.roll(pixels[:, 1], -1)
                          - pixels[:, 1] * np.roll(pixels[:, 0], -1))) / 2
        if area < 1e-10:
            raise ValueError("Batch polygon must have nonzero area and not cross itself")
        low = pixels.min(axis=0)
        span = np.ptp(pixels, axis=0)
        lengths = span
        path = Path(np.vstack([pixels, pixels[0]]))
        inside = path.contains_points
        transform = lambda pts: pts
    center = np.asarray([low + span / 2])
    if count == 1 and inside(center)[0]:
        return np.column_stack([plane[None, :], transform(center)])
    # Keep sampling bounded even for highly elongated or almost empty polygons.
    budget = max(4, count * 2)
    while budget <= 1_000_000:
        ny = max(1, int(np.ceil(np.sqrt(budget * lengths[0] / lengths[1]))))
        nx = max(1, int(np.ceil(budget / ny)))
        if ny * nx > 1_000_000:
            break
        yy, xx = np.meshgrid((np.arange(ny) + .5) / ny, (np.arange(nx) + .5) / nx, indexing="ij")
        candidates = low + np.column_stack([yy.ravel(), xx.ravel()]) * span
        candidates = candidates[inside(candidates)]
        if len(candidates) >= count:
            # Select across the whole region, not only its first rows.
            indices = np.floor((np.arange(count) + .5) * len(candidates) / count).astype(int)
            points = transform(candidates[indices])
            return np.column_stack([np.tile(plane, (count, 1)), points])
        budget *= 2
    raise ValueError("Shape is too thin or complex to sample; simplify it or reduce the point count")


def region_center(vertices, shape_type):
    """Geometric center in source (y,x) pixels, independent of sampling density."""
    pixels = np.asarray(vertices, dtype=float)[:, -2:]
    if shape_type in ("rectangle", "ellipse"):
        return pixels.mean(axis=0)
    following = np.roll(pixels, -1, axis=0)
    cross = pixels[:, 0] * following[:, 1] - following[:, 0] * pixels[:, 1]
    area2 = cross.sum()
    if not np.isfinite(area2) or abs(area2) < 1e-10:
        raise ValueError("Batch polygon must have nonzero area")
    return ((pixels + following) * cross[:, None]).sum(axis=0) / (3 * area2)


def batch_positions(image_sequence, batches, image_shape, model=None, **kwargs):
    """Compile each region independently, even when regions share an image plane."""
    from .raman_selection import selected_positions, target
    positions = []
    if not batches:
        raise ValueError("Draw at least one Raman batch shape")
    for batch in batches:
        points = np.asarray(batch["points"], dtype=float)
        if len(points) != batch["count"]:
            raise ValueError("Batch preview is incomplete; regenerate its points")
        center = region_center(batch["vertices"], batch["shape_type"])
        result = selected_positions(image_sequence, points, image_shape, model, batch=True,
                                    batch_center=center, **kwargs)
        if len(result) != 1:
            raise ValueError("Each batch must belong to one image plane")
        target(result[0])["batch_region"] = {
            **{key: value for key, value in batch.items() if key != "points"},
            "center_yx": center.tolist(), "stage_centered": True,
        }
        positions.append(result[0].replace(name=batch["name"]))
    return positions
