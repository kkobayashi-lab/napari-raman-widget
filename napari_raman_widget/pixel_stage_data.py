"""Read camera frames and their matching stage coordinates for calibration."""

import json
from pathlib import Path

import numpy as np

from .large_dataset_viewing import AcquisitionIndex


def load_pixel_stage_data(path):
    """Return one common T/C/Z plane per saved position and matching stage XY.

    No missing-position filling or Raman spectra loading is performed. Legacy
    xarray Zarr exports remain accepted alongside saved TIFF acquisitions.
    """
    path = Path(path)
    if any((path / marker).exists() for marker in (".zgroup", "zarr.json")):
        import xarray as xr

        with xr.open_zarr(path) as ds:
            sequence = ds.attrs.get("useq_sequence")
            if sequence is None:
                raise ValueError("Dataset has no useq_sequence attribute")
            sequence = json.loads(sequence) if isinstance(sequence, str) else sequence
            image = ds["image"].isel(t=0, c=0, z=0).transpose("p", "y", "x")
            indices = np.asarray(image.coords["p"].values)
            if not np.all(np.isfinite(indices)) or not np.all(indices == indices.astype(int)):
                raise ValueError("Dataset position indices must be integers")
            xy = _position_xy(sequence, indices.astype(int))
            frames = image.values
    else:
        acquisition = AcquisitionIndex.build(path, images_only=True)
        if not acquisition.image_keys:
            raise ValueError("Calibration needs saved camera images; this run has none")
        # Choose one common plane, rather than mixing channels or timepoints
        # when a position is missing an image.
        first = acquisition.image_keys[0]
        keys = sorted(
            (key for key in acquisition.image_keys
             if (key.t, key.c, key.z) == (first.t, first.c, first.z)),
            key=lambda key: key.p,
        )
        xy = _position_xy(acquisition.sequence, [key.p for key in keys])
        frames = np.stack([acquisition.load_image(key) for key in keys])
    if len(frames) < 3:
        raise ValueError("Calibration needs at least 3 saved positions in one T/C/Z plane")
    if frames.ndim != 3:
        raise ValueError("Calibration needs 2D camera images")
    return frames, xy


def _position_xy(sequence, indices):
    if sequence is None:
        raise ValueError("Calibration needs useq-sequence.json in the acquisition folder")
    if sequence.get("grid_plan") or any(
        (position.get("sequence") or {}).get("grid_plan")
        for position in sequence.get("stage_positions", [])
    ):
        raise ValueError(
            "Calibration requires explicit stage positions or a dense Raman grid; "
            "native MDA grid tiles are not supported"
        )
    if "raman_grid" in (sequence.get("metadata") or {}):
        from useq import MDASequence
        from raman_mda_engine.grid_plan import grid_position

        seq = MDASequence.model_validate(sequence)
        positions = [grid_position(seq, int(index)) for index in indices]
        xy = np.asarray([(position.x, position.y) for position in positions], dtype=float)
    else:
        positions = sequence.get("stage_positions") or []
        if any(index < 0 or index >= len(positions) for index in indices):
            raise ValueError("Camera position index has no matching saved stage position")
        try:
            xy = np.asarray([(positions[index]["x"], positions[index]["y"])
                             for index in indices], dtype=float)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("Saved stage positions need numeric X and Y coordinates") from error
    if not np.isfinite(xy).all():
        raise ValueError("Saved stage XY coordinates must be finite")
    return xy
