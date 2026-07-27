"""Lazy helpers for browsing and stage-stitching large Raman MDA runs.

This module intentionally has no Qt, napari, pymmcore, or hardware imports so
the GUI and notebook can share the same tested implementation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from itertools import product
import json
import math
import os
from pathlib import Path
import re
from typing import NamedTuple
from xml.etree import ElementTree

import h5py
import numpy as np
import tifffile


_IMAGE_RE = re.compile(
    r"^t(?P<t>\d+)_p(?P<p>\d+)_c(?P<c>\d+)_z(?P<z>\d+)\.tiff?$",
    re.IGNORECASE,
)
_RAMAN_RE = re.compile(
    r"^raman_p(?P<p>\d+)_t(?P<t>\d+)_z(?P<z>\d+)_data\.npy$",
    re.IGNORECASE,
)
_LOCATION_RE = re.compile(
    r"^raman_p(?P<p>\d+)_t(?P<t>\d+)_z(?P<z>\d+)_locations\.npy$",
    re.IGNORECASE,
)
_META_RE = re.compile(
    r"^raman_p(?P<p>\d+)_t(?P<t>\d+)_z(?P<z>\d+)_meta\.json$",
    re.IGNORECASE,
)


class ImageKey(NamedTuple):
    t: int
    p: int
    c: int
    z: int


class RamanKey(NamedTuple):
    t: int
    p: int
    z: int


@dataclass(frozen=True)
class OmePlaneRef:
    path: Path
    page: int


@dataclass(frozen=True)
class H5EventRef:
    path: Path
    event: int


@dataclass(frozen=True)
class RamanCellRecord:
    """One spatial Raman point and every spectrum acquired at that point."""

    fov_index: int
    image_source_position: int
    cell_index: int
    raman_key: RamanKey
    point_yx: tuple[float, float] | None
    designation: str
    spectrum_refs: tuple[tuple[RamanKey, int], ...]

    @property
    def repeat_count(self) -> int:
        return len(self.spectrum_refs)


@dataclass(frozen=True)
class RamanImageMatch:
    raman_key: RamanKey
    image_key: ImageKey
    raman_stage_xyz: tuple[float, float, float]
    image_pixel_xy: tuple[float, float]
    center_distance_pixels: float
    fov_index: int | None = None
    cell_index: int | None = None


def _scan(directory: Path, pattern: re.Pattern, key_type):
    found = {}
    with os.scandir(directory) as entries:
        for entry in entries:
            if not entry.is_file():
                continue
            match = pattern.match(entry.name)
            if match is None:
                continue
            values = {name: int(value) for name, value in match.groupdict().items()}
            if key_type is ImageKey:
                key = ImageKey(values["t"], values["p"], values["c"], values["z"])
            else:
                key = RamanKey(values["t"], values["p"], values["z"])
            found[key] = Path(entry.path)
    return found


def _scan_raman(directory: Path):
    spectra = {}
    locations = {}
    metadata = {}
    with os.scandir(directory) as entries:
        for entry in entries:
            if not entry.is_file():
                continue
            match = _RAMAN_RE.match(entry.name)
            destination = spectra
            if match is None:
                match = _LOCATION_RE.match(entry.name)
                destination = locations
            if match is None:
                match = _META_RE.match(entry.name)
                destination = metadata
            if match is None:
                continue
            values = {name: int(value) for name, value in match.groupdict().items()}
            key = RamanKey(values["t"], values["p"], values["z"])
            destination[key] = Path(entry.path)
    return spectra, locations, metadata


def _ome_local_position_and_shape(path: Path):
    """Return original position, T/C/Z sizes, and channel labels for one TIFF."""
    with tifffile.TiffFile(path) as tif:
        xml = tif.ome_metadata
        page_count = len(tif.pages)
    if page_count == 0:
        return None
    if not xml:
        raise ValueError(f"OME metadata missing from {path}")

    root = ElementTree.fromstring(xml)
    namespace = {"ome": root.tag.partition("}")[0].lstrip("{")}
    selected = None
    for image in root.findall("ome:Image", namespace):
        pixels = image.find("ome:Pixels", namespace)
        if pixels is None:
            continue
        for uuid in pixels.findall("ome:TiffData/ome:UUID", namespace):
            if Path(uuid.attrib.get("FileName", "")).name == path.name:
                selected = (image, pixels)
                break
        if selected is not None:
            break
    if selected is None:
        raise ValueError(f"OME metadata does not reference {path.name}")

    image, pixels = selected
    name = image.attrib.get("Name", "")
    match = re.search(r"p(\d+)$", name, re.IGNORECASE)
    original_position = int(match.group(1)) if match else 0
    sizes = tuple(int(pixels.attrib[f"Size{axis}"]) for axis in "TCZ")
    dimension_order = pixels.attrib.get("DimensionOrder", "XYZCT")
    storage_axes = tuple(
        axis.lower()
        for axis in reversed(dimension_order[2:])
        if axis.lower() in {"t", "c", "z"}
    )
    if page_count > math.prod(sizes):
        raise ValueError(
            f"OME plane count mismatch in {path}: metadata={math.prod(sizes)}, "
            f"pages={page_count}"
        )
    channel_names = tuple(
        channel.attrib.get("Name") or channel.attrib.get("ID", "")
        for channel in pixels.findall("ome:Channel", namespace)
    )
    return original_position, sizes, channel_names, page_count, storage_axes


def _scan_ome(directory: Path):
    found: dict[ImageKey, OmePlaneRef] = {}
    channel_names: tuple[str, ...] = ()
    paths = sorted(
        path
        for pattern in ("*.ome.tif", "*.ome.tiff")
        for path in directory.glob(pattern)
    )
    for path in dict.fromkeys(paths):
        info = _ome_local_position_and_shape(path)
        if info is None:
            continue
        (
            position,
            (size_t, size_c, size_z),
            names,
            page_count,
            storage_axes,
        ) = info
        if names:
            channel_names = names
        page = 0
        counts = {"t": size_t, "c": size_c, "z": size_z}
        ranges = (range(counts[axis]) for axis in storage_axes)
        for coordinates in product(*ranges):
            if page >= page_count:
                break
            indices = dict(zip(storage_axes, coordinates))
            key = ImageKey(
                indices.get("t", 0),
                position,
                indices.get("c", 0),
                indices.get("z", 0),
            )
            found[key] = OmePlaneRef(path, page)
            page += 1
    return found, channel_names


def _scan_h5(path: Path):
    spectra: dict[RamanKey, H5EventRef] = {}
    locations: dict[RamanKey, H5EventRef] = {}
    metadata: dict[RamanKey, H5EventRef] = {}
    with h5py.File(path, "r") as handle:
        events = handle["raman/events"]
        for event_index, event in enumerate(events):
            key = RamanKey(int(event["t"]), int(event["p"]), int(event["z"]))
            ref = H5EventRef(path, event_index)
            spectra[key] = ref
            locations[key] = ref
            metadata[key] = ref
    return spectra, locations, metadata


@dataclass(frozen=True)
class AcquisitionIndex:
    """Small in-memory index; image and spectral pixels remain on disk."""

    imaging_folder: Path
    raman_folder: Path
    sequence_file: Path | None
    image_files: dict[ImageKey, Path | OmePlaneRef]
    raman_files: dict[RamanKey, Path | H5EventRef]
    location_files: dict[RamanKey, Path | H5EventRef]
    metadata_files: dict[RamanKey, Path | H5EventRef]
    sequence: dict | None
    image_channel_names: tuple[str, ...] = ()
    _stitch_preview_cache: dict = field(
        default_factory=dict,
        init=False,
        repr=False,
        compare=False,
    )
    @classmethod
    def build(
        cls,
        imaging_folder,
        raman_folder=None,
        sequence_file=None,
    ) -> "AcquisitionIndex":
        imaging_folder = Path(imaging_folder).expanduser().resolve()
        if not imaging_folder.is_dir():
            raise NotADirectoryError(f"Imaging folder not found: {imaging_folder}")

        run_folder = (
            imaging_folder
            if (imaging_folder / "images").is_dir()
            else imaging_folder.parent
            if imaging_folder.name == "images"
            else imaging_folder
        )
        ome_folder = (
            imaging_folder / "images"
            if (imaging_folder / "images").is_dir()
            else imaging_folder
        )
        if raman_folder is None or not str(raman_folder).strip():
            h5_candidate = run_folder / "raman.h5"
            raman_folder = (
                h5_candidate if h5_candidate.is_file() else run_folder / "raman"
            )
        else:
            raman_folder = Path(raman_folder).expanduser().resolve()
        if not raman_folder.exists():
            raise FileNotFoundError(f"Raman source not found: {raman_folder}")
        if not (raman_folder.is_dir() or raman_folder.is_file()):
            raise ValueError(f"Invalid Raman source: {raman_folder}")

        if sequence_file is None:
            candidate = run_folder / "useq-sequence.json"
            sequence_file = candidate if candidate.exists() else None
        else:
            sequence_file = Path(sequence_file).expanduser().resolve()
            if not sequence_file.is_file():
                raise FileNotFoundError(f"Sequence file not found: {sequence_file}")

        image_channel_names: tuple[str, ...] = ()
        image_files = _scan(imaging_folder, _IMAGE_RE, ImageKey)
        if not image_files and ome_folder.is_dir():
            image_files, image_channel_names = _scan_ome(ome_folder)
        if not image_files:
            raise FileNotFoundError(
                "No legacy plane TIFFs or OME-TIFF acquisition files found in "
                f"{imaging_folder}"
            )
        sequence = None
        if sequence_file is not None:
            sequence = json.loads(sequence_file.read_text(encoding="utf-8"))
        if raman_folder.is_file():
            raman_files, location_files, metadata_files = _scan_h5(raman_folder)
        else:
            raman_files, location_files, metadata_files = _scan_raman(
                raman_folder
            )

        return cls(
            imaging_folder=ome_folder,
            raman_folder=raman_folder,
            sequence_file=sequence_file,
            image_files=image_files,
            raman_files=raman_files,
            location_files=location_files,
            metadata_files=metadata_files,
            sequence=sequence,
            image_channel_names=image_channel_names,
        )

    @cached_property
    def image_keys(self) -> tuple[ImageKey, ...]:
        return tuple(sorted(self.image_files))

    @cached_property
    def image_source_position(self) -> np.ndarray | None:
        if not self.raman_folder.is_file():
            return None
        with h5py.File(self.raman_folder, "r") as handle:
            dataset = handle.get("positions/image_source_position")
            return (
                np.asarray(dataset, dtype=np.int64)
                if dataset is not None
                else None
            )

    def source_position_for_logical(self, position: int) -> int:
        position = int(position)
        mapping = self.image_source_position
        if mapping is not None and 0 <= position < len(mapping):
            source = int(mapping[position])
            if source >= 0:
                return source
        # Legacy runs have no explicit reuse mapping. In that case, equality
        # means literal p equality; a missing image p is handled geometrically.
        return position

    @cached_property
    def raman_fov_source_positions(self) -> tuple[int, ...]:
        return tuple(
            sorted(
                {
                    self.source_position_for_logical(key.p)
                    for key in self.raman_files
                }
            )
        )

    def raman_times(self, fov_index: int) -> tuple[int, ...]:
        source_position = self.raman_fov_source_positions[int(fov_index)]
        return tuple(
            sorted(
                {
                    key.t
                    for key in self.raman_files
                    if self.source_position_for_logical(key.p)
                    == source_position
                }
            )
        )

    def raman_z_indices(
        self, fov_index: int, time_index: int
    ) -> tuple[int, ...]:
        source_position = self.raman_fov_source_positions[int(fov_index)]
        return tuple(
            sorted(
                {
                    key.z
                    for key in self.raman_files
                    if key.t == int(time_index)
                    and self.source_position_for_logical(key.p)
                    == source_position
                }
            )
        )

    def _raman_keys_for_fov(
        self,
        fov_index: int,
        time_index: int | None = None,
        z_index: int | None = None,
    ) -> tuple[RamanKey, ...]:
        try:
            source_position = self.raman_fov_source_positions[int(fov_index)]
        except IndexError as exc:
            raise IndexError(f"Raman FOV index out of range: {fov_index}") from exc
        keys = sorted(
            key
            for key in self.raman_files
            if self.source_position_for_logical(key.p) == source_position
        )
        if not keys:
            return ()
        selected_t = (
            min(key.t for key in keys)
            if time_index is None
            else int(time_index)
        )
        keys = [key for key in keys if key.t == selected_t]
        if not keys:
            return ()
        selected_z = (
            min(key.z for key in keys)
            if z_index is None
            else int(z_index)
        )
        return tuple(key for key in keys if key.z == selected_z)

    def raman_cells(
        self,
        fov_index: int,
        time_index: int | None = None,
        z_index: int | None = None,
    ) -> tuple[RamanCellRecord, ...]:
        """Group spectrum rows by saved spatial point within one physical FOV."""
        fov_index = int(fov_index)
        keys = self._raman_keys_for_fov(
            fov_index, time_index=time_index, z_index=z_index
        )
        source_position = self.raman_fov_source_positions[fov_index]
        groups: list[dict] = []

        for key in keys:
            spectrum_count = self.raman_spectrum_count(key)
            points = self.load_locations(key)
            labels = self.load_designations(key)
            assignments: list[tuple[int, np.ndarray | None, str]] = []

            if len(points) == spectrum_count:
                assignments = [
                    (
                        index,
                        points[index],
                        labels[index] if index < len(labels) else "",
                    )
                    for index in range(spectrum_count)
                ]
            elif len(points) == 1:
                label = labels[0] if labels else ""
                assignments = [
                    (index, points[0], label)
                    for index in range(spectrum_count)
                ]
            elif len(points) and spectrum_count % len(points) == 0:
                repeats = spectrum_count // len(points)
                for point_index, point in enumerate(points):
                    label = (
                        labels[point_index]
                        if point_index < len(labels)
                        else ""
                    )
                    assignments.extend(
                        (
                            point_index * repeats + repeat,
                            point,
                            label,
                        )
                        for repeat in range(repeats)
                    )
            elif spectrum_count == 1 and len(points):
                assignments = [
                    (
                        0,
                        points.mean(axis=0),
                        "integrated batch",
                    )
                ]
            else:
                # Without a point-to-spectrum mapping, never average rows:
                # treating each row as its own cell is the conservative choice.
                assignments = [
                    (index, None, "")
                    for index in range(spectrum_count)
                ]

            for spectrum_index, point, designation in assignments:
                destination = None
                if point is not None:
                    for group in groups:
                        existing = group["point"]
                        if existing is not None and np.allclose(
                            existing, point, rtol=0, atol=1e-9
                        ):
                            destination = group
                            break
                if destination is None:
                    destination = {
                        "key": key,
                        "point": (
                            None
                            if point is None
                            else np.asarray(point, dtype=float)
                        ),
                        "designation": designation,
                        "refs": [],
                    }
                    groups.append(destination)
                destination["refs"].append((key, spectrum_index))

        return tuple(
            RamanCellRecord(
                fov_index=fov_index,
                image_source_position=source_position,
                cell_index=cell_index,
                raman_key=group["key"],
                point_yx=(
                    None
                    if group["point"] is None
                    else tuple(float(value) for value in group["point"])
                ),
                designation=str(group["designation"]),
                spectrum_refs=tuple(group["refs"]),
            )
            for cell_index, group in enumerate(groups)
        )

    def raman_cell(
        self,
        fov_index: int,
        cell_index: int,
        time_index: int | None = None,
        z_index: int | None = None,
    ) -> RamanCellRecord:
        cells = self.raman_cells(
            fov_index, time_index=time_index, z_index=z_index
        )
        try:
            return cells[int(cell_index)]
        except IndexError as exc:
            raise IndexError(
                f"Cell index {cell_index} is unavailable in Raman FOV "
                f"{fov_index}; this FOV contains {len(cells)} cells"
            ) from exc

    def load_cell_spectra(self, cell: RamanCellRecord) -> np.ndarray:
        cache: dict[RamanKey, np.ndarray] = {}
        rows = []
        for key, spectrum_index in cell.spectrum_refs:
            if key not in cache:
                cache[key] = self.load_spectra(key)
            rows.append(cache[key][spectrum_index])
        if not rows:
            raise ValueError(
                f"No spectra mapped to FOV {cell.fov_index}, "
                f"cell {cell.cell_index}"
            )
        return np.stack(rows)

    @cached_property
    def image_shape(self) -> tuple[int, int]:
        sample = self.load_image(self.image_keys[0])
        return tuple(int(value) for value in sample.shape[-2:])

    @cached_property
    def stage_xyz_um(self) -> np.ndarray:
        if self.sequence is None:
            raise ValueError("Stitching needs a useq-sequence.json file")
        positions = self.sequence.get("stage_positions") or []
        if not positions:
            raise ValueError("Sequence metadata has no stage positions")
        return np.asarray(
            [
                (
                    float(position["x"]),
                    float(position["y"]),
                    float(position.get("z", 0.0)),
                )
                for position in positions
            ],
            dtype=float,
        )

    def summary(self) -> dict:
        image_keys = self.image_keys
        return {
            "imaging_folder": str(self.imaging_folder),
            "raman_folder": str(self.raman_folder),
            "sequence_file": (
                str(self.sequence_file) if self.sequence_file is not None else None
            ),
            "image_files": len(image_keys),
            "raman_files": len(self.raman_files),
            "times": sorted({key.t for key in image_keys}),
            "positions": len({key.p for key in image_keys}),
            "channels": sorted({key.c for key in image_keys}),
            "z_planes": sorted({key.z for key in image_keys}),
            "image_shape": self.image_shape,
        }

    def load_image(self, key: ImageKey) -> np.ndarray:
        try:
            path = self.image_files[key]
        except KeyError as exc:
            raise KeyError(f"No image for t/p/c/z={tuple(key)}") from exc
        if isinstance(path, OmePlaneRef):
            with tifffile.TiffFile(path.path) as tif:
                return np.asarray(tif.pages[path.page].asarray())
        return tifffile.imread(path)

    def load_spectra(self, key: RamanKey) -> np.ndarray:
        try:
            path = self.raman_files[key]
        except KeyError as exc:
            raise KeyError(f"No Raman data for t/p/z={tuple(key)}") from exc
        if isinstance(path, H5EventRef):
            with h5py.File(path.path, "r") as handle:
                event = handle["raman/events"][path.event]
                start = int(event["spectra_start"])
                stop = start + int(event["spectra_count"])
                spectra = np.asarray(handle["raman/spectra"][start:stop])
        else:
            spectra = np.asarray(np.load(path, mmap_mode="r")).squeeze()
        if spectra.ndim == 1:
            return spectra[np.newaxis, :]
        return spectra.reshape(-1, spectra.shape[-1])

    def raman_spectrum_count(self, key: RamanKey) -> int:
        try:
            path = self.raman_files[key]
        except KeyError as exc:
            raise KeyError(f"No Raman data for t/p/z={tuple(key)}") from exc
        if isinstance(path, H5EventRef):
            with h5py.File(path.path, "r") as handle:
                return int(
                    handle["raman/events"][path.event]["spectra_count"]
                )
        spectra = np.asarray(np.load(path, mmap_mode="r")).squeeze()
        return 1 if spectra.ndim == 1 else int(np.prod(spectra.shape[:-1]))

    def load_locations(self, key: RamanKey) -> np.ndarray:
        path = self.location_files.get(key)
        if path is None:
            return np.empty((0, 2), dtype=float)
        if isinstance(path, H5EventRef):
            with h5py.File(path.path, "r") as handle:
                event = handle["raman/events"][path.event]
                start = int(event["points_start"])
                stop = start + int(event["points_count"])
                locations = np.asarray(
                    handle["raman/points_yx"][start:stop], dtype=float
                )
        else:
            locations = np.asarray(np.load(path, mmap_mode="r"), dtype=float)
        return locations.reshape(-1, 2)

    def load_designations(self, key: RamanKey) -> tuple[str, ...]:
        path = self.raman_files.get(key)
        if not isinstance(path, H5EventRef):
            return ()
        with h5py.File(path.path, "r") as handle:
            event = handle["raman/events"][path.event]
            start = int(event["points_start"])
            stop = start + int(event["points_count"])
            return tuple(handle["raman/designations"].asstr()[start:stop])

    def load_raman_metadata(self, key: RamanKey) -> dict:
        path = self.metadata_files.get(key)
        if path is None:
            return {}
        if isinstance(path, H5EventRef):
            with h5py.File(path.path, "r") as handle:
                event = handle["raman/events"][path.event]
                return {
                    "t": int(event["t"]),
                    "p": int(event["p"]),
                    "z": int(event["z"]),
                    "timestamp_ns": int(event["timestamp_ns"]),
                    "exposure_ms": float(event["exposure_ms"]),
                    "stage_xyz_um": np.asarray(
                        event["stage_xyz_um"], dtype=float
                    ).tolist(),
                }
        return json.loads(path.read_text(encoding="utf-8"))

    def spectral_axis(self, length: int) -> tuple[np.ndarray, str]:
        if self.raman_folder.is_file():
            with h5py.File(self.raman_folder, "r") as handle:
                dataset = handle.get("axes/wavenumber_cm_inverse")
                axis = (
                    np.asarray(dataset, dtype=float)
                    if dataset is not None
                    else np.empty(0)
                )
        else:
            path = self.raman_folder / "wavenumbers.npy"
            axis = (
                np.asarray(np.load(path, mmap_mode="r"), dtype=float)
                if path.exists()
                else np.empty(0)
            )
        if axis.ndim == 1 and len(axis) == length:
            return axis, "Raman shift (cm$^{-1}$)"
        return np.arange(length), "Spectral pixel"


def _locations_to_image_pixels(
    locations: np.ndarray, image_shape: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray]:
    """Convert saved [row, column] fractions to matplotlib x/y pixels."""
    if len(locations) == 0:
        return np.empty(0), np.empty(0)
    height, width = image_shape
    if np.nanmax(np.abs(locations)) <= 1.5:
        rows = locations[:, 0] * height
        columns = locations[:, 1] * width
    else:
        rows = locations[:, 0]
        columns = locations[:, 1]
    return columns, rows


def load_vandermonde_geometry(model_path, objective=None) -> dict:
    """Derive the local stage/image geometry at the calibrated image center."""
    model_path = Path(model_path).expanduser().resolve()
    model = json.loads(model_path.read_text(encoding="utf-8"))
    if "objectives" in model:
        if objective is None:
            raise ValueError("objective is required for a multi-objective model")
        objective = str(objective)
        try:
            model = model["objectives"][objective]
        except KeyError as exc:
            available = ", ".join(sorted(model["objectives"]))
            raise ValueError(
                f"No Vandermonde model for objective {objective!r}; "
                f"available: {available}"
            ) from exc

    coefficients = np.asarray(model["C"], dtype=float)
    degree = int(model["degree"])
    if degree < 1 or coefficients.shape[0] < 3:
        raise ValueError("Vandermonde model needs linear terms for stitching")
    # Term order is [1, y, x, y^2, xy, x^2, ...], so at the centered
    # origin the two linear coefficient rows are the exact Jacobian.
    image_to_stage = np.column_stack((coefficients[2], coefficients[1]))
    if not np.all(np.isfinite(image_to_stage)):
        raise ValueError("Vandermonde center Jacobian contains non-finite values")
    if abs(np.linalg.det(image_to_stage)) < 1e-12:
        raise ValueError("Vandermonde center Jacobian is singular")
    stage_to_image = np.linalg.inv(image_to_stage)
    axis_pixel_sizes = np.linalg.norm(image_to_stage, axis=0)
    return {
        "model_path": str(model_path),
        "objective": None if objective is None else str(objective),
        "degree": degree,
        "coefficients": coefficients,
        "img_center": model.get("img_center"),
        "xy_center": model.get("xy_center"),
        "image_to_stage_um_per_pixel": image_to_stage,
        "stage_to_image_pixels_per_um": stage_to_image,
        "pixel_size_x_um": float(axis_pixel_sizes[0]),
        "pixel_size_y_um": float(axis_pixel_sizes[1]),
        "pixel_size_mean_um": float(axis_pixel_sizes.mean()),
    }


def _vandermonde_design(points: np.ndarray, degree: int) -> np.ndarray:
    points = np.atleast_2d(np.asarray(points, dtype=float))
    x, y = points[:, 0], points[:, 1]
    terms = []
    for total_degree in range(degree + 1):
        for x_degree in range(total_degree + 1):
            y_degree = total_degree - x_degree
            terms.append((x**x_degree) * (y**y_degree))
    return np.stack(terms, axis=1)


def _resolve_raman_key(
    acquisition: AcquisitionIndex, raman_index: int
) -> RamanKey:
    raman_index = int(raman_index)
    common_key = RamanKey(0, raman_index, 0)
    if common_key in acquisition.raman_files:
        return common_key
    matches = sorted(
        key for key in acquisition.raman_files if key.p == raman_index
    )
    if not matches:
        raise KeyError(f"No Raman data found for index p={raman_index}")
    return matches[0]


def _raman_stage_xyz(
    acquisition: AcquisitionIndex,
    raman_key: RamanKey,
    geometry: dict,
    point_yx: tuple[float, float] | None = None,
) -> tuple[float, float, float] | None:
    metadata = acquisition.load_raman_metadata(raman_key)
    if not metadata:
        return None
    if "stage_xyz_um" in metadata:
        stage_xyz = np.asarray(metadata["stage_xyz_um"], dtype=float)
    else:
        stage_xyz = np.asarray(
            [
                float(metadata["x"]),
                float(metadata["y"]),
                float(metadata.get("z", 0.0)),
            ]
        )
    locations = (
        acquisition.load_locations(raman_key)
        if point_yx is None
        else np.asarray([point_yx], dtype=float)
    )
    if len(locations) == 0:
        return tuple(stage_xyz)

    height, width = acquisition.image_shape
    columns, rows = _locations_to_image_pixels(locations, (height, width))
    spot_xy = np.array([columns.mean(), rows.mean()])
    image_center = np.asarray(
        geometry.get("img_center") or [width / 2, height / 2],
        dtype=float,
    )
    pixel_offset = spot_xy - image_center
    coefficients = np.asarray(geometry["coefficients"], dtype=float)
    degree = int(geometry["degree"])
    stage_offset = (
        _vandermonde_design(pixel_offset, degree) @ coefficients
    )[0]
    # The calibration fits stage position as a function of where one fixed
    # sample feature appears. Therefore the feature/sample coordinate is the
    # current stage coordinate minus the fitted pixel-dependent displacement.
    stage_xyz[:2] -= stage_offset
    return tuple(float(value) for value in stage_xyz)


def _z_stage_coordinate(
    acquisition: AcquisitionIndex, position: int, z_index: int
) -> float:
    stage_z = float(acquisition.stage_xyz_um[position, 2])
    if acquisition.sequence is None:
        return stage_z
    z_plan = acquisition.sequence.get("z_plan") or {}
    relative = z_plan.get("relative")
    if relative is not None and z_index < len(relative):
        stage_z += float(relative[z_index])
    return stage_z


def _resolve_image_channel(
    acquisition: AcquisitionIndex,
    imaging_channel: int | str | None,
    time_index: int,
) -> int:
    available = sorted(
        {key.c for key in acquisition.image_files if key.t == time_index}
    )
    if not available:
        raise ValueError(f"No imaging channels found at t={time_index}")
    if imaging_channel is None:
        return available[0]
    if isinstance(imaging_channel, str):
        requested = imaging_channel.strip()
        if requested.isdigit():
            imaging_channel = int(requested)
        else:
            channels = [
                {"config": name} for name in acquisition.image_channel_names
            ]
            if not channels and acquisition.sequence is not None:
                channels = acquisition.sequence.get("channels") or []
            by_name = {
                str(channel.get("config", "")).casefold(): index
                for index, channel in enumerate(channels)
            }
            match = by_name.get(requested.casefold())
            if match is None or match not in available:
                labels = [
                    f"{index}: {channels[index].get('config', index)}"
                    if index < len(channels)
                    else str(index)
                    for index in available
                ]
                raise ValueError(
                    f"Imaging channel {requested!r} is unavailable; "
                    f"choose one of {', '.join(labels)}"
                )
            return match
    channel_index = int(imaging_channel)
    if channel_index not in available:
        raise ValueError(
            f"Imaging channel c={channel_index} is unavailable; "
            f"available indices: {available}"
        )
    return channel_index


def _image_channel_label(
    acquisition: AcquisitionIndex, channel_index: int
) -> str:
    if channel_index < len(acquisition.image_channel_names):
        label = acquisition.image_channel_names[channel_index]
        if label:
            return str(label)
    if acquisition.sequence is not None:
        channels = acquisition.sequence.get("channels") or []
        if channel_index < len(channels):
            label = channels[channel_index].get("config")
            if label:
                return str(label)
    return f"channel {channel_index}"


def find_image_for_raman_index(
    acquisition: AcquisitionIndex,
    raman_index: int,
    geometry: dict,
    imaging_channel: int | str | None = None,
    cell_index: int = 0,
    time_index: int | None = None,
    z_index: int | None = None,
) -> RamanImageMatch | None:
    """Find the image containing one cell from a physical Raman FOV."""
    cell = acquisition.raman_cell(
        raman_index,
        cell_index,
        time_index=time_index,
        z_index=z_index,
    )
    raman_key = cell.raman_key
    raman_stage_xyz = _raman_stage_xyz(
        acquisition,
        raman_key,
        geometry,
        point_yx=cell.point_yx,
    )
    if raman_stage_xyz is None:
        return None

    height, width = acquisition.image_shape
    image_center = np.asarray(
        geometry.get("img_center") or [width / 2, height / 2],
        dtype=float,
    )
    stage_to_image = np.asarray(
        geometry["stage_to_image_pixels_per_um"], dtype=float
    )
    raman_xy = np.asarray(raman_stage_xyz[:2])

    image_times = sorted({key.t for key in acquisition.image_files})
    requested_t = cell.raman_key.t
    image_t = (
        requested_t
        if requested_t in image_times
        else min(image_times, key=lambda value: abs(value - requested_t))
    )
    selected_c = _resolve_image_channel(
        acquisition, imaging_channel, image_t
    )

    # A physical FOV mapping is authoritative. Use that image series directly
    # even if calibration would put the marker outside the frame. Geometric
    # searching is only a fallback when that image FOV does not exist.
    direct_candidates = sorted(
        key
        for key in acquisition.image_keys
        if key.t == image_t
        and key.p == cell.image_source_position
        and key.c == selected_c
        and key.p < len(acquisition.stage_xyz_um)
    )
    if direct_candidates:
        requested_z = cell.raman_key.z
        image_key = min(
            direct_candidates,
            key=lambda key: (abs(key.z - requested_z), key.z),
        )
        image_stage_xy = acquisition.stage_xyz_um[image_key.p, :2]
        pixel_xy = image_center + stage_to_image @ (
            image_stage_xy - raman_xy
        )
        center_distance = float(np.linalg.norm(pixel_xy - image_center))
        return RamanImageMatch(
            raman_key=raman_key,
            image_key=image_key,
            raman_stage_xyz=raman_stage_xyz,
            image_pixel_xy=(float(pixel_xy[0]), float(pixel_xy[1])),
            center_distance_pixels=center_distance,
            fov_index=cell.fov_index,
            cell_index=cell.cell_index,
        )

    candidates = []
    for key in acquisition.image_keys:
        if key.t != image_t or key.c != selected_c:
            continue
        if key.p >= len(acquisition.stage_xyz_um):
            continue
        image_stage_xy = acquisition.stage_xyz_um[key.p, :2]
        pixel_xy = image_center + stage_to_image @ (
            image_stage_xy - raman_xy
        )
        x, y = pixel_xy
        if not (0 <= x < width and 0 <= y < height):
            continue
        z_distance = abs(
            _z_stage_coordinate(acquisition, key.p, key.z)
            - raman_stage_xyz[2]
        )
        center_distance = float(np.linalg.norm(pixel_xy - image_center))
        candidates.append((z_distance, center_distance, key, pixel_xy))
    if not candidates:
        return None

    _, center_distance, image_key, pixel_xy = min(
        candidates, key=lambda item: (item[1], item[0], item[2])
    )
    return RamanImageMatch(
        raman_key=raman_key,
        image_key=image_key,
        raman_stage_xyz=raman_stage_xyz,
        image_pixel_xy=(float(pixel_xy[0]), float(pixel_xy[1])),
        center_distance_pixels=center_distance,
        fov_index=cell.fov_index,
        cell_index=cell.cell_index,
    )


def _mean_raman_spectrum(
    acquisition: AcquisitionIndex,
    raman_index: int,
    cell_index: int = 0,
    time_index: int | None = None,
    z_index: int | None = None,
):
    cell = acquisition.raman_cell(
        raman_index,
        cell_index,
        time_index=time_index,
        z_index=z_index,
    )
    raman_key = cell.raman_key
    spectra = acquisition.load_cell_spectra(cell)
    spectrum = spectra.mean(axis=0)
    axis, axis_label = acquisition.spectral_axis(spectrum.shape[-1])
    acquisition_label = (
        f"mean of {len(spectra)} repeats"
        if len(spectra) > 1
        else "1 spectrum"
    )
    repeat_label = (
        f"FOV {cell.fov_index}, cell {cell.cell_index}; "
        f"{acquisition_label}"
    )
    return raman_key, spectrum, axis, axis_label, repeat_label


def _plot_raman_spectrum(
    spectrum_ax,
    raman_key: RamanKey,
    spectrum: np.ndarray,
    axis: np.ndarray,
    axis_label: str,
    repeat_label: str,
):
    spectrum_ax.plot(axis, spectrum, linewidth=1)
    spectrum_ax.set(
        title=f"Raman p={raman_key.p} ({repeat_label})",
        xlabel=axis_label,
        ylabel="Intensity (a.u.)",
    )


def plot_raman_side_by_side(
    acquisition: AcquisitionIndex,
    raman_index: int,
    geometry: dict,
    *,
    imaging_channel: int | str | None = None,
    cell_index: int = 0,
    time_index: int | None = None,
    z_index: int | None = None,
):
    """Plot one Raman index first, with its best containing BF image."""
    import matplotlib.pyplot as plt

    spectrum_data = _mean_raman_spectrum(
        acquisition,
        raman_index,
        cell_index,
        time_index=time_index,
        z_index=z_index,
    )
    match = find_image_for_raman_index(
        acquisition,
        raman_index,
        geometry,
        imaging_channel=imaging_channel,
        cell_index=cell_index,
        time_index=time_index,
        z_index=z_index,
    )
    fig, (spectrum_ax, image_ax) = plt.subplots(
        1, 2, figsize=(12, 5), constrained_layout=True
    )
    _plot_raman_spectrum(spectrum_ax, *spectrum_data)

    if match is not None:
        image = acquisition.load_image(match.image_key)
        x, y = match.image_pixel_xy
        image_ax.imshow(image, cmap="gray")
        image_ax.scatter(
            [x],
            [y],
            s=120,
            facecolor="none",
            edgecolor="red",
            linewidth=2,
        )
        image_ax.set_title(
            f"Best image t/p/c/z={tuple(match.image_key)}\n"
            f"{match.center_distance_pixels:.1f} px from center"
        )
    else:
        image_ax.set_title("No image contains this Raman point")
    image_ax.set_axis_off()
    return fig, match


def raman_stitched_preview(
    acquisition: AcquisitionIndex,
    raman_index: int,
    geometry: dict,
    *,
    imaging_channel: int | str | None = None,
    max_side: int = 2048,
    cell_index: int = 0,
    time_index: int | None = None,
    z_index: int | None = None,
) -> tuple[
    np.ndarray | None,
    tuple[float, float] | None,
    dict | None,
    RamanImageMatch | None,
]:
    """Build the matching BF stitch and locate one Raman point within it."""
    match = find_image_for_raman_index(
        acquisition,
        raman_index,
        geometry,
        imaging_channel=imaging_channel,
        cell_index=cell_index,
        time_index=time_index,
        z_index=z_index,
    )
    if match is None:
        return None, None, None, None

    stage_to_image = np.asarray(
        geometry["stage_to_image_pixels_per_um"], dtype=float
    )
    key = match.image_key
    cache_key = (
        key.t,
        key.c,
        key.z,
        int(max_side),
        stage_to_image.tobytes(),
    )
    context = acquisition._stitch_preview_cache.get(cache_key)
    if context is None:
        mosaic, metadata = stitch_preview(
            acquisition,
            t=key.t,
            c=key.c,
            z=key.z,
            stage_to_image=stage_to_image,
            max_side=max_side,
        )
        keys = _selected_image_keys(
            acquisition, t=key.t, c=key.c, z=key.z
        )
        left, top, _, _ = _tile_geometry(
            acquisition,
            keys,
            pixel_size_um=None,
            stage_to_image=stage_to_image,
            invert_x=False,
            invert_y=False,
        )
        context = mosaic, metadata, keys, left, top
        if len(acquisition._stitch_preview_cache) >= 2:
            acquisition._stitch_preview_cache.clear()
        acquisition._stitch_preview_cache[cache_key] = context
    else:
        mosaic, metadata, keys, left, top = context

    tile_index = keys.index(key)
    stride = metadata["downsample_stride"]
    marker_x = (
        left[tile_index] + match.image_pixel_xy[0]
    ) / stride
    marker_y = (
        top[tile_index] + match.image_pixel_xy[1]
    ) / stride
    marker_xy = (float(marker_x), float(marker_y))
    metadata = dict(metadata)
    metadata["raman_index"] = int(raman_index)
    metadata["cell_index"] = int(cell_index)
    metadata["raman_marker_xy"] = marker_xy
    return mosaic, marker_xy, metadata, match


def plot_raman_on_stitched_image(
    acquisition: AcquisitionIndex,
    raman_index: int,
    geometry: dict,
    *,
    imaging_channel: int | str | None = None,
    max_side: int = 2048,
    cell_index: int = 0,
    time_index: int | None = None,
    z_index: int | None = None,
):
    """Plot one Raman spectrum beside its marked stitched BF preview."""
    import matplotlib.pyplot as plt

    spectrum_data = _mean_raman_spectrum(
        acquisition,
        raman_index,
        cell_index,
        time_index=time_index,
        z_index=z_index,
    )
    mosaic, marker_xy, metadata, match = raman_stitched_preview(
        acquisition,
        raman_index,
        geometry,
        imaging_channel=imaging_channel,
        max_side=max_side,
        cell_index=cell_index,
        time_index=time_index,
        z_index=z_index,
    )
    fig, (spectrum_ax, mosaic_ax) = plt.subplots(
        1, 2, figsize=(13, 5), constrained_layout=True
    )
    _plot_raman_spectrum(spectrum_ax, *spectrum_data)

    if mosaic is not None:
        mosaic_ax.imshow(mosaic, cmap="gray")
        mosaic_ax.scatter(
            [marker_xy[0]],
            [marker_xy[1]],
            s=140,
            facecolor="none",
            edgecolor="red",
            linewidth=2,
        )
        mosaic_ax.set_title(
            f"Stitched "
            f"{_image_channel_label(acquisition, match.image_key.c)} "
            f"(c={match.image_key.c}) z={match.image_key.z}: "
            f"{metadata['tiles']} tiles"
        )
    else:
        mosaic_ax.set_title("No stitched image contains this Raman point")
    mosaic_ax.set_axis_off()
    return fig, match, metadata


def _selected_image_keys(
    acquisition: AcquisitionIndex, t: int, c: int, z: int
) -> list[ImageKey]:
    return sorted(
        key
        for key in acquisition.image_files
        if key.t == t and key.c == c and key.z == z
    )


def _tile_geometry(
    acquisition: AcquisitionIndex,
    keys: list[ImageKey],
    pixel_size_um: float | None,
    stage_to_image: np.ndarray | None,
    invert_x: bool,
    invert_y: bool,
):
    if stage_to_image is not None:
        transform = np.asarray(stage_to_image, dtype=float)
        if transform.shape != (2, 2) or not np.all(np.isfinite(transform)):
            raise ValueError("stage_to_image must be a finite 2x2 matrix")
    else:
        if (
            pixel_size_um is None
            or not np.isfinite(pixel_size_um)
            or pixel_size_um <= 0
        ):
            raise ValueError(
                "Provide stage_to_image from the Vandermonde model or a "
                "positive pixel_size_um fallback"
            )
        transform = np.eye(2, dtype=float) / pixel_size_um
    stage = acquisition.stage_xyz_um
    missing = [key.p for key in keys if key.p >= len(stage)]
    if missing:
        raise ValueError(
            "Sequence has no stage position for image position(s): "
            f"{sorted(set(missing))[:10]}"
        )
    height, width = acquisition.image_shape
    sign_x = -1.0 if invert_x else 1.0
    sign_y = -1.0 if invert_y else 1.0
    stage_xy = stage[[key.p for key in keys], :2]
    stage_xy = stage_xy - stage_xy[0]
    # The calibration maps an image-feature offset to the stage motion that
    # brings that feature to image center. A mosaic places each field in
    # sample/world coordinates, whose displacement has the opposite sign.
    image_xy = -(stage_xy @ transform.T)
    centers_x = sign_x * image_xy[:, 0]
    centers_y = sign_y * image_xy[:, 1]
    left = centers_x - width / 2
    top = centers_y - height / 2
    left -= left.min()
    top -= top.min()
    full_width = int(math.ceil(left.max() + width))
    full_height = int(math.ceil(top.max() + height))
    return left, top, full_height, full_width


def stitch_preview(
    acquisition: AcquisitionIndex,
    *,
    t: int,
    c: int,
    z: int,
    pixel_size_um: float | None = None,
    stage_to_image: np.ndarray | None = None,
    invert_x: bool = False,
    invert_y: bool = False,
    max_side: int = 2048,
) -> tuple[np.ndarray, dict]:
    """Fuse positioned tiles into a RAM-bounded navigation mosaic."""
    keys = _selected_image_keys(acquisition, t=t, c=c, z=z)
    if not keys:
        raise KeyError(f"No images found for t={t}, c={c}, z={z}")
    left, top, full_height, full_width = _tile_geometry(
        acquisition,
        keys,
        pixel_size_um,
        stage_to_image,
        invert_x,
        invert_y,
    )
    stride = max(1, int(math.ceil(max(full_height, full_width) / max_side)))
    preview_height = int(math.ceil(full_height / stride))
    preview_width = int(math.ceil(full_width / stride))
    summed = np.zeros((preview_height, preview_width), dtype=np.float32)
    weights = np.zeros((preview_height, preview_width), dtype=np.float32)

    for key, tile_left, tile_top in zip(keys, left, top):
        tile = acquisition.load_image(key)[::stride, ::stride]
        x0 = int(round(tile_left / stride))
        y0 = int(round(tile_top / stride))
        y1 = min(y0 + tile.shape[0], preview_height)
        x1 = min(x0 + tile.shape[1], preview_width)
        visible = tile[: y1 - y0, : x1 - x0]
        summed[y0:y1, x0:x1] += visible
        weights[y0:y1, x0:x1] += 1

    mosaic = np.divide(
        summed,
        weights,
        out=np.zeros_like(summed),
        where=weights > 0,
    )
    metadata = {
        "t": t,
        "c": c,
        "z": z,
        "tiles": len(keys),
        "full_resolution_shape": (full_height, full_width),
        "preview_shape": mosaic.shape,
        "downsample_stride": stride,
        "pixel_size_um": pixel_size_um,
        "stage_to_image": (
            None
            if stage_to_image is None
            else np.asarray(stage_to_image, dtype=float).tolist()
        ),
        "invert_x": invert_x,
        "invert_y": invert_y,
        "placement_method": "stage",
    }
    return mosaic, metadata


def stitch_all_channels_preview(
    acquisition: AcquisitionIndex,
    *,
    t: int,
    z: int,
    pixel_size_um: float | None = None,
    stage_to_image: np.ndarray | None = None,
    invert_x: bool = False,
    invert_y: bool = False,
    max_side: int = 2048,
) -> dict[int, tuple[np.ndarray, dict]]:
    channels = sorted(
        {key.c for key in acquisition.image_files if key.t == t and key.z == z}
    )
    return {
        channel: stitch_preview(
            acquisition,
            t=t,
            c=channel,
            z=z,
            pixel_size_um=pixel_size_um,
            stage_to_image=stage_to_image,
            invert_x=invert_x,
            invert_y=invert_y,
            max_side=max_side,
        )
        for channel in channels
    }


def fiji_tile_configuration(
    acquisition: AcquisitionIndex,
    *,
    t: int,
    c: int,
    z: int,
    pixel_size_um: float | None = None,
    stage_to_image: np.ndarray | None = None,
    invert_x: bool = False,
    invert_y: bool = False,
) -> str:
    """Return a Fiji Grid/Collection Stitching TileConfiguration string."""
    keys = _selected_image_keys(acquisition, t=t, c=c, z=z)
    if not keys:
        raise KeyError(f"No images found for t={t}, c={c}, z={z}")
    left, top, _, _ = _tile_geometry(
        acquisition,
        keys,
        pixel_size_um,
        stage_to_image,
        invert_x,
        invert_y,
    )
    lines = ["# Define the number of dimensions we are working on", "dim = 2", ""]
    for key, x, y in zip(keys, left, top):
        lines.append(
            f"{acquisition.image_files[key].as_posix()}; ; ({x:.3f}, {y:.3f})"
        )
    return "\n".join(lines) + "\n"


def estimated_eager_size(
    acquisition: AcquisitionIndex, position_count: int = 100_000
) -> dict:
    """Estimate why eager arrays/dataframes are unsuitable at large scale."""
    image = acquisition.load_image(acquisition.image_keys[0])
    image_bytes = int(image.nbytes)
    if acquisition.raman_files:
        first_raman_key = sorted(acquisition.raman_files)[0]
        spectra = acquisition.load_spectra(first_raman_key)
        spectrum_bytes = int(spectra.nbytes)
        spectra_per_file = int(len(spectra))
        spectral_length = int(spectra.shape[-1])
    else:
        spectrum_bytes = spectra_per_file = spectral_length = 0
    return {
        "assumed_positions": int(position_count),
        "one_image_mib": image_bytes / 2**20,
        "eager_images_gib_per_channel_t_z": image_bytes * position_count / 2**30,
        "one_raman_file_kib": spectrum_bytes / 2**10,
        "eager_raman_gib": spectrum_bytes * position_count / 2**30,
        "spectra_per_raman_file": spectra_per_file,
        "spectral_length": spectral_length,
        "prototype_active_pixel_payload_mib": (
            image_bytes + spectrum_bytes
        )
        / 2**20,
    }
