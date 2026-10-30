"""Reduce acquired grid spectra to one calibrated intensity per spatial point."""
import numpy as np
from pathlib import Path
from superqt.utils import ensure_main_thread

from raman_mda_engine.grid_plan import GRID_KEY, grid_indices


class RamanGridImage:
    def __init__(self, sequence, shift, collector, *, frame_shape=None, pixel_size=1, output_dir=None):
        self.regions = sequence.metadata[GRID_KEY]["regions"]
        self.galvo = sequence.metadata[GRID_KEY].get("backend") == "galvo"
        self.center_volts = sequence.metadata[GRID_KEY].get("center_volts")
        self.shift = float(shift)
        self.collector = collector
        self.bin = None
        self.actual_shift = None
        self.values = {}
        self.error = None
        self.frame_shape = frame_shape
        self.pixel_size = pixel_size or 1
        self.output_dir = Path(output_dir) if output_dir is not None else None
        self.scan_preview = None
        self.preview_error = None
        self.layers = {}
        self._rendered = {}
        self._dirty = {}
        self.viewer = None
        self.status = None
        self._active_key = None

    @ensure_main_thread
    def start(self, viewer, status, sequence):
        """Open the first blank map before the first spectrum arrives."""
        self.viewer, self.status = viewer, status
        channel = sequence.metadata.get("raman", {}).get("channel", "RM")
        c = next((i for i, item in enumerate(sequence.channels) if item.config == channel), 0)
        planes = sequence.metadata.get("raman", {}).get("z", [0])
        z = planes[0] if isinstance(planes, (list, tuple)) and planes else 0
        key = (0, 0, z, c)
        self.values.setdefault(key, {})
        self.show(viewer, status, save=False, keys=[key])

    def capture_scan_preview(self, viewer):
        """Freeze the pre-acquisition canvas, including visible scan overlays."""
        if self.output_dir is None:
            return
        try:
            self.scan_preview = np.array(viewer.screenshot(canvas_only=True, flash=False), copy=True)
        except Exception as error:
            self.preview_error = f"Could not capture scan preview: {error}"

    def _save_scan_preview(self):
        if self.output_dir is None:
            return
        if self.scan_preview is not None:
            try:
                from PIL import Image

                self.output_dir.mkdir(parents=True, exist_ok=True)
                Image.fromarray(self.scan_preview).save(self.output_dir / "scan_preview.png")
            except Exception as error:
                self.preview_error = f"Could not save scan preview: {error}"
        return self.preview_error

    @ensure_main_thread
    def add(self, event, spectra):
        """Keep only the selected bin, averaging repeats without retaining spectra."""
        if self.error is not None:
            return
        try:
            spectra = np.atleast_2d(np.asarray(spectra))
            if self.bin is None:
                axis = np.asarray(self.collector.get_wavenumbers(spectra.shape[-1]), dtype=float)
                if axis.ndim != 1 or len(axis) != spectra.shape[-1] or not np.all(np.isfinite(axis)):
                    raise ValueError("No valid Raman-shift calibration is available")
                if not axis.min() <= self.shift <= axis.max():
                    raise ValueError(f"Raman shift {self.shift:g} cm⁻¹ is outside the calibrated range ({axis.min():g}–{axis.max():g})")
                self.bin = int(np.argmin(abs(axis - self.shift)))
                self.actual_shift = float(axis[self.bin])
            p = event.index.get("p", 0)
            for region_index, region in enumerate(self.regions):
                nx, ny = region["x"][2], region["y"][2]
                if p >= nx * ny:
                    p -= nx * ny
                    continue
                order = region.get("order") or {"x": "row_wise_snake", "y": "column_wise_snake"}.get(region.get("snake_axis"), "column_wise")
                row, column = grid_indices(p, ny, nx, order)
                key = (region_index, event.index.get("t", 0), event.index.get("z", 0), event.index.get("c", 0))
                values = np.asarray(spectra[..., self.bin], dtype=float).ravel()
                values = values[np.isfinite(values)]
                points = self.values.setdefault(key, {})
                total, count = points.get((row, column), (0.0, 0))
                points[row, column] = (total + values.sum(), count + len(values))
                self._dirty.setdefault(key, set()).add((row, column))
                if self.viewer is not None:
                    self.show(self.viewer, self.status, save=False, keys=[key])
                break
        except Exception as error:
            # A display failure must not interrupt hardware acquisition or saving.
            self.error = str(error)
            if self.status is not None:
                self.status.setText(f"Status: Raman image unavailable — {error}")

    @ensure_main_thread
    def show(self, viewer, status, reference_layer=None, *, save=True, keys=None):
        preview_error = self._save_scan_preview() if save else None
        if self.error:
            detail = self.error + (f"; {preview_error}" if preview_error else "")
            status.setText(f"Status: MDA completed; Raman image unavailable — {detail}")
            return
        shape = self.frame_shape
        scale, translate = [self.pixel_size] * 2, [0, 0]
        if reference_layer is not None:
            # RGB's final component axis is not a spatial dimension in napari.
            shape = reference_layer.data.shape[-3:-1] if reference_layer.rgb else reference_layer.data.shape[-2:]
            scale = tuple(reference_layer.scale[-2:])
            translate = tuple(reference_layer.translate[-2:])
        save_errors = [preview_error] if preview_error else []
        first_display = not self.layers
        selected_keys = list(self.values) if keys is None else keys
        for key in selected_keys:
            region_index, t, z, c = key
            points = self.values[key]
            region = self.regions[region_index]
            nx, ny = region["x"][2], region["y"][2]
            if self.galvo:
                # The shared napari canvas uses camera-sized spatial extents.
                # Literal volts here made a 0.2 V map sub-pixel beside a 1k
                # camera image (even hidden MDA layers affect viewer bounds).
                # Stretch grid cells only for display; export native samples.
                shape = shape or (ny, nx)
            data = self._render_map(key, shape or (ny, nx))
            shift = self.actual_shift if self.actual_shift is not None else self.shift
            name = f"Raman {shift:g} cm⁻¹ — grid {region_index + 1}, t={t}, z={z}, c={c}"
            layer = self.layers.get(key)
            if layer is not None:
                layer.data = data
                layer.scale = scale
                layer.translate = translate
                layer.name = name
                layer.metadata["raman_shift_cm"] = self.actual_shift
            else:
                layer = viewer.add_image(data, name=name,
                             colormap="inferno", scale=scale, translate=translate,
                             rgb=False, visible=True, interpolation2d="nearest",
                             metadata={"requested_shift_cm": self.shift, "raman_shift_cm": self.actual_shift,
                                       "stage_grid_um": region, "native_grid_shape": (ny, nx),
                                       "display_coordinates": "MDA frame", "resampling": "nearest",
                                       "display_flip_xy": True,
                                       "reduction": "mean of repeated spectra"})
                layer._keep_auto_contrast = True
                if self.galvo:
                    layer.metadata.pop("stage_grid_um", None)
                    layer.metadata.update(galvo_grid_volts=region, center_volts=self.center_volts,
                                          display_coordinates="preview canvas (not camera registration)",
                                          display_flip_xy=True)
                self.layers[key] = layer
            if not save and self._active_key != key:
                if self._active_key in self.layers:
                    self.layers[self._active_key].visible = False
                layer.visible = True
                self._active_key = key
            if save and points and self.output_dir is not None:
                stem = f"raman_shift_{self.actual_shift:g}cm-1_grid{region_index + 1}_t{t}_z{z}_c{c}"
                try:
                    export_layer = self._galvo_export_layer(layer, key) if self.galvo else layer
                    self._save_images(export_layer, self.output_dir / stem)
                except Exception as error:
                    save_errors.append(f"{stem}: {error}")
        if first_display and self.values:
            viewer.reset_view()
        if save_errors:
            status.setText("Status: MDA completed; could not save Raman images — " + "; ".join(save_errors))

    def _galvo_export_layer(self, layer, key):
        """Save one quantitative pixel per grid point, with voltage axes."""
        from types import SimpleNamespace

        region = self.regions[key[0]]
        data = np.full((region["y"][2], region["x"][2]), np.nan, dtype=np.float32)
        for (row, col), (total, count) in self.values[key].items():
            data[row, col] = total / count if count else np.nan
        scale = [(region[a][1] - region[a][0]) / (region[a][2] - 1)
                 if region[a][2] > 1 else 1 for a in ("y", "x")]
        translate = [(region[a][0] + region[a][1]) / 2 if region[a][2] == 1 else region[a][0]
                     for a in ("y", "x")]
        return SimpleNamespace(data=data, scale=scale, translate=translate,
                               contrast_limits=layer.contrast_limits, colormap=layer.colormap,
                               metadata={**layer.metadata, "display_coordinates": "galvo voltage Y/X (V)",
                                         "resampling": "none", "display_flip_xy": False})

    def _render_map(self, key, shape):
        """Update only newly acquired blocks in the frame-sized image."""
        shape = tuple(map(int, shape))
        region = self.regions[key[0]]
        cached = self._rendered.get(key)
        points = self.values[key]
        if cached is None or cached[0].shape != shape:
            data = np.full(shape, np.nan, dtype=np.float32)
            indices = []
            for length, axis in zip(shape, ("y", "x")):
                start, stop, count = region[axis]
                index = np.minimum(((np.arange(length) + 0.5) * count / length).astype(int), count - 1)
                if stop < start:
                    index = count - 1 - index
                # Match the brightfield-facing preview convention for both
                # stage and galvo scans. Traversal order is decoded in add();
                # this is a display-only reversal of X and Y.
                indices.append(index[::-1])
            cached = (data, *indices)
            self._rendered[key] = cached
            changed = points.keys()
        else:
            changed = self._dirty.get(key, ())
        data, rows, columns = cached
        for row, column in changed:
            total, count = points[row, column]
            data[np.ix_(rows == row, columns == column)] = total / count if count else np.nan
        self._dirty.pop(key, None)
        return data

    @staticmethod
    def _save_images(layer, path):
        """Save quantitative float pixels plus the displayed colormap as a PNG."""
        import tifffile
        from PIL import Image

        data = np.asarray(layer.data, dtype=np.float32)
        path.parent.mkdir(parents=True, exist_ok=True)
        low, high = map(float, layer.contrast_limits)
        metadata = {**layer.metadata, "axes": "YX",
                    "scale": list(map(float, layer.scale)),
                    "translate": list(map(float, layer.translate)),
                    "contrast_limits": [low, high], "colormap": layer.colormap.name}
        tifffile.imwrite(str(path) + ".tif", data, photometric="minisblack", metadata=metadata)
        finite = np.isfinite(data)
        normalized = np.zeros_like(data)
        if high > low:
            normalized[finite] = np.clip((data[finite] - low) / (high - low), 0, 1)
        rgba = np.asarray(layer.colormap.map(normalized.ravel())).reshape((*data.shape, 4))
        rgba[~finite, 3] = 0
        Image.fromarray(np.rint(np.clip(rgba, 0, 1) * 255).astype(np.uint8)).save(str(path) + "_preview.png")
