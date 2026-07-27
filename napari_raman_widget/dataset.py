"""Load and assemble a Raman MDA experiment into DataFrames and xarray."""
import json
import re
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import tifffile
import xarray as xr


def load_experiment(
    path, zarr_output="image_data.zarr", batch=None, wavenumbers=None
):
    """Load raman spectra, locations, metadata, and tiff images from an MDA run.

    Parameters
    ----------
    path : str or Path
        Path to an MDA run containing ``images/`` and ``raman.h5``, or a
        legacy run containing plane TIFFs and a ``raman/`` subfolder.
    zarr_output : str
        Where to save the assembled image zarr.
    batch : bool or None
        Whether the raman data is batch mode. Auto-detected if None.
    wavenumbers : array-like or None
        Optional current LightField relative-wavenumber calibration for legacy
        runs. A spectral axis saved with the acquisition takes precedence.

    Returns
    -------
    df : pd.DataFrame or None
        Spectra with columns for pixel values, X, Y, and time.
        MultiIndex: (t, p, z, pt). None if raman loading failed.
    df_locs : pd.DataFrame or None
        Raw laser locations. MultiIndex: (t, p, z, pt). Columns: X, Y.
        None if raman loading failed.
    da : xr.DataArray
        Image stack with dims (t, p, c, z, y, x).
    """
    path = Path(path)
    raman_path = path / "raman"
    streaming_layout = (path / "raman.h5").is_file()
    acquisition = None
    if streaming_layout:
        from .large_dataset_viewing import AcquisitionIndex

        acquisition = AcquisitionIndex.build(path)
        sample_img = acquisition.load_image(acquisition.image_keys[0])
        tiff_folder = path / "images"
    else:
        tiff_folder = path

    # --- Get image dimensions from first tiff ---
    if acquisition is None:
        sample_tiff = next(tiff_folder.glob("*.tiff"), None)
        if sample_tiff is None:
            raise FileNotFoundError(f"No tiff files found in {tiff_folder}")
        sample_img = tifffile.imread(sample_tiff)
    img_y, img_x = sample_img.shape[-2:]

    # -- RAMAN --
    df, df_locs, max_p = None, None, None
    try:
        if acquisition is not None:
            df, df_locs, max_p = _load_raman_index(
                acquisition, img_x, img_y, batch
            )
        else:
            df, df_locs, max_p = _load_raman(
                raman_path, img_x, img_y, batch, wavenumbers
            )
    except Exception as e:
        warnings.warn(
            f"Raman loading failed ({type(e).__name__}: {e}). "
            "Continuing with tiff -> zarr assembly only."
        )

    # -- TIFF -> xarray --
    tiff_records, tiff_coords = [], []
    if acquisition is not None:
        for key in acquisition.image_keys:
            tiff_records.append(acquisition.load_image(key))
            tiff_coords.append(tuple(key))
    else:
        tiff_pat = re.compile(r"t(\d+)_p(\d+)_c(\d+)_z(\d+)\.tiff")
        for file in sorted(tiff_folder.glob("*.tiff")):
            if (match := tiff_pat.search(file.name)):
                t, p, c, z = map(int, match.groups())
                tiff_records.append(tifffile.imread(file))
                tiff_coords.append((t, p, c, z))
    if not tiff_records:
        raise FileNotFoundError(
            f"No tiffs matching t*_p*_c*_z*.tiff found in {tiff_folder}"
        )
    tiff_coords = np.array(tiff_coords)
    t_vals = np.unique(tiff_coords[:, 0])
    p_vals = np.unique(tiff_coords[:, 1])
    c_vals = np.unique(tiff_coords[:, 2])
    z_vals = np.unique(tiff_coords[:, 3])

    # Fall back to the tiff p range if raman loading failed
    if max_p is None:
        max_p = int(p_vals.max())
    full_p_range = np.arange(0, max_p + 1)

    def nearest_p(p, known):
        idx = np.searchsorted(known, p, side="right") - 1
        return known[np.clip(idx, 0, len(known) - 1)]

    p_fill = {p: nearest_p(p, p_vals) for p in full_p_range}
    coord_index = {
        "t": {v: i for i, v in enumerate(t_vals)},
        "p": {v: i for i, v in enumerate(p_vals)},
        "c": {v: i for i, v in enumerate(c_vals)},
        "z": {v: i for i, v in enumerate(z_vals)},
    }
    arr_known = np.zeros(
        (len(t_vals), len(p_vals), len(c_vals), len(z_vals), img_y, img_x),
        dtype=tiff_records[0].dtype,
    )
    for (t, p, c, z), img in zip(tiff_coords, tiff_records):
        arr_known[
            coord_index["t"][t],
            coord_index["p"][p],
            coord_index["c"][c],
            coord_index["z"][z],
        ] = img
    arr = np.stack(
        [
            arr_known[:, coord_index["p"][p_fill[p]], :, :, :, :]
            for p in full_p_range
        ],
        axis=1,
    )
    da = xr.DataArray(
        arr,
        dims=["t", "p", "c", "z", "y", "x"],
        coords={"t": t_vals, "p": full_p_range, "c": c_vals, "z": z_vals},
        name="image",
    )
    ds = da.to_dataset()

    # Attach the useq sequence (if present) as a dataset attribute
    useq_file = path / "useq-sequence.json"
    if useq_file.exists():
        ds.attrs["useq_sequence"] = useq_file.read_text()
    else:
        warnings.warn(f"No useq-sequence.json found in {path}")

    ds.to_zarr(zarr_output, mode="w")
    print(f"Saved Zarr to {zarr_output}")
    return df, df_locs, da


def _load_raman_index(acquisition, img_x, img_y, batch):
    """Assemble the streaming HDF5 records without loading the full run."""
    records, index = [], []
    location_records, location_index, designations = [], [], []
    time_by_event = {}

    for key in sorted(acquisition.raman_files):
        spectra = acquisition.load_spectra(key)
        locations = acquisition.load_locations(key)
        labels = acquisition.load_designations(key)
        metadata = acquisition.load_raman_metadata(key)
        timestamp_ns = metadata.get("timestamp_ns")
        time_by_event[tuple(key)] = (
            pd.to_datetime(timestamp_ns, unit="ns", utc=True)
            if timestamp_ns is not None
            else pd.NaT
        )

        for point, spectrum in enumerate(spectra):
            records.append(spectrum)
            index.append((key.t, key.p, key.z, point))
        pixel_locations = locations * [img_y, img_x]
        # Stored points are y/x; the legacy DataFrame exposes X/Y.
        for point, (row, column) in enumerate(pixel_locations):
            location_records.append((column, row))
            location_index.append((key.t, key.p, key.z, point))
            designations.append(labels[point] if point < len(labels) else "")

    if not records:
        raise FileNotFoundError(
            f"No committed Raman events found in {acquisition.raman_folder}"
        )
    columns, _ = acquisition.spectral_axis(len(records[0]))
    df = pd.DataFrame(
        records,
        columns=columns,
        index=pd.MultiIndex.from_tuples(index, names=["t", "p", "z", "pt"]),
    )
    df_locs = pd.DataFrame(
        location_records,
        index=pd.MultiIndex.from_tuples(
            location_index, names=["t", "p", "z", "pt"]
        ),
        columns=["X", "Y"],
    )
    df_locs["designation"] = designations

    if batch is None:
        import h5py

        with h5py.File(acquisition.raman_folder, "r") as handle:
            batch = bool(handle.attrs.get("batch", False))

    if batch:
        if len(df_locs):
            loc_summary = (
                df_locs.groupby(level=["t", "p", "z"])[["X", "Y"]]
                .mean()
                .assign(pt=0)
                .reset_index()
                .set_index(["t", "p", "z", "pt"])
            )
            df = df.merge(loc_summary, left_index=True, right_index=True, how="left")
    elif len(df_locs):
        df = df.merge(
            df_locs[["X", "Y"]], left_index=True, right_index=True, how="left"
        )
    df["time"] = df.index.droplevel("pt").map(time_by_event)
    df.attrs["spectral_axis"] = "wavenumber"
    df.attrs["spectral_axis_units"] = "cm^-1"
    max_p = int(df.index.get_level_values("p").max())
    return df, df_locs, max_p


def _load_raman(raman_path, img_x, img_y, batch, wavenumbers=None):
    """Assemble raman spectra + locations + times. Returns (df, df_locs, max_p)."""
    data_pat = re.compile(r"raman_p(\d+)_t(\d+)_z(\d+)_data\.npy")
    loc_pat = re.compile(r"raman_p(\d+)_t(\d+)_z(\d+)_locations\.npy")
    meta_pat = re.compile(r"raman_p(\d+)_t(\d+)_z(\d+)_meta\.json")
    data_files = sorted(
        f for f in raman_path.glob("*_data.npy") if data_pat.search(f.name)
    )
    if not data_files:
        raise FileNotFoundError(f"No *_data.npy files found in {raman_path}")
    if batch is None:
        sample = np.load(data_files[0])
        batch = sample.squeeze().ndim == 1
    saved_axis = raman_path / "wavenumbers.npy"
    if saved_axis.exists():
        wavenumbers = np.load(saved_axis)
    records, index = [], []
    for file in data_files:
        p, t, z = map(int, data_pat.search(file.name).groups())
        spec = np.load(file)
        if batch:
            records.append(spec.squeeze())
            index.append((t, p, z, 0))
        else:
            for pt, acc_spec in enumerate(spec):
                records.append(acc_spec)
                index.append((t, p, z, pt))
    spectrum_length = len(records[0])
    if wavenumbers is not None:
        wavenumbers = np.asarray(wavenumbers, dtype=float)
        if wavenumbers.ndim != 1 or len(wavenumbers) != spectrum_length:
            raise ValueError(
                "Wavenumber calibration length does not match Raman spectra"
            )
        columns = wavenumbers
    else:
        columns = None
    df = pd.DataFrame(
        records,
        columns=columns,
        index=pd.MultiIndex.from_tuples(index, names=["t", "p", "z", "pt"]),
    )
    records_locs, index_locs = [], []
    for file in sorted(raman_path.glob("*_locations.npy")):
        match = loc_pat.search(file.name)
        if not match:
            continue
        p, t, z = map(int, match.groups())
        coords = np.load(file) * [img_x, img_y]
        for pt, row in enumerate(coords):
            index_locs.append((t, p, z, pt))
            records_locs.append(row)
    df_locs = pd.DataFrame(
        records_locs,
        index=pd.MultiIndex.from_tuples(
            index_locs, names=["t", "p", "z", "pt"]
        ),
        columns=["X", "Y"],
    )
    if batch:
        loc_summary = (
            df_locs.groupby(level=["t", "p", "z"])
            .agg(X=("X", "mean"), Y=("Y", "mean"))
            .assign(pt=0)
            .reset_index()
            .set_index(["t", "p", "z", "pt"])
        )
        df = df.merge(loc_summary, left_index=True, right_index=True)
    else:
        df = df.merge(df_locs, left_index=True, right_index=True)
    time_dict = {}
    for file in raman_path.glob("*_meta.json"):
        match = meta_pat.search(file.name)
        if not match:
            continue
        p, t, z = map(int, match.groups())
        with open(file) as f:
            meta = json.load(f)
        time_dict[(t, p, z)] = pd.to_datetime(meta["time"])
    df["time"] = df.index.droplevel("pt").map(time_dict)
    if wavenumbers is not None:
        df.attrs["spectral_axis"] = "wavenumber"
        df.attrs["spectral_axis_units"] = "cm^-1"
    max_p = int(df.index.get_level_values("p").max())
    return df, df_locs, max_p
