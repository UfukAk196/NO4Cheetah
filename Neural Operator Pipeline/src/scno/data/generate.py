"""G3 - File writer: generates the dataset as HDF5 files (h5py).

``write_hdf5`` writes one file, ``generate_cheetah_sc`` one per required resolution:
    {root}/cheetah_sc_train_{res}.h5   and   {root}/cheetah_sc_test_{res}.h5

Layout of a file (N samples, grid nx x ny x nz):
    rho               (N, nx, ny, nz) float32  [C/m^3]  charge density
    phi               (N, nx, ny, nz) float32  [V]      potential
    cell_size         (N, 3)  float64  [m]  cell size WITHOUT gamma
    gamma             (N,)    float64
    kind              (N,)    int64         index into ``DISTRIBUTIONS``
    seed              (N,)    int64         seed of the sample
    params/<name>     (N,)                  drawn beam parameters
    attrs: dataset_name, seed, grid_shape, grid_extent, distributions,
           first_index, cheetah_version, date

The physical fields are stored, not rho_hat/phi_hat. ``CheetahDataset`` derives the
nondimensionalization (and thus total_charge, reference_length, normalized_extent)
from rho, cell_size and gamma on loading -- so raw and scaled fields can be compared
and the scaling changed without regenerating.

Seeds: each sample gets the unique, consecutive seed ``seed + g`` (g = global index,
``first_index`` + local index). Train AND test samples therefore never overlap (not
even between different test resolutions); every sample is reproducible.

CLI: ``python -m scno.data.generate --config configs/default.yaml`` (reads ``data:``).
"""

from __future__ import annotations

import argparse
import os
from datetime import date
from pathlib import Path

import cheetah
import h5py
import numpy as np
from tqdm import tqdm

from scno.utils.common import load_config

from .fields import _as_grid_shape, make_sample, make_space_charge_kick
from .sampling import LOG_UNIFORM_KEYS, UNIFORM_KEYS, sample_beam_params

DATASET_NAME = "cheetah_sc"

# Distribution families of the beams; ``kind`` in the file is the index here.
# ``ParticleBeam.from_parameters`` generates Gaussian beams.
DISTRIBUTIONS = ("gaussian",)

SCALAR_KEYS = ("gamma",)
VECTOR_KEYS = ("cell_size",)
PARAM_KEYS = LOG_UNIFORM_KEYS + UNIFORM_KEYS


def _probe_seed(base_seed: int, global_index: int) -> int:
    """Unique, reproducible seed of sample ``global_index``."""
    return base_seed + global_index


def dataset_path(root: str | Path, kind: str, res: int) -> Path:
    """File name of a split file, e.g. ``root/cheetah_sc_train_32.h5``."""
    return Path(root) / f"{DATASET_NAME}_{kind}_{res}.h5"


def write_hdf5(
    path: str | Path,
    num_samples: int,
    grid_shape: int | tuple[int, int, int],
    seed: int,
    ranges: dict[str, dict[str, float]] | None = None,
    num_particles: int | None = None,
    first_index: int = 0,
    batch_size: int = 64,
    desc: str | None = None,
) -> Path:
    """Generates ``num_samples`` samples with Cheetah and writes them to ``path``.

    Parameters
    ----------
    path : str | Path
        Target .h5 file (the directory is created). It is first written to
        ``<path>.tmp`` and renamed only at the end, so that an aborted run leaves no
        half file that would later count as "present".
    num_samples : int
        Number of samples.
    grid_shape : int | (nx, ny, nz)
        Grid resolution; an int means res^3.
    seed : int
        Base seed; beam parameters and sample seeds are derived from it.
    ranges : dict, optional
        Beam parameter bounds (G1). Default: ``sampling.DEFAULT_RANGES``.
    num_particles : int, optional
        Fixed number of particles per beam instead of the drawn ``n_particles``.
    first_index : int
        Global index of the first sample. Samples with the same global index have
        identical beam parameters for the same ``seed``.
    batch_size : int
        How many samples are buffered before they are written to the file.
    desc : str, optional
        Label of the progress bar.

    Returns
    -------
    Path of the written file.
    """
    if num_samples < 1:
        raise ValueError(f"num_samples must be >= 1, got {num_samples}")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    grid = _as_grid_shape(grid_shape)
    n = num_samples

    # The parameter list is a deterministic stream: entry g is the same for every list
    # length. So draw up to first_index + n and cut off.
    all_params = sample_beam_params(first_index + n, seed=seed, ranges=ranges)[
        first_index:
    ]

    tmp_path = path.with_name(path.name + ".tmp")
    with h5py.File(tmp_path, "w") as f:
        chunks = (1, *grid)
        rho_ds = f.create_dataset("rho", (n, *grid), dtype="float32", chunks=chunks)
        phi_ds = f.create_dataset("phi", (n, *grid), dtype="float32", chunks=chunks)
        scalars = {k: np.empty(n, dtype="float64") for k in SCALAR_KEYS}
        vectors = {k: np.empty((n, 3), dtype="float64") for k in VECTOR_KEYS}
        params = {k: np.empty(n, dtype="float64") for k in PARAM_KEYS}
        seeds = np.empty(n, dtype="int64")

        rho_buf, phi_buf, start = [], [], 0
        for i in tqdm(range(n), desc=desc or path.name):
            g = first_index + i
            p = dict(all_params[i])
            if num_particles is not None:
                p["n_particles"] = int(num_particles)
            p["seed"] = _probe_seed(seed, g)

            sample = make_sample(p, grid)
            rho, phi = sample["rho"], sample["phi"]
            if not (rho.isfinite().all() and phi.isfinite().all()):
                raise ValueError(f"NaN/Inf in sample {g} ({path.name}, params={p})")

            rho_buf.append(rho.numpy())
            phi_buf.append(phi.numpy())
            for k in SCALAR_KEYS:
                scalars[k][i] = sample[k]
            for k in VECTOR_KEYS:
                vectors[k][i] = sample[k].numpy()
            for k in PARAM_KEYS:
                params[k][i] = p[k]
            seeds[i] = p["seed"]

            if len(rho_buf) == batch_size or i == n - 1:
                stop = start + len(rho_buf)
                rho_ds[start:stop] = np.stack(rho_buf)
                phi_ds[start:stop] = np.stack(phi_buf)
                rho_buf, phi_buf, start = [], [], stop

        for k, v in {**scalars, **vectors}.items():
            f.create_dataset(k, data=v)
        f.create_dataset("kind", data=np.zeros(n, dtype="int64"))  # all gaussian
        f.create_dataset("seed", data=seeds)
        group = f.create_group("params")
        for k, v in params.items():
            group.create_dataset(k, data=v.astype("int64") if k == "n_particles" else v)

        f.attrs["dataset_name"] = DATASET_NAME
        f.attrs["seed"] = seed
        f.attrs["first_index"] = first_index
        f.attrs["grid_shape"] = np.asarray(grid, dtype="int64")
        f.attrs["grid_extent"] = float(make_space_charge_kick(grid).grid_extent_x)
        f.attrs["distributions"] = np.array(DISTRIBUTIONS, dtype=h5py.string_dtype())
        f.attrs["cheetah_version"] = cheetah.__version__
        f.attrs["date"] = date.today().isoformat()

    os.replace(tmp_path, path)
    return path


def generate_cheetah_sc(
    root: str | Path,
    n_train: int,
    n_tests: list[int],
    train_resolution: int,
    test_resolutions: list[int],
    seed: int,
    ranges: dict[str, dict[str, float]] | None = None,
) -> list[Path]:
    """Generates the Cheetah space-charge dataset as HDF5 files.

    Parameters
    ----------
    root : str | Path
        Target directory of the .h5 files (created if needed).
    n_train : int
        Number of training samples.
    n_tests : list[int]
        Number of test samples PER test resolution (length = len(test_resolutions)).
    train_resolution : int
        Resolution of the training data.
    test_resolutions : list[int]
        Resolutions of the test data.
    seed : int
        Base seed; all sample seeds are derived from it.
    ranges : dict, optional
        Beam parameter bounds (G1). Defaults to ``sampling.DEFAULT_RANGES``.

    Returns
    -------
    list[Path]
        Paths of the written .h5 files.
    """
    if len(n_tests) != len(test_resolutions):
        raise ValueError(
            f"n_tests ({len(n_tests)}) must match "
            f"test_resolutions ({len(test_resolutions)})"
        )

    written = [
        write_hdf5(
            dataset_path(root, "train", train_resolution),
            n_train,
            train_resolution,
            seed,
            ranges=ranges,
            first_index=0,
            desc=f"train @ {train_resolution}",
        )
    ]

    # Test data (several resolutions), each with independent samples via the
    # consecutive global index -> no overlap among each other.
    offset = n_train
    for res, n_test in zip(test_resolutions, n_tests):
        written.append(
            write_hdf5(
                dataset_path(root, "test", res),
                n_test,
                res,
                seed,
                ranges=ranges,
                first_index=offset,
                desc=f"test @ {res}",
            )
        )
        offset += n_test

    return written


def main(argv: list[str] | None = None) -> None:
    """CLI: python -m scno.data.generate --config configs/default.yaml"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to the YAML config; its `data:` section describes the dataset.",
    )
    args = parser.parse_args(argv)

    data = load_config(args.config)["data"]
    written = generate_cheetah_sc(
        root=data["data_root"],
        n_train=data["n_train"],
        n_tests=data["n_tests"],
        train_resolution=data["train_resolution"],
        test_resolutions=data["test_resolutions"],
        seed=data["seed"],
        ranges=data.get("ranges"),
    )
    print(f"Written files ({len(written)}):")
    for path in written:
        print(f"  {path}")


if __name__ == "__main__":
    main()
