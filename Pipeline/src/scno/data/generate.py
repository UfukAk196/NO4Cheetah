"""HDF5 generator for ``dataset.CheetahDataset`` (neuralop ``H5pyDataset``).

Writes exactly the files ``H5pyDataset`` reads:
    {root}/cheetah_sc_train_{train_res}.h5   and   {root}/cheetah_sc_test_{res}.h5

Each file holds ``x`` (rho) and ``y`` (phi), both ``(N, nx, ny, nz)`` float32 with no
channel dim, plus a ``params`` group (cheap, makes every sample reproducible). The
physical fields are stored as they come out of Cheetah -- no nondimensionalization,
no metadata. Loading and encoding are done by ``dataset.py``.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import torch
from cheetah.accelerator import SpaceChargeKick
from cheetah.particles import ParticleBeam
from tqdm import tqdm

DATASET_NAME = "cheetah_sc"

# Beam parameter bounds: sigmas log-uniform, the rest uniform (``_draw_params``).
DEFAULT_RANGES: dict[str, dict[str, float]] = {
    "sigma_x": {"min": 5.0e-6, "max": 1.5e-3},
    "sigma_y": {"min": 5.0e-6, "max": 1.5e-3},
    "sigma_tau": {"min": 1.0e-3, "max": 5.0e-3},
    "energy": {"min": 5.0e7, "max": 1.5e8},
    "total_charge": {"min": 1.0e-10, "max": 2.0e-9},
    "n_particles": {"min": 1.0e6, "max": 1.0e6},
}

_LOG_UNIFORM_KEYS = ("sigma_x", "sigma_y", "sigma_tau")
_UNIFORM_KEYS = ("energy", "total_charge", "n_particles")


def dataset_path(root: str | Path, kind: str, res: int) -> Path:
    """File name of a split file, e.g. ``root/cheetah_sc_train_32.h5``."""
    return Path(root) / f"{DATASET_NAME}_{kind}_{res}.h5"


def _draw_params(rng: np.random.Generator, ranges: dict) -> dict:
    """One set of beam parameters: log-uniform sigmas, uniform energy/charge/n."""
    p = {
        k: float(rng.uniform(ranges[k]["min"], ranges[k]["max"])) for k in _UNIFORM_KEYS
    }
    p["n_particles"] = max(1, int(round(p["n_particles"])))
    for k in _LOG_UNIFORM_KEYS:
        lo, hi = float(ranges[k]["min"]), float(ranges[k]["max"])
        p[k] = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
    return p


def _build_beam(params: dict) -> ParticleBeam:
    """Beam (batch (1,)) of one sample; seed ``torch`` before for a reproducible beam."""
    return ParticleBeam.from_parameters(
        num_particles=int(params["n_particles"]),
        sigma_x=torch.tensor([float(params["sigma_x"])]),
        sigma_y=torch.tensor([float(params["sigma_y"])]),
        sigma_tau=torch.tensor([float(params["sigma_tau"])]),
        energy=torch.tensor([float(params["energy"])]),
        total_charge=torch.tensor([float(params["total_charge"])]),
    )


def _grid(beam: ParticleBeam, sck: SpaceChargeKick) -> tuple[torch.Tensor, torch.Tensor]:
    """(grid_dimensions, cell_size), both (1, 3), as in ``SpaceChargeKick.track``."""
    grid_dimensions = torch.stack(
        [
            sck.grid_extent_x * beam.sigma_x,
            sck.grid_extent_y * beam.sigma_y,
            sck.grid_extent_tau * beam.sigma_tau,
        ],
        dim=-1,
    )
    cell_size = (
        2
        * grid_dimensions
        / torch.tensor(
            sck.grid_shape, device=grid_dimensions.device, dtype=grid_dimensions.dtype
        )
    )
    return grid_dimensions, cell_size


def _simulate(params: dict, sck: SpaceChargeKick) -> tuple[torch.Tensor, torch.Tensor]:
    """(rho, phi) for one sample, mirroring ``SpaceChargeKick.track`` blocks 2-7."""
    beam = _build_beam(params)
    grid_dimensions, cell_size = _grid(beam, sck)
    xp = beam.to_xyz_pxpypz()

    nx, ny, nz = sck.grid_shape
    rho = sck._array_rho(beam, xp, cell_size, grid_dimensions)[..., :nx, :ny, :nz][0]
    phi = sck._solve_poisson_equation(beam, xp, cell_size, grid_dimensions)[0]
    return rho.float(), phi.float() # TODO: maybe also return the grid_dimensions and cell_size for normalization and/ or loss calculation


def _write_split(
    path: Path,
    n: int,
    res: int,
    seed: int,
    ranges: dict,
    desc: str,
) -> Path:
    """Generates ``n`` samples at resolution ``res`` and writes them to ``path``."""
    sck = SpaceChargeKick(effect_length=torch.tensor(0.0), grid_shape=(res, res, res))
    rng = np.random.default_rng(seed)

    x = torch.empty((n, res, res, res), dtype=torch.float32)
    y = torch.empty((n, res, res, res), dtype=torch.float32)
    params = {
        k: np.empty(n, dtype="float64") for k in _UNIFORM_KEYS + _LOG_UNIFORM_KEYS
    }

    for i in tqdm(range(n), desc=desc):
        p = _draw_params(rng, ranges)
        torch.manual_seed(seed + i)  # reproducible beam per sample
        rho, phi = _simulate(p, sck)
        if not (rho.isfinite().all() and phi.isfinite().all()):
            raise ValueError(f"NaN/Inf in sample {i} ({path.name}, params={p})")
        x[i], y[i] = rho, phi
        for k in params:
            params[k][i] = p[k]

    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.create_dataset("x", data=x.numpy())
        f.create_dataset("y", data=y.numpy())
        group = f.create_group("params")
        for k, v in params.items():
            group.create_dataset(
                k, data=v.astype("int64") if k == "n_particles" else v
            )
    return path


def _has_samples(path: Path, n: int) -> bool:
    """True if ``path`` exists and holds at least ``n`` samples."""
    if not path.exists():
        return False
    try:
        with h5py.File(path, "r") as f:
            return f["x"].shape[0] >= n
    except Exception:
        return False


def generate_cheetah_sc(
    root: str | Path,
    n_train: int,
    n_tests: list[int],
    train_resolution: int,
    test_resolutions: list[int],
    seed: int = 0,
    ranges: dict | None = None,
) -> list[Path]:
    """Generates the missing .h5 files for ``CheetahDataset``.

    Only files that are absent or hold fewer than the requested number of samples
    are (re)written, so a call on existing data is a no-op. Train and test samples
    never overlap: each file uses its own seed (``seed`` for train, ``seed + 1 + j``
    for test file ``j``).

    Returns the paths of the files that were written.
    """
    if len(n_tests) != len(test_resolutions):
        raise ValueError(
            f"n_tests ({len(n_tests)}) must match "
            f"test_resolutions ({len(test_resolutions)})"
        )
    if ranges is None:
        ranges = DEFAULT_RANGES

    written = []
    train_path = dataset_path(root, "train", train_resolution)
    if not _has_samples(train_path, n_train):
        written.append(
            _write_split(
                train_path,
                n_train,
                train_resolution,
                seed,
                ranges,
                desc=f"train @ {train_resolution}",
            )
        )

    for j, (res, n_test) in enumerate(zip(test_resolutions, n_tests)):
        test_path = dataset_path(root, "test", res)
        if not _has_samples(test_path, n_test):
            written.append(
                _write_split(
                    test_path,
                    n_test,
                    res,
                    seed + 1 + j,
                    ranges,
                    desc=f"test @ {res}",
                )
            )

    return written
