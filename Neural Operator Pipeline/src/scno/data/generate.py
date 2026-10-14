"""G3 - Datei-Writer: erzeugt den Datensatz als HDF5-Dateien (h5py).

``write_hdf5`` schreibt eine Datei, ``generate_cheetah_sc`` pro benoetigter
Aufloesung je eine:
    {root}/cheetah_sc_train_{res}.h5   und   {root}/cheetah_sc_test_{res}.h5

Layout einer Datei (N Proben, Gitter nx x ny x nz):
    rho_hat           (N, nx, ny, nz) float32   dimensionslose Dichte
    phi_hat           (N, nx, ny, nz) float32   dimensionsloses Potential
    total_charge      (N,)    float64  [C]  Skalierungsladung Q_s (s. fields.py)
    reference_length  (N,)    float64  [m]  L der gamma-gestreckten Box
    aspect            (N, 3)  float64       Ei / L, Produkt = 1
    cell_size         (N, 3)  float64  [m]  Zellgroesse OHNE gamma
    gamma             (N,)    float64
    kind              (N,)    int64         Index in ``DISTRIBUTIONS``
    seed              (N,)    int64         Seed der Probe
    params/<name>     (N,)                  gezogene Strahlparameter
    attrs: dataset_name, seed, grid_shape, grid_extent, distributions,
           first_index, cheetah_version, date

Seeds: Jede Probe bekommt den eindeutigen, fortlaufenden Seed ``seed + g``
(g = globaler Index, ``first_index`` + lokaler Index). Dadurch ueberschneiden
sich Train- UND Test-Proben nie (auch nicht zwischen verschiedenen
Test-Aufloesungen); jede Probe ist reproduzierbar.
"""

from __future__ import annotations

import argparse
import os
from datetime import date
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from tqdm import tqdm

import cheetah

from .sampling import DEFAULT_RANGES, LOG_UNIFORM_KEYS, UNIFORM_KEYS, sample_beam_params
from .fields import DEFAULT_GRID_EXTENT, _as_grid_shape, make_sample

DATASET_NAME = "cheetah_sc"

# Verteilungsfamilien der Strahlen; ``kind`` in der Datei ist der Index hier.
# ``ParticleBeam.from_parameters`` erzeugt Gauss-Strahlen.
DISTRIBUTIONS = ("gaussian",)

SCALAR_KEYS = ("total_charge", "reference_length", "gamma")
VECTOR_KEYS = ("aspect", "cell_size")
PARAM_KEYS = LOG_UNIFORM_KEYS + UNIFORM_KEYS


def _load_yaml_config(path: str | Path) -> dict:
    """Liest die CLI-Config (YAML). Nur ``yaml.safe_load``, keine Zwangs-Schema."""
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _probe_seed(base_seed: int, global_index: int) -> int:
    """Eindeutiger, reproduzierbarer Seed fuer Probe ``global_index``."""
    return base_seed + global_index


def dataset_path(root: str | Path, kind: str, res: int) -> Path:
    """Dateiname einer Split-Datei, z.B. ``root/cheetah_sc_train_32.h5``."""
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
    """Erzeugt ``num_samples`` Proben mit Cheetah und schreibt sie nach ``path``.

    Parameters
    ----------
    path : str | Path
        Ziel-.h5-Datei (Verzeichnis wird angelegt). Geschrieben wird zuerst
        in ``<path>.tmp`` und erst am Ende umbenannt, damit ein abgebrochener
        Lauf keine halbe Datei hinterlaesst, die spaeter als "vorhanden" gilt.
    num_samples : int
        Anzahl Proben.
    grid_shape : int | (nx, ny, nz)
        Gitteraufloesung; ein int meint res^3.
    seed : int
        Basis-Seed; Strahlparameter und Proben-Seeds werden daraus abgeleitet.
    ranges : dict, optional
        Strahlparameter-Grenzen (G1). Default: ``sampling.DEFAULT_RANGES``.
    num_particles : int, optional
        Feste Teilchenzahl je Strahl statt der gezogenen ``n_particles``.
    first_index : int
        Globaler Index der ersten Probe. Proben mit gleichem globalem Index
        haben bei gleichem ``seed`` identische Strahlparameter.
    batch_size : int
        Wie viele Proben gepuffert werden, bevor sie in die Datei gehen.
    desc : str, optional
        Beschriftung des Fortschrittsbalkens.

    Returns
    -------
    Path der geschriebenen Datei.
    """
    if ranges is None:
        ranges = DEFAULT_RANGES
    if num_samples < 1:
        raise ValueError(f"num_samples muss >= 1 sein, war {num_samples}")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    grid = _as_grid_shape(grid_shape)
    n = num_samples

    # Die Parameterliste ist ein deterministischer Strom: Eintrag g ist fuer
    # jedes Listenende gleich. Also bis first_index + n ziehen und abschneiden.
    all_params = sample_beam_params(first_index + n, seed=seed, ranges=ranges)[first_index:]

    tmp_path = path.with_name(path.name + ".tmp")
    with h5py.File(tmp_path, "w") as f:
        chunks = (1, *grid)
        rho_ds = f.create_dataset("rho_hat", (n, *grid), dtype="float32", chunks=chunks)
        phi_ds = f.create_dataset("phi_hat", (n, *grid), dtype="float32", chunks=chunks)
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
            rho, phi = sample["rho_hat"], sample["phi_hat"]
            if not (rho.isfinite().all() and phi.isfinite().all()):
                raise ValueError(f"NaN/Inf in Probe {g} ({path.name}, params={p})")

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
        f.create_dataset("kind", data=np.zeros(n, dtype="int64"))  # alle gaussian
        f.create_dataset("seed", data=seeds)
        group = f.create_group("params")
        for k, v in params.items():
            group.create_dataset(k, data=v.astype("int64") if k == "n_particles" else v)

        f.attrs["dataset_name"] = DATASET_NAME
        f.attrs["seed"] = seed
        f.attrs["first_index"] = first_index
        f.attrs["grid_shape"] = np.asarray(grid, dtype="int64")
        f.attrs["grid_extent"] = DEFAULT_GRID_EXTENT
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
    """Erzeugt den Cheetah-Space-Charge-Datensatz als HDF5-Dateien.

    Parameters
    ----------
    root : str | Path
        Zielverzeichnis fuer die .h5-Dateien (wird angelegt, falls noetig).
    n_train : int
        Anzahl Training-Proben.
    n_tests : list[int]
        Anzahl Test-Proben JE Test-Aufloesung (Laenge = len(test_resolutions)).
    train_resolution : int
        Aufloesung der Trainingsdaten.
    test_resolutions : list[int]
        Aufloesungen der Testdaten.
    seed : int
        Basis-Seed; alle Proben-Seeds werden daraus abgeleitet.
    ranges : dict, optional
        Strahlparameter-Grenzen (G1). Fehlt die Angabe, werden die
        TODO-Defaults von ``sampling.DEFAULT_RANGES`` verwendet.

    Returns
    -------
    list[Path]
        Pfade der geschriebenen .h5-Dateien.
    """
    if len(n_tests) != len(test_resolutions):
        raise ValueError(
            f"n_tests ({len(n_tests)}) must match test_resolutions ({len(test_resolutions)})"
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

    # Testdaten (mehrere Aufloesungen), jeweils unabhaengige Proben via
    # fortlaufendem globalem Index -> keine Ueberschneidung untereinander.
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
    """CLI: python -m data.generate --config configs/generate.yaml"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/generate.yaml",
        help="Pfad zur YAML-Config mit Generierungseinstellungen und ranges.",
    )
    args = parser.parse_args(argv)

    cfg = _load_yaml_config(args.config)
    ranges = cfg.get("ranges", DEFAULT_RANGES)
    written = generate_cheetah_sc(
        root=cfg["root"],
        n_train=cfg["n_train"],
        n_tests=cfg["n_tests"],
        train_resolution=cfg["train_resolution"],
        test_resolutions=cfg["test_resolutions"],
        seed=cfg["seed"],
        ranges=ranges,
    )
    print(f"Geschriebene Dateien ({len(written)}):")
    for path in written:
        print(f"  {path}")


if __name__ == "__main__":
    main()
