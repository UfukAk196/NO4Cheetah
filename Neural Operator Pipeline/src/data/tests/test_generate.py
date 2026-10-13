"""G3-Abnahmetest: Datei-Writer erzeugt korrekte HDF5-Dateien.

Prueft (Plan G3):
- kleiner Lauf (n_train=4, n_tests=[2,2], res 16/32) legt die Dateien an
- alle Datensaetze vorhanden, Formen (N,res,res,res), dtype float32
- keine NaN/Inf
- Train- und Test-Seeds ueberschneiden sich nicht
- ``write_hdf5`` einzeln: Gitterform, feste Teilchenzahl, Attribute
"""

from __future__ import annotations

import h5py
import numpy as np

from data.generate import DATASET_NAME, DISTRIBUTIONS, generate_cheetah_sc, write_hdf5


SMALL_RANGES = {
    "sigma_x": {"min": 5.0e-4, "max": 1.5e-3},
    "sigma_y": {"min": 5.0e-4, "max": 1.5e-3},
    "sigma_tau": {"min": 1.0e-3, "max": 5.0e-3},
    "energy": {"min": 5.0e8, "max": 1.5e9},
    "total_charge": {"min": 1.0e-10, "max": 2.0e-9},
    "n_particles": {"min": 1.0e4, "max": 3.0e4},
}

EXPECTED_KEYS = {
    "rho_hat", "phi_hat", "total_charge", "reference_length", "aspect",
    "cell_size", "gamma", "kind", "seed", "params",
}


def _generate(tmp_path, seed=7):
    return generate_cheetah_sc(
        root=tmp_path,
        n_train=4,
        n_tests=[2, 2],
        train_resolution=16,
        test_resolutions=[16, 32],
        seed=seed,
        ranges=SMALL_RANGES,
    )


def test_generate_writes_files(tmp_path):
    written = _generate(tmp_path)

    expected_names = {
        f"{DATASET_NAME}_train_16.h5",
        f"{DATASET_NAME}_test_16.h5",
        f"{DATASET_NAME}_test_32.h5",
    }
    got_names = {p.name for p in written}
    assert got_names == expected_names, f"erwartete Dateien: {expected_names - got_names}"
    for name in expected_names:
        assert (tmp_path / name).exists()
    assert not list(tmp_path.glob("*.tmp")), "temporaere Datei liegen geblieben"


def test_file_contents_and_shapes(tmp_path):
    _generate(tmp_path)

    for name, n, res in (("train_16", 4, 16), ("test_32", 2, 32)):
        with h5py.File(tmp_path / f"{DATASET_NAME}_{name}.h5", "r") as f:
            assert set(f.keys()) == EXPECTED_KEYS
            for key in ("rho_hat", "phi_hat"):
                assert f[key].shape == (n, res, res, res), (key, f[key].shape)
                assert f[key].dtype == np.float32
                assert np.isfinite(f[key][:]).all()
            assert f["aspect"].shape == (n, 3)
            assert f["cell_size"].shape == (n, 3)
            for key in ("total_charge", "reference_length", "gamma", "kind", "seed"):
                assert f[key].shape == (n,), key
            assert (f["kind"][:] < len(DISTRIBUTIONS)).all()
            assert tuple(f.attrs["grid_shape"]) == (res, res, res)


def test_train_test_seeds_disjoint(tmp_path):
    base_seed = 7
    _generate(tmp_path, seed=base_seed)

    def seeds(name):
        with h5py.File(tmp_path / f"{DATASET_NAME}_{name}.h5", "r") as f:
            return set(f["seed"][:].tolist())

    train_seeds = seeds("train_16")
    test16_seeds = seeds("test_16")
    test32_seeds = seeds("test_32")

    assert len(train_seeds) == 4
    assert len(test16_seeds) == 2
    assert len(test32_seeds) == 2
    # keine Ueberschneidung: weder train<->test noch zwischen Test-Aufloesungen
    assert train_seeds.isdisjoint(test16_seeds)
    assert train_seeds.isdisjoint(test32_seeds)
    assert test16_seeds.isdisjoint(test32_seeds)

    # Seeds sind genau die fortlaufenden base+global_index
    expected = set(range(base_seed, base_seed + 8))
    assert train_seeds | test16_seeds | test32_seeds == expected


def test_write_hdf5_single_file(tmp_path):
    path = write_hdf5(
        tmp_path / "smoke.h5",
        num_samples=3,
        grid_shape=(16, 12, 8),
        seed=1,
        ranges=SMALL_RANGES,
        num_particles=20_000,
        batch_size=2,  # zwei Schreibbloecke: 2 + 1
    )
    with h5py.File(path, "r") as f:
        assert f["rho_hat"].shape == (3, 16, 12, 8)
        assert (f["params/n_particles"][:] == 20_000).all()
        assert np.isfinite(f["phi_hat"][:]).all()
        assert np.abs(f["rho_hat"][:].mean(axis=(1, 2, 3)) - 1.0).max() < 1e-4
