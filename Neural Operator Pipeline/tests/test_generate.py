"""G3 acceptance test: the file writer produces correct HDF5 files.

Checks (plan G3):
- a small run (n_train=4, n_tests=[2, 2], res 16/32) creates the files
- all datasets present (raw fields rho/phi), shapes (N, res, res, res), dtype float32
- no NaN/Inf
- train and test seeds do not overlap
- ``write_hdf5`` alone: grid shape, fixed number of particles, attributes
"""

from __future__ import annotations

import h5py
import numpy as np

from scno.data.generate import DATASET_NAME, DISTRIBUTIONS, generate_cheetah_sc, write_hdf5

# Raw fields; the scaling quantities are derived only on loading (CheetahDataset).
EXPECTED_KEYS = {"rho", "phi", "cell_size", "gamma", "kind", "seed", "params"}


def _generate(tmp_path, ranges, seed=7):
    return generate_cheetah_sc(
        root=tmp_path,
        n_train=4,
        n_tests=[2, 2],
        train_resolution=16,
        test_resolutions=[16, 32],
        seed=seed,
        ranges=ranges,
    )


def test_generate_writes_files(tmp_path, small_ranges):
    written = _generate(tmp_path, small_ranges)

    expected_names = {
        f"{DATASET_NAME}_train_16.h5",
        f"{DATASET_NAME}_test_16.h5",
        f"{DATASET_NAME}_test_32.h5",
    }
    got_names = {p.name for p in written}
    assert got_names == expected_names, f"expected files: {expected_names - got_names}"
    for name in expected_names:
        assert (tmp_path / name).exists()
    assert not list(tmp_path.glob("*.tmp")), "temporary file left behind"


def test_file_contents_and_shapes(tmp_path, small_ranges):
    _generate(tmp_path, small_ranges)

    for name, n, res in (("train_16", 4, 16), ("test_32", 2, 32)):
        with h5py.File(tmp_path / f"{DATASET_NAME}_{name}.h5", "r") as f:
            assert set(f.keys()) == EXPECTED_KEYS
            for key in ("rho", "phi"):
                assert f[key].shape == (n, res, res, res), (key, f[key].shape)
                assert f[key].dtype == np.float32
                assert np.isfinite(f[key][:]).all()
            assert f["cell_size"].shape == (n, 3)
            for key in ("gamma", "kind", "seed"):
                assert f[key].shape == (n,), key
            assert (f["kind"][:] < len(DISTRIBUTIONS)).all()
            assert tuple(f.attrs["grid_shape"]) == (res, res, res)
            assert f.attrs["grid_extent"] == 3.0


def test_train_test_seeds_disjoint(tmp_path, small_ranges):
    base_seed = 7
    _generate(tmp_path, small_ranges, seed=base_seed)

    def seeds(name):
        with h5py.File(tmp_path / f"{DATASET_NAME}_{name}.h5", "r") as f:
            return set(f["seed"][:].tolist())

    train_seeds = seeds("train_16")
    test16_seeds = seeds("test_16")
    test32_seeds = seeds("test_32")

    assert len(train_seeds) == 4
    assert len(test16_seeds) == 2
    assert len(test32_seeds) == 2
    # No overlap: neither train <-> test nor between test resolutions
    assert train_seeds.isdisjoint(test16_seeds)
    assert train_seeds.isdisjoint(test32_seeds)
    assert test16_seeds.isdisjoint(test32_seeds)

    # The seeds are exactly the consecutive base + global_index
    expected = set(range(base_seed, base_seed + 8))
    assert train_seeds | test16_seeds | test32_seeds == expected


def test_write_hdf5_single_file(tmp_path, small_ranges):
    path = write_hdf5(
        tmp_path / "single.h5",
        num_samples=3,
        grid_shape=(16, 12, 8),
        seed=1,
        ranges=small_ranges,
        num_particles=20_000,
        batch_size=2,  # two write blocks: 2 + 1
    )
    with h5py.File(path, "r") as f:
        assert f["rho"].shape == (3, 16, 12, 8)
        assert (f["params/n_particles"][:] == 20_000).all()
        assert np.isfinite(f["phi"][:]).all()
        # Stored physically: charge on the grid <= drawn beam charge
        charge = np.abs(
            f["rho"][:].astype("float64").sum(axis=(1, 2, 3))
            * f["cell_size"][:].prod(axis=1)
        )
        assert (charge <= 1.001 * f["params/total_charge"][:]).all()
        assert (charge > 0.5 * f["params/total_charge"][:]).all()
