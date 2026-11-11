"""Acceptance test for the lean HDF5 generator (``generate.py``).

Checks:
- a small run (n_train=4, n_tests=[2, 2], res 16/32) creates the .h5 files
- ``x``/``y`` shapes (N, res, res, res), dtype float32, no NaN/Inf
- train and test samples do not overlap
- a second call on existing data is a no-op
"""

from __future__ import annotations

import h5py
import numpy as np

from scno.data.generate import DATASET_NAME, dataset_path, generate_cheetah_sc


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


def test_file_contents_and_shapes(tmp_path, small_ranges):
    _generate(tmp_path, small_ranges)

    for kind, res, n in (("train", 16, 4), ("test", 16, 2), ("test", 32, 2)):
        with h5py.File(dataset_path(tmp_path, kind, res), "r") as f:
            assert set(f.keys()) == {"rho", "phi", "params"}
            for key in ("rho", "phi"):
                assert f[key].shape == (n, res, res, res), (key, f[key].shape)
                assert f[key].dtype == np.float32
                assert np.isfinite(f[key][:]).all()


def test_train_and_test_do_not_overlap(tmp_path, small_ranges):
    _generate(tmp_path, small_ranges)

    with h5py.File(dataset_path(tmp_path, "train", 16), "r") as f:
        train = f["rho"][:2]
    with h5py.File(dataset_path(tmp_path, "test", 16), "r") as f:
        test = f["rho"][:]
    assert not np.array_equal(train, test)


def test_second_call_is_noop(tmp_path, small_ranges):
    _generate(tmp_path, small_ranges)
    again = _generate(tmp_path, small_ranges)
    assert again == []
