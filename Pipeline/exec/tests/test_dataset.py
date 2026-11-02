"""G4+G5 acceptance test: dataset, loading functions and DataLoader.

Checks (plan G4+G5):
- ``CheetahDataset``: sample with x (4, res, res, res) = rho_hat + log(E_i/L),
  y (1, res, res, res) and scaling; ``nondim=False`` gives rho/phi physically;
  ``downsample`` coarsens consistently; old files (rho_hat/phi_hat) are rejected with a
  clear message.
- ``load_cheetah_sc``: a batch from ``train_loader`` has shape (B, 4, res, res, res) /
  (B, 1, res, res, res); ``test_loaders`` has the resolutions as keys; the output
  normalizer is fitted only for the raw ablation; existing files are kept.
"""

from __future__ import annotations

import h5py
import numpy as np
import pytest
import torch

from scno.data.dataset import CheetahDataset, load_cheetah_sc, missing_files
from scno.data.generate import DATASET_NAME

META_KEYS = (
    "rho",
    "phi",
    "rho_hat",
    "phi_hat",
    "total_charge",
    "reference_length",
    "normalized_extent",
    "cell_size",
    "gamma",
    "kind",
)


def _load(tmp_path, ranges, **kwargs):
    return load_cheetah_sc(
        data_root=tmp_path,
        n_train=4,
        n_tests=[2, 2],
        batch_size=2,
        test_batch_sizes=[2, 2],
        train_resolution=16,
        test_resolutions=[16, 32],
        generate=True,
        seed=5,
        ranges=ranges,
        num_workers=0,
        **kwargs,
    )


def test_cheetah_dataset_item(h5_file):
    ds = CheetahDataset(h5_file)
    assert len(ds) == 6
    assert ds.grid_shape == (16, 16, 16)

    item = ds[0]
    assert tuple(item["x"].shape) == (4, 16, 16, 16)
    assert tuple(item["y"].shape) == (1, 16, 16, 16)
    assert item["x"].dtype == torch.float32
    assert torch.equal(item["x"][0], item["rho_hat"])
    assert torch.equal(item["y"][0], item["phi_hat"])
    # Channels 1..3: constant log(E_i/L)
    log_aspect = item["normalized_extent"].log().float()
    for c in range(3):
        assert torch.all(item["x"][1 + c] == log_aspect[c])
    for key in META_KEYS:
        assert key in item, key
    assert tuple(item["normalized_extent"].shape) == (3,)
    assert abs(float(item["rho_hat"].double().mean()) - 1.0) < 1e-5

    assert set(CheetahDataset(h5_file, return_meta=False)[0].keys()) == {"x", "y"}


def test_cheetah_dataset_raw(h5_file):
    """nondim=False: x/y physical, same channel structure, same meta data."""
    hat = CheetahDataset(h5_file)[2]
    raw = CheetahDataset(h5_file, nondim=False)[2]

    assert torch.equal(raw["x"][0], raw["rho"])
    assert torch.equal(raw["y"][0], raw["phi"])
    assert torch.equal(raw["x"][1:], hat["x"][1:])  # log(E_i/L) unchanged
    for key in ("rho", "phi", "rho_hat", "phi_hat", "total_charge"):
        assert torch.equal(raw[key], hat[key]), key


def test_cheetah_dataset_downsample(h5_file):
    full = CheetahDataset(h5_file)[1]
    coarse = CheetahDataset(h5_file, downsample=2)[1]

    assert tuple(coarse["rho_hat"].shape) == (8, 8, 8)
    assert torch.allclose(coarse["rho_hat"].mean(), full["rho_hat"].mean(), rtol=1e-5)
    assert torch.allclose(coarse["total_charge"], full["total_charge"], rtol=1e-5)
    assert torch.allclose(coarse["cell_size"], 2 * full["cell_size"])
    assert torch.allclose(coarse["normalized_extent"], full["normalized_extent"])

    with pytest.raises(ValueError):
        CheetahDataset(h5_file, downsample=3)


def test_read_xy_matches_items(h5_file):
    for nondim in (True, False):
        ds = CheetahDataset(h5_file, nondim=nondim)
        x, y = ds.read_xy(3)
        assert tuple(x.shape) == (3, 4, 16, 16, 16)
        assert tuple(y.shape) == (3, 1, 16, 16, 16)
        for i in range(3):
            assert torch.allclose(x[i], ds[i]["x"])
            assert torch.allclose(y[i], ds[i]["y"])


def test_old_format_rejected(tmp_path):
    path = tmp_path / "old.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("rho_hat", data=np.zeros((1, 4, 4, 4), dtype="float32"))
    with pytest.raises(ValueError, match="old file format"):
        CheetahDataset(path)


def test_load_cheetah_sc_batch_and_processors(tmp_path, small_ranges):
    train_loader, test_loaders, data_processor = _load(tmp_path, small_ranges)

    # Keys of test_loaders = resolutions
    assert set(test_loaders.keys()) == {16, 32}

    # A batch from the training loader: dict {"x", "y"} with channel dimension
    batch = next(iter(train_loader))
    assert set(batch.keys()) == {"x", "y"}
    assert tuple(batch["x"].shape) == (2, 4, 16, 16, 16), batch["x"].shape
    assert tuple(batch["y"].shape) == (2, 1, 16, 16, 16), batch["y"].shape
    assert batch["x"].dtype == torch.float32
    assert batch["y"].dtype == torch.float32

    test_batch = next(iter(test_loaders[32]))
    assert tuple(test_batch["y"].shape) == (2, 1, 32, 32, 32)

    # nondim=True: the physical scaling is the only one, no Gaussian layer on top
    assert data_processor.in_normalizer is None
    assert data_processor.out_normalizer is None


def test_output_normalizer_only_for_raw_ablation(tmp_path, small_ranges):
    """encode_output=None resolves to `not nondim`, an explicit value wins."""
    _, _, raw = _load(tmp_path, small_ranges, nondim=False)
    assert raw.out_normalizer is not None
    assert raw.out_normalizer.mean is not None
    assert raw.out_normalizer.std is not None

    _, _, forced = _load(tmp_path, small_ranges, encode_output=True)
    assert forced.out_normalizer is not None


def test_load_cheetah_sc_reuses_existing_files(tmp_path, small_ranges):
    """Existing files must not be regenerated."""
    _load(tmp_path, small_ranges)
    assert missing_files(tmp_path, 16, [16, 32]) == []

    path = tmp_path / f"{DATASET_NAME}_train_16.h5"
    mtime_before = path.stat().st_mtime_ns
    _load(tmp_path, small_ranges)
    assert path.stat().st_mtime_ns == mtime_before, "existing file was regenerated"
