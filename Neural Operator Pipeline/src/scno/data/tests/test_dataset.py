"""G4+G5-Abnahmetest: Dataset, Lade-Funktionen und DataLoader.

Prueft (Plan G4+G5):
- ``CheetahDataset``: Probe mit x (4,res,res,res) = rho_hat + log(E_i/L),
  y (1,res,res,res) und Skalierung; ``downsample`` vergroebert konsistent.
- ``make_loaders``: Train/Val-Split einer Datei.
- ``load_cheetah_sc``: ein Batch aus ``train_loader`` hat Form
  (B,4,res,res,res)/(B,1,res,res,res); ``test_loaders`` hat die Aufloesungen
  als Keys; Normalisierer sind gefittet; vorhandene Dateien bleiben.
"""

from __future__ import annotations

import pytest
import torch

from scno.data.dataset import CheetahDataset, load_cheetah_sc, make_loaders, missing_files
from scno.data.generate import DATASET_NAME, write_hdf5


SMALL_RANGES = {
    "sigma_x": {"min": 5.0e-4, "max": 1.5e-3},
    "sigma_y": {"min": 5.0e-4, "max": 1.5e-3},
    "sigma_tau": {"min": 1.0e-3, "max": 5.0e-3},
    "energy": {"min": 5.0e8, "max": 1.5e9},
    "total_charge": {"min": 1.0e-10, "max": 2.0e-9},
    "n_particles": {"min": 1.0e4, "max": 2.0e4},
}


@pytest.fixture(scope="module")
def h5_file(tmp_path_factory):
    path = tmp_path_factory.mktemp("h5") / "small.h5"
    return write_hdf5(path, num_samples=6, grid_shape=16, seed=5, ranges=SMALL_RANGES)


def _load(tmp_path):
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
        ranges=SMALL_RANGES,
        num_workers=0,
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
    # Kanaele 1..3: konstant log(E_i/L)
    log_aspect = item["aspect"].log().float()
    for c in range(3):
        assert torch.all(item["x"][1 + c] == log_aspect[c])
    for key in ("total_charge", "reference_length", "aspect", "cell_size", "gamma", "kind"):
        assert key in item, key
    assert tuple(item["aspect"].shape) == (3,)

    assert set(CheetahDataset(h5_file, return_meta=False)[0].keys()) == {"x", "y"}


def test_cheetah_dataset_downsample(h5_file):
    full = CheetahDataset(h5_file)[1]
    coarse = CheetahDataset(h5_file, downsample=2)[1]

    assert tuple(coarse["rho_hat"].shape) == (8, 8, 8)
    assert torch.allclose(coarse["rho_hat"].mean(), full["rho_hat"].mean(), rtol=1e-5)
    assert torch.allclose(coarse["cell_size"], 2 * full["cell_size"])
    assert torch.equal(coarse["aspect"], full["aspect"])

    with pytest.raises(ValueError):
        CheetahDataset(h5_file, downsample=3)


def test_make_loaders_split(h5_file):
    train_loader, val_loader = make_loaders(h5_file, batch_size=2, val_fraction=0.34, seed=0)
    assert len(train_loader.dataset) == 4
    assert len(val_loader.dataset) == 2

    batch = next(iter(train_loader))
    assert tuple(batch["rho_hat"].shape) == (2, 16, 16, 16)
    assert tuple(batch["aspect"].shape) == (2, 3)


def test_load_cheetah_sc_batch_and_processors(tmp_path):
    train_loader, test_loaders, data_processor = _load(tmp_path)

    # Keys von test_loaders = Aufloesungen
    assert set(test_loaders.keys()) == {16, 32}

    # Ein Batch aus dem Train-Loader: Dict {"x","y"} mit Kanal-Dimension
    batch = next(iter(train_loader))
    assert set(batch.keys()) == {"x", "y"}
    assert tuple(batch["x"].shape) == (2, 4, 16, 16, 16), batch["x"].shape
    assert tuple(batch["y"].shape) == (2, 1, 16, 16, 16), batch["y"].shape
    assert batch["x"].dtype == torch.float32
    assert batch["y"].dtype == torch.float32

    test_batch = next(iter(test_loaders[32]))
    assert tuple(test_batch["y"].shape) == (2, 1, 32, 32, 32)

    # Normalisierer sind gefittet (encode_output=True -> out_normalizer)
    assert data_processor.out_normalizer is not None, "out_normalizer fehlt"
    assert data_processor.out_normalizer.mean is not None
    assert data_processor.out_normalizer.std is not None


def test_load_cheetah_sc_reuses_existing_files(tmp_path):
    """Bei vorhandenen Dateien darf nicht neu generiert werden."""
    _load(tmp_path)
    assert missing_files(tmp_path, 16, [16, 32]) == []

    path = tmp_path / f"{DATASET_NAME}_train_16.h5"
    mtime_before = path.stat().st_mtime_ns
    _load(tmp_path)
    assert path.stat().st_mtime_ns == mtime_before, "vorhandene Datei wurde neu generiert"
