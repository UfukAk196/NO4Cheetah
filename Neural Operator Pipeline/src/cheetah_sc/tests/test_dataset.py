"""G4+G5-Abnahmetest: Dataset, Lade-Funktion und DataLoader.

Prueft (Plan G4+G5):
- ``CheetahDataset(PTDataset)``: train_db[0] liefert Dict mit
  x (4,res,res,res) und y (1,res,res,res)
- ``load_cheetah_sc``: ein Batch aus ``train_loader`` hat Form
  (B,4,res,res,res)/(B,1,res,res,res); ``test_loaders`` hat die Aufloesungen
  als Keys; Normalisierer sind gefittet.
"""

from __future__ import annotations

import torch

from cheetah_sc.dataset import CheetahDataset, load_cheetah_sc
from cheetah_sc.generate import DATASET_NAME


SMALL_RANGES = {
    "sigma_x": {"min": 5.0e-4, "max": 1.5e-3},
    "sigma_y": {"min": 5.0e-4, "max": 1.5e-3},
    "sigma_tau": {"min": 1.0e-3, "max": 5.0e-3},
    "energy": {"min": 5.0e8, "max": 1.5e9},
    "total_charge": {"min": 1.0e-10, "max": 2.0e-9},
    "n_particles": {"min": 1.0e4, "max": 2.0e4},
}


def test_cheetah_dataset_indexing(tmp_path):
    n_train, n_tests, train_res = 4, [2, 2], 16
    test_res = [16, 32]
    dataset = CheetahDataset(
        root_dir=tmp_path,
        n_train=n_train,
        n_tests=n_tests,
        batch_size=2,
        test_batch_sizes=[2, 2],
        train_resolution=train_res,
        test_resolutions=test_res,
        generate=True,
        seed=5,
        ranges=SMALL_RANGES,
    )

    # Dateien wurden generiert
    assert (tmp_path / f"{DATASET_NAME}_train_{train_res}.pt").exists()
    for res in test_res:
        assert (tmp_path / f"{DATASET_NAME}_test_{res}.pt").exists()

    sample = dataset.train_db[0]
    assert isinstance(sample, dict)
    assert set(sample.keys()) == {"x", "y"}
    assert tuple(sample["x"].shape) == (4, 16, 16, 16), sample["x"].shape
    assert tuple(sample["y"].shape) == (1, 16, 16, 16), sample["y"].shape
    assert sample["x"].dtype == torch.float32
    assert sample["y"].dtype == torch.float32

    # test_dbs hat die Aufloesungen als Keys
    assert set(dataset.test_dbs.keys()) == {16, 32}


def test_load_cheetah_sc_batch_and_processors(tmp_path):
    train_loader, test_loaders, data_processor = load_cheetah_sc(
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

    # Keys von test_loaders = Aufloesungen
    assert set(test_loaders.keys()) == {16, 32}

    # Ein Batch aus dem Train-Loader: Dict {"x","y"} mit Kanal-Dimension
    batch = next(iter(train_loader))
    assert set(batch.keys()) == {"x", "y"}
    assert tuple(batch["x"].shape) == (2, 4, 16, 16, 16), batch["x"].shape
    assert tuple(batch["y"].shape) == (2, 1, 16, 16, 16), batch["y"].shape
    assert batch["x"].dtype == torch.float32
    assert batch["y"].dtype == torch.float32

    # Normalisierer sind gefittet (encode_output=True -> out_normalizer)
    assert data_processor.out_normalizer is not None, "out_normalizer fehlt"
    assert data_processor.out_normalizer.mean is not None
    assert data_processor.out_normalizer.std is not None


def test_cheetah_dataset_reuses_existing_files(tmp_path):
    """Bei vorhandenen Dateien darf nicht neu generiert werden.

    Nach dem ersten init sind keine Dateien mehr 'missing' -> ein erneuter
    init mit generate=True ruft generate_cheetah_sc nicht erneut auf; die
    Datei wird also nicht ueberschrieben.
    """
    n_train, n_tests, train_res = 4, [2, 2], 16
    test_res = [16, 32]

    dataset = CheetahDataset(
        root_dir=tmp_path,
        n_train=n_train,
        n_tests=n_tests,
        batch_size=2,
        test_batch_sizes=[2, 2],
        train_resolution=train_res,
        test_resolutions=test_res,
        generate=True,
        seed=5,
        ranges=SMALL_RANGES,
    )

    # Nach der Generierung: keine fehlende Datei mehr.
    assert dataset._missing_files(tmp_path, 16, [16, 32]) == []

    # Inhalt der Train-Datei einfrieren; erneuter init darf nicht neu schreiben.
    path = tmp_path / f"{DATASET_NAME}_train_{train_res}.pt"
    x_before = torch.load(path, weights_only=False)["x"].clone()

    CheetahDataset(
        root_dir=tmp_path,
        n_train=n_train,
        n_tests=n_tests,
        batch_size=2,
        test_batch_sizes=[2, 2],
        train_resolution=train_res,
        test_resolutions=test_res,
        generate=True,
        seed=5,
        ranges=SMALL_RANGES,
    )

    x_after = torch.load(path, weights_only=False)["x"]
    assert torch.equal(x_before, x_after), "vorhandene Datei wurde neu generiert"
