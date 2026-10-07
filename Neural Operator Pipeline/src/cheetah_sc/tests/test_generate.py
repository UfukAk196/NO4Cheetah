"""G3-Abnahmetest: Datei-Writer erzeugt korrekte .pt-Dateien.

Prueft (Plan G3):
- kleiner Lauf (n_train=4, n_tests=[2,2], res 16/32) legt die Dateien an
- Keys x, y, meta vorhanden
- Formen (N,4,res,res,res)/(N,1,res,res,res), dtype float32
- keine NaN/Inf
- Train- und Test-Seeds ueberschneiden sich nicht
"""

from __future__ import annotations

import torch
from cheetah_sc.generate import generate_cheetah_sc


SMALL_RANGES = {
    "sigma_x": {"min": 5.0e-4, "max": 1.5e-3},
    "sigma_y": {"min": 5.0e-4, "max": 1.5e-3},
    "sigma_tau": {"min": 1.0e-3, "max": 5.0e-3},
    "energy": {"min": 5.0e8, "max": 1.5e9},
    "total_charge": {"min": 1.0e-10, "max": 2.0e-9},
    "n_particles": {"min": 1.0e4, "max": 3.0e4},
}


def test_generate_writes_files(tmp_path):
    n_train, n_tests = 4, [2, 2]
    train_res, test_res = 16, [16, 32]

    written = generate_cheetah_sc(
        root=tmp_path,
        n_train=n_train,
        n_tests=n_tests,
        train_resolution=train_res,
        test_resolutions=test_res,
        seed=7,
        ranges=SMALL_RANGES,
    )

    expected_names = {
        "cheetah_sc_train_16.pt",
        "cheetah_sc_test_16.pt",
        "cheetah_sc_test_32.pt",
    }
    got_names = {p.name for p in written}
    assert got_names == expected_names, f"erwartete Dateien: {expected_names - got_names}"
    for name in expected_names:
        assert (tmp_path / name).exists()


def test_file_contents_and_shapes(tmp_path):
    n_train, n_tests = 4, [2, 2]
    train_res, test_res = 16, [16, 32]
    generate_cheetah_sc(
        root=tmp_path,
        n_train=n_train,
        n_tests=n_tests,
        train_resolution=train_res,
        test_resolutions=test_res,
        seed=7,
        ranges=SMALL_RANGES,
    )

    # Train-Datei: (n_train, 4, 16, 16, 16) / (n_train, 1, 16, 16, 16)
    train = torch.load(tmp_path / "cheetah_sc_train_16.pt", weights_only=False)
    assert set(train.keys()) == {"x", "y", "meta"}
    assert tuple(train["x"].shape) == (4, 4, 16, 16, 16), train["x"].shape
    assert tuple(train["y"].shape) == (4, 1, 16, 16, 16), train["y"].shape
    assert train["x"].dtype == torch.float32
    assert train["y"].dtype == torch.float32
    assert torch.isfinite(train["x"]).all()
    assert torch.isfinite(train["y"]).all()

    # Test-32-Datei: (2, 4, 32, 32, 32) / (2, 1, 32, 32, 32)
    test32 = torch.load(tmp_path / "cheetah_sc_test_32.pt", weights_only=False)
    assert tuple(test32["x"].shape) == (2, 4, 32, 32, 32), test32["x"].shape
    assert tuple(test32["y"].shape) == (2, 1, 32, 32, 32), test32["y"].shape
    assert test32["x"].dtype == torch.float32
    assert test32["y"].dtype == torch.float32
    assert torch.isfinite(test32["x"]).all()
    assert torch.isfinite(test32["y"]).all()


def test_train_test_seeds_disjoint(tmp_path):
    n_train, n_tests = 4, [2, 2]
    train_res, test_res = 16, [16, 32]
    base_seed = 7
    generate_cheetah_sc(
        root=tmp_path,
        n_train=n_train,
        n_tests=n_tests,
        train_resolution=train_res,
        test_resolutions=test_res,
        seed=base_seed,
        ranges=SMALL_RANGES,
    )

    def seeds(fname):
        return [p["seed"] for p in torch.load(tmp_path / fname, weights_only=False)["meta"]["params"]]

    train_seeds = set(seeds("cheetah_sc_train_16.pt"))
    test16_seeds = set(seeds("cheetah_sc_test_16.pt"))
    test32_seeds = set(seeds("cheetah_sc_test_32.pt"))

    assert len(train_seeds) == n_train
    assert len(test16_seeds) == n_tests[0]
    assert len(test32_seeds) == n_tests[1]
    # keine Ueberschneidung: weder train<->test noch zwischen Test-Aufloesungen
    assert train_seeds.isdisjoint(test16_seeds)
    assert train_seeds.isdisjoint(test32_seeds)
    assert test16_seeds.isdisjoint(test32_seeds)

    # Seeds sind genau die fortlaufenden base+global_index
    expected = set(range(base_seed, base_seed + n_train + sum(n_tests)))
    assert train_seeds | test16_seeds | test32_seeds == expected
