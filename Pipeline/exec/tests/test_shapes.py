"""Shapes of the in- and output components of ``CheetahDataset`` with beam parameters.

Generates a small file (n=3, res 16) and prints and checks the shape of every
component along the way from the .h5 file to the model input:
- in the file:     rho/phi (N, res, res, res), params (N,)
- one sample:      x = [rho, *features] (1 + k, res, res, res), y = phi (1, res, res, res)
- a batch:         (B, 1 + k, res, res, res) in, (B, 1, res, res, res) out

Call with ``-s`` to see the printed table: pytest exec/tests/test_shapes.py -s
"""

from __future__ import annotations

from torch.utils.data import default_collate

from scno.data.dataset import CheetahDataset

N, RES = 3, 16
PARAM_KEYS = ("sigma_x", "sigma_y", "sigma_tau", "energy")
INPUT_PARAMS = ("log_sigma_x", "log_sigma_y", "log_sigma_tau", "gamma")
K = len(INPUT_PARAMS)


def test_component_shapes(tmp_path, small_ranges):
    ds = CheetahDataset(
        tmp_path, N, [], RES, [], input_params=INPUT_PARAMS, ranges=small_ranges
    ).train_db
    item = ds[0]
    params = ds.data["params"]

    rows = [
        ("rho  file", ds.data["rho"].shape),
        ("phi  file", ds.data["phi"].shape),
        *((f"{k}  file", params[k].shape) for k in PARAM_KEYS),
        *((f"{k}  sample", params[k][0].shape) for k in PARAM_KEYS),
        ("x = [rho, *features]  sample", tuple(item["x"].shape)),
        ("y = phi  sample", tuple(item["y"].shape)),
    ]

    assert ds.data["rho"].shape == (N, RES, RES, RES)
    assert ds.data["phi"].shape == (N, RES, RES, RES)
    for k in PARAM_KEYS:
        assert params[k].shape == (N,), k
        assert params[k][0].shape == (), k  # one scalar per sample
        assert params[k][0] > 0, k
    assert item["x"].shape == (1 + K, RES, RES, RES)
    assert item["y"].shape == (1, RES, RES, RES)

    batch = default_collate([ds[i] for i in range(N)])
    rows += [
        ("model input  batch", tuple(batch["x"].shape)),
        ("model output batch", tuple(batch["y"].shape)),
    ]
    assert batch["x"].shape == (N, 1 + K, RES, RES, RES)
    assert batch["y"].shape == (N, 1, RES, RES, RES)

    width = max(len(name) for name, _ in rows)
    print()
    for name, shape in rows:
        print(f"  {name:<{width}}  {shape}")
