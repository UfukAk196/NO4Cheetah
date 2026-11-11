"""Acceptance test for ``dataset.CheetahDataset`` (``CheetahH5Dataset`` per file).

Checks that the .h5 files from ``generate`` load as 3D grids with the
expected channel dimension and normalizer shapes, that beam parameters
(``input_params``) become constant, standardized input channels, and that a minimal
training step runs end to end.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from neuralop import LpLoss, Trainer
from neuralop.models import TFNO
from neuralop.training import AdamW

from scno.data.dataset import ELECTRON_MASS_EV, CheetahDataset, load_cheetah_sc
from scno.data.generate import generate_cheetah_sc


def _make_data(tmp_path, ranges):
    generate_cheetah_sc(
        root=tmp_path,
        n_train=8,
        n_tests=[4, 2],
        train_resolution=16,
        test_resolutions=[16, 32],
        seed=0,
        ranges=ranges,
    )


INPUT_PARAMS = ("log_sigma_x", "log_sigma_y", "log_sigma_tau", "gamma")


def test_dataset_channel_dim(tmp_path, small_ranges):
    _make_data(tmp_path, small_ranges)

    # The files hold enough samples, so CheetahDataset generates nothing.
    ds = CheetahDataset(tmp_path, 4, [2, 2], 16, [16, 32])
    assert ds.train_db[0]["x"].shape == (1, 16, 16, 16)
    assert ds.train_db[0]["y"].shape == (1, 16, 16, 16)
    assert len(ds.test_dbs[32]) == 2
    assert sorted(ds.test_dbs.keys()) == [16, 32]


def test_normalizer_shapes(tmp_path, small_ranges):
    _make_data(tmp_path, small_ranges)

    ds = CheetahDataset(
        tmp_path,
        4,
        [2, 2],
        16,
        [16, 32],
        encode_input=True,
        encode_output=True,
    )
    # channel-wise: one mean/std per channel, broadcast over the spatial dims
    assert ds.data_processor.in_normalizer.mean.shape == (1, 1, 1, 1, 1)
    assert ds.data_processor.out_normalizer.mean.shape == (1, 1, 1, 1, 1)


def test_feature_channels(tmp_path, small_ranges):
    _make_data(tmp_path, small_ranges)

    ds = CheetahDataset(tmp_path, 4, [2, 2], 16, [16, 32], input_params=INPUT_PARAMS)
    for db in (ds.train_db, *ds.test_dbs.values()):
        res = db[0]["x"].shape[-1]
        assert db[0]["x"].shape == (1 + len(INPUT_PARAMS), res, res, res)
        assert db[0]["y"].shape == (1, res, res, res)

    # rho first, then one constant channel per parameter, in raw units
    item, params = ds.train_db[1], ds.train_db.data["params"]
    assert torch.equal(item["x"][0], torch.from_numpy(ds.train_db.data["rho"][1]))
    expected = [
        np.log(params["sigma_x"][1]),
        np.log(params["sigma_y"][1]),
        np.log(params["sigma_tau"][1]),
        params["energy"][1] / ELECTRON_MASS_EV,
    ]
    for channel, value in zip(item["x"][1:], expected):
        assert torch.allclose(channel, torch.full_like(channel, float(value)))


@pytest.mark.parametrize(
    "encoding, stats_shape", [("channel-wise", (1, 5, 1, 1, 1)), ("pixel-wise", (1, 5, 16, 16, 16))]
)
def test_input_normalizer_with_features(tmp_path, small_ranges, encoding, stats_shape):
    _make_data(tmp_path, small_ranges)

    ds = CheetahDataset(
        tmp_path, 8, [2, 2], 16, [16, 32], input_params=INPUT_PARAMS, encoding=encoding
    )
    in_normalizer = ds.data_processor.in_normalizer
    assert in_normalizer.mean.shape == stats_shape
    assert in_normalizer.std.shape == stats_shape

    if encoding == "channel-wise":
        # every input channel standardized over the training set (train mode: x and y)
        batch = torch.utils.data.default_collate([ds.train_db[i] for i in range(8)])
        ds.data_processor.train()
        x = ds.data_processor.preprocess(batch)["x"]
        mean, std = x.mean(dim=(0, 2, 3, 4)), x.std(dim=(0, 2, 3, 4))
        assert torch.allclose(mean, torch.zeros(5), atol=1e-4), mean
        # features: std over 8 samples (unbiased) vs. over 8 x 16^3 values -> sqrt(7/8)
        assert torch.allclose(std, torch.ones(5), atol=0.1), std


@pytest.mark.parametrize("input_params", [(), INPUT_PARAMS])
def test_train_smoke(tmp_path, small_ranges, input_params):
    _make_data(tmp_path, small_ranges)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_loader, test_loaders, data_processor = load_cheetah_sc(
        n_train=8,
        batch_size=4,
        n_tests=[4, 2],
        test_resolutions=[16, 32],
        test_batch_sizes=[2, 2],
        train_resolution=16,
        data_root=tmp_path,
        input_params=input_params,
        seed=0,
        ranges=small_ranges,
    )
    data_processor = data_processor.to(device)

    model = TFNO(
        n_modes=(5, 5, 5),
        in_channels=1 + len(input_params),
        out_channels=1,
        hidden_channels=8,
        n_layers=2,
        domain_padding=0.125,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=2.0e-4, weight_decay=4.0e-6)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0, total_iters=0)
    l2loss = LpLoss(d=3, p=2)

    trainer = Trainer(
        model=model,
        n_epochs=1,
        device=device,
        data_processor=data_processor,
        wandb_log=False,
        eval_interval=1,
        use_distributed=False,
        verbose=False,
    )
    trainer.train(
        train_loader=train_loader,
        test_loaders=test_loaders,
        optimizer=optimizer,
        scheduler=scheduler,
        regularizer=False,
        training_loss=l2loss,
        eval_losses={"l2": l2loss},
    )
