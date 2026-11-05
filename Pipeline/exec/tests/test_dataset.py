"""Acceptance test for ``dataset.CheetahDataset`` (``CheetahH5Dataset`` per file).

Checks that the .h5 files from ``generate`` load as 3D grids with the
expected channel dimension and normalizer shapes, and that a minimal training step
runs end to end.
"""

from __future__ import annotations

import torch
from neuralop import LpLoss, Trainer
from neuralop.models import TFNO
from neuralop.training import AdamW

from scno.data.dataset import CheetahDataset, load_cheetah_sc
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


def test_train_smoke(tmp_path, small_ranges):
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
        seed=0,
        ranges=small_ranges,
    )
    data_processor = data_processor.to(device)

    model = TFNO(
        n_modes=(5, 5, 5),
        in_channels=1,
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
