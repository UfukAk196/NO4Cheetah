"""
Training with the neuraloperator ``Trainer``.

The runs (settings in ``scno.config``, data, saving) are ``scripts/train_model.py`` and
``scripts/tune_model.py``; this module provides the training they share: ``fit``, the
``Trainer`` that records its history, and the history as table rows.
"""

from __future__ import annotations

import torch
from neuralop import Trainer
from neuralop.models import FNO
from neuralop.utils import count_model_params

from .utils.losses import SobolevLoss


class HistoryTrainer(Trainer):
    """``Trainer`` that records the metrics of every epoch.

    The neuralop ``Trainer`` only prints the values to stdout and does not store them.
    Here ``log_training`` (every epoch) and ``log_eval`` (every ``eval_interval``
    epochs) are overridden and the values are additionally stored in ``self.history``:

        history["train"] = {"epoch": [...], "avg_loss": [...], "train_err": [...]}
        history["eval"]  = {"epoch": [...], "<res>_h1": [...], "<res>_l2": [...]}
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.history = {"train": {}, "eval": {}}

    def log_training(self, epoch, time, avg_loss, train_err, avg_lasso_loss=None, lr=None):
        super().log_training(
            epoch=epoch,
            time=time,
            avg_loss=avg_loss,
            train_err=train_err,
            avg_lasso_loss=avg_lasso_loss,
            lr=lr,
        )
        train_hist = self.history["train"]
        train_hist.setdefault("epoch", []).append(epoch)
        train_hist.setdefault("avg_loss", []).append(float(avg_loss))
        train_hist.setdefault("train_err", []).append(float(train_err))
        train_hist.setdefault("lr", []).append(float(lr) if lr is not None else float("nan"))

    def log_eval(self, epoch, eval_metrics):
        super().log_eval(epoch=epoch, eval_metrics=eval_metrics)
        eval_hist = self.history["eval"]
        eval_hist.setdefault("epoch", []).append(epoch)
        for metric, value in eval_metrics.items():
            if isinstance(value, torch.Tensor):
                value = value.item()
            if isinstance(value, (int, float)):
                eval_hist.setdefault(metric, []).append(float(value))


def fit(
    loaders: tuple,
    device: torch.device,
    model_kwargs: dict,
    loss_kwargs: dict,
    training: dict,
    eval_interval: int,
    nondim: bool = True,
    wandb_log: bool = False,
    trainer_cls: type[HistoryTrainer] = HistoryTrainer,
    **trainer_kwargs,
) -> tuple[FNO, HistoryTrainer]:
    """Builds the FNO and trains it with AdamW, a cosine schedule and ``SobolevLoss``.

    Shared by ``train_model.py`` and ``tune_model.py`` (which passes its own settings
    and a pruning ``trainer_cls``), so tuning runs exactly the training code path.
    Evaluated with the H1 (field + gradient) and L2 Sobolev losses on every test loader.

    :param loaders: ``(train_loader, test_loaders, data_processor)`` of ``load_cheetah_sc``.
    :param device: Device the model is trained on.
    :param model_kwargs: Keyword arguments of ``neuralop.models.FNO``.
    :param loss_kwargs: Keyword arguments of the training ``SobolevLoss``.
    :param training: ``{"epochs", "learning_rate", "weight_decay"}``.
    :param eval_interval: Epochs between evaluations.
    :param nondim: Whether the data is nondimensionalized (``DATA["nondim"]``).
    :param wandb_log: Log to Weights & Biases (``wandb.init`` must have been called).
    :param trainer_cls: ``HistoryTrainer`` or a subclass; gets ``trainer_kwargs``.
    :return: The trained model and the trainer with its ``history``.
    """
    train_loader, test_loaders, data_processor = loaders
    if loss_kwargs["physics_weight"] > 0.0 and (
        not nondim
        or data_processor.in_normalizer is not None
        or data_processor.out_normalizer is not None
    ):
        raise ValueError(
            "physics_weight > 0 needs nondimensionalized data without normalizers: the "
            "Poisson residual reads rho_hat and log(E_i/L) from x and assumes phi_hat."
        )

    model = FNO(**model_kwargs).to(device)
    print(f"Model: FNO, {count_model_params(model):,} parameters, device {device}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training["learning_rate"],
        weight_decay=training["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5
    )

    trainer = trainer_cls(
        model=model,
        n_epochs=training["epochs"],
        device=device,
        data_processor=data_processor,
        eval_interval=eval_interval,
        wandb_log=wandb_log,
        verbose=True,
        **trainer_kwargs,
    )
    trainer.train(
        train_loader=train_loader,
        test_loaders=test_loaders,
        optimizer=optimizer,
        scheduler=scheduler,
        training_loss=SobolevLoss(**loss_kwargs),
        eval_losses={
            "h1": SobolevLoss(gradient_weight=1.0),
            "l2": SobolevLoss(gradient_weight=0.0),
        },
    )
    return model, trainer


def history_rows(history: dict) -> list[dict]:
    """``HistoryTrainer.history`` -> one row per epoch, train and eval columns merged.

    Eval columns stay empty in epochs without evaluation (every ``eval_interval``).
    """
    train, evaluation = history["train"], history["eval"]
    eval_columns = [key for key in evaluation if key != "epoch"]
    eval_by_epoch = {
        epoch: {key: evaluation[key][i] for key in eval_columns}
        for i, epoch in enumerate(evaluation.get("epoch", []))
    }
    return [
        {
            **{key: values[i] for key, values in train.items()},
            **eval_by_epoch.get(epoch, dict.fromkeys(eval_columns, "")),
        }
        for i, epoch in enumerate(train.get("epoch", []))
    ]
