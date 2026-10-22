"""
Training with the neuraloperator ``Trainer``.

The training run itself (settings, data, model, saving) is ``scripts/train_model.py``;
this module only provides the ``Trainer`` that records its history.
"""

from __future__ import annotations

import torch
from neuralop import Trainer


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
