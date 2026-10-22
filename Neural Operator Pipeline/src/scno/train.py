"""
Training with the neuraloperator ``Trainer``.

One path for every run: config -> ``load_cheetah_sc`` -> ``FNO`` -> ``HistoryTrainer``
-> neuralop checkpoint. Copy configs for experiments instead of editing them, so that
numbers and settings stay together.

Call:
    python -m scno.train --config configs/default.yaml
    python -m scno.train --config configs/smoke.yaml --override training.epochs=5
    python -m scno.train --config configs/smoke.yaml --override data.nondim=false

Output in ``output_dir``: ``config.yaml`` (the resolved config), ``best_model_*`` (best
checkpoint by the first test resolution's h1 loss), ``model_*`` (final checkpoint),
``history.json`` and the loss curves.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml
from neuralop import Trainer
from neuralop.models import FNO
from neuralop.utils import count_model_params

from scno.data.dataset import load_cheetah_sc
from scno.utils.common import load_config, pick_device, set_seed
from scno.utils.losses import SobolevLoss


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


def apply_overrides(cfg: dict, overrides: list[str]) -> dict:
    """Applies ``a.b.c=value`` overrides (values parsed as YAML) to ``cfg`` in place."""
    for item in overrides:
        key, value = item.split("=", 1)
        node = cfg
        *path, last = key.split(".")
        for part in path:
            node = node[part]
        node[last] = yaml.safe_load(value)
    return cfg


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--override", nargs="*", default=[], help="e.g. training.epochs=50")
    p.add_argument("--plot", action="store_true", help="plot predictions after training")
    args = p.parse_args()

    cfg = apply_overrides(load_config(args.config), args.override)
    set_seed(cfg["seed"])
    device = pick_device(cfg["device"])
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    data_cfg = cfg["data"]
    train_loader, test_loaders, data_processor = load_cheetah_sc(**data_cfg)
    if cfg["loss"].get("physics_weight", 0.0) > 0.0 and (
        not data_cfg["nondim"]
        or data_processor.in_normalizer is not None
        or data_processor.out_normalizer is not None
    ):
        raise ValueError(
            "loss.physics_weight > 0 needs nondimensionalized data without normalizers: "
            "the Poisson residual reads rho_hat and log(E_i/L) from x and assumes phi_hat."
        )

    model = FNO(**cfg["model"]).to(device)
    print(f"Model: FNO, {count_model_params(model):,} parameters, device {device}")

    training = cfg["training"]
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=training["learning_rate"], weight_decay=training["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=training["epochs"])
    training_loss = SobolevLoss(**cfg["loss"])
    eval_losses = {
        "h1": SobolevLoss(gradient_weight=1.0),
        "l2": SobolevLoss(gradient_weight=0.0),
    }

    if cfg["logging"]["wandb"]:
        import wandb

        wandb.init(project=cfg["logging"]["project"], config=cfg)

    trainer = HistoryTrainer(
        model=model,
        n_epochs=training["epochs"],
        device=device,
        data_processor=data_processor,
        eval_interval=cfg["logging"]["eval_interval"],
        wandb_log=cfg["logging"]["wandb"],
        verbose=True,
    )
    trainer.train(
        train_loader=train_loader,
        test_loaders=test_loaders,
        optimizer=optimizer,
        scheduler=scheduler,
        training_loss=training_loss,
        eval_losses=eval_losses,
        save_best=f"{data_cfg['test_resolutions'][0]}_h1",
        save_dir=out_dir,
    )

    model.save_checkpoint(out_dir, "model")
    (out_dir / "history.json").write_text(json.dumps(trainer.history, indent=2))
    print(f"Checkpoints in {out_dir}")

    from scno.evaluate import plot_losses, plot_predictions

    plot_losses(trainer.history, save_dir=out_dir, tag=out_dir.name)
    if args.plot:
        for res, loader in test_loaders.items():
            plot_predictions(
                model,
                next(iter(loader)),
                data_processor,
                data_cfg["nondim"],
                tag=f"{out_dir.name}_res{res}",
            )


if __name__ == "__main__":
    main()
