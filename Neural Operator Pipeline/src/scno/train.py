"""
Trainingsschleife.

Bewusst schlicht gehalten: eine Datei, keine Framework-Magie, alles sichtbar.
Wenn das Projekt waechst, kann man das jederzeit gegen Lightning tauschen --
aber am Anfang willst du jede Zeile sehen koennen, die deine Zahlen erzeugt.

Aufruf:
    python -m scno.train --config configs/default.yaml
    python -m scno.train --config configs/default.yaml --override training.epochs=200
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from .data.dataset import make_loaders
from .losses import build_loss, relative_l2
from .models.fno import build_model
from .normalization import Scaling, build_input
from .utils import count_parameters, load_config, pick_device, save_checkpoint, set_seed


def batch_to_input(batch: dict, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, Scaling]:
    rho = batch["rho_hat"].to(device).float()
    phi = batch["phi_hat"].to(device).float()
    scaling = Scaling(
        total_charge=batch["total_charge"].to(device).float(),
        reference_length=batch["reference_length"].to(device).float(),
        aspect=batch["aspect"].to(device).float(),
    )
    x = build_input(rho, scaling)
    return x, phi[:, None], scaling


def run_epoch(model, loader, loss_fn, device, optimizer=None) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals: dict[str, float] = {}
    count = 0

    for batch in loader:
        x, target, scaling = batch_to_input(batch, device)
        with torch.set_grad_enabled(training):
            pred = model(x)
            loss, metrics = loss_fn(
                pred, target, rho_hat=x[:, :1], aspect=scaling.aspect
            )
        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        n = x.shape[0]
        count += n
        totals["loss"] = totals.get("loss", 0.0) + float(loss.detach()) * n
        for k, v in metrics.items():
            totals[k] = totals.get(k, 0.0) + float(v) * n

    return {k: v / max(count, 1) for k, v in totals.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--override", nargs="*", default=[], help="z.B. training.epochs=50")
    args = p.parse_args()

    cfg = load_config(args.config)
    for item in args.override:
        key, value = item.split("=", 1)
        node = cfg
        *path, last = key.split(".")
        for part in path:
            node = node[part]
        node[last] = yaml_value(value)

    set_seed(cfg["seed"])
    device = pick_device(cfg["device"])
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader = make_loaders(
        cfg["data"]["path"],
        batch_size=cfg["training"]["batch_size"],
        val_fraction=cfg["data"]["val_fraction"],
        num_workers=cfg["data"]["num_workers"],
        seed=cfg["seed"],
    )

    model = build_model(cfg["model"]).to(device)
    print(f"Modell: {cfg['model']['name']}, {count_parameters(model):,} Parameter, Geraet {device}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["training"]["epochs"]
    )
    loss_fn = build_loss(cfg["loss"])

    logger = None
    if cfg["logging"]["wandb"]:
        import wandb

        logger = wandb.init(project=cfg["logging"]["project"], config=cfg)

    history = []
    best = float("inf")
    for epoch in range(1, cfg["training"]["epochs"] + 1):
        t0 = time.time()
        train_metrics = run_epoch(model, train_loader, loss_fn, device, optimizer)
        val_metrics = run_epoch(model, val_loader, loss_fn, device, None)
        scheduler.step()

        row = {
            "epoch": epoch,
            "lr": scheduler.get_last_lr()[0],
            "seconds": time.time() - t0,
            **{f"train/{k}": v for k, v in train_metrics.items()},
            **{f"val/{k}": v for k, v in val_metrics.items()},
        }
        history.append(row)
        if logger:
            logger.log(row)

        if epoch % cfg["logging"]["print_every"] == 0 or epoch == 1:
            print(
                f"Epoche {epoch:4d} | train {train_metrics['loss']:.4e} "
                f"| val {val_metrics['loss']:.4e} "
                f"| feld {val_metrics.get('loss_field', float('nan')):.4e} "
                f"| grad {val_metrics.get('loss_grad', float('nan')):.4e} "
                f"| {row['seconds']:.1f}s"
            )

        if val_metrics["loss"] < best:
            best = val_metrics["loss"]
            save_checkpoint(out_dir / "best.pt", model, optimizer, epoch, cfg, best)

    save_checkpoint(out_dir / "last.pt", model, optimizer, cfg["training"]["epochs"], cfg, best)
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    print(f"Bester Validierungsverlust: {best:.4e} -> {out_dir/'best.pt'}")


def yaml_value(text: str):
    import yaml

    return yaml.safe_load(text)


if __name__ == "__main__":
    main()
