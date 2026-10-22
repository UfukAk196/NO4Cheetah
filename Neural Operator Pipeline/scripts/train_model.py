"""
Trains the FNO on the Cheetah space-charge dataset with the neuralop ``Trainer``.

All settings are the constants below; ``evaluate_model.py`` and ``inspect_scaling.py``
import them, so that evaluation and diagnosis see exactly the training setup. Missing
data files are generated on the first run.

Writes only:
    models/<MODEL_NAME>.pth                  state dict of the final model (incl. the
                                             neuralop ``_metadata`` with the init kwargs)
    figures/loss_curves_<MODEL_NAME>.png     loss curves
    tables/<MODEL_NAME>_history.csv          losses per epoch

Call: python scripts/train_model.py
"""

from pathlib import Path

import torch
from neuralop.models import FNO
from neuralop.utils import count_model_params

from scno.data.dataset import load_cheetah_sc
from scno.evaluate import plot_losses
from scno.train import HistoryTrainer
from scno.utils.common import pick_device, save_csv, set_seed
from scno.utils.losses import SobolevLoss

ROOT = Path(__file__).resolve().parents[1]
MODEL_NAME = "fno_cheetah_space_charge"
MODEL_PATH = ROOT / "models" / f"{MODEL_NAME}.pth"

SEED = 0
DEVICE = "auto"  # auto | cpu | cuda | mps

# Keyword arguments of scno.data.dataset.load_cheetah_sc.
DATA = {
    "data_root": ROOT / "data",
    "seed": 0,
    "nondim": True,  # True: rho_hat -> phi_hat, False: rho [C/m^3] -> phi [V] (ablation)
    # "encode_output": unset = not nondim (Gaussian normalizer only for the raw ablation)
    "n_train": 16,  # TODO: target 5k-50k samples
    "n_tests": [4, 4, 4],
    "train_resolution": 16,
    "test_resolutions": [16, 32, 64],
    "batch_size": 4,
    "test_batch_sizes": [4, 4, 4],
    "num_workers": 0,  # keep at 0 on Windows
    # "ranges": override of scno.data.sampling.DEFAULT_RANGES, same shape
}

# Keyword arguments of neuralop.models.FNO.
MODEL = {
    "n_modes": (8, 8, 8),  # kept Fourier modes per axis
    "in_channels": 4,  # rho_hat + 3 aspect ratios log(E_i/L)
    "out_channels": 1,  # phi_hat
    "hidden_channels": 16,
    "n_layers": 3,
    "domain_padding": 0.25,  # padding against the open boundaries
    "factorization": None,  # e.g. "tucker" for fewer weights
    "rank": 1.0,
}

# Keyword arguments of scno.utils.losses.SobolevLoss.
LOSS = {
    "gradient_weight": 1.0,  # weight of the gradient error -- the kick is a gradient
    "physics_weight": 0.0,  # > 0 adds the Poisson residual (PINN part)
}

TRAINING = {
    "epochs": 1000,
    "learning_rate": 1.0e-4,
    "weight_decay": 1.0e-4,
}

EVAL_INTERVAL = 100  # epochs between evaluations on the test resolutions
WANDB = False  # log to Weights & Biases (project "space-charge-neural-operator")


def history_rows(history: dict) -> list[dict]:
    """``HistoryTrainer.history`` -> one row per epoch, train and eval columns merged.

    Eval columns stay empty in epochs without evaluation (every ``EVAL_INTERVAL``).
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


def main() -> int:
    set_seed(SEED)
    device = pick_device(DEVICE)

    train_loader, test_loaders, data_processor = load_cheetah_sc(**DATA)
    if LOSS["physics_weight"] > 0.0 and (
        not DATA["nondim"]
        or data_processor.in_normalizer is not None
        or data_processor.out_normalizer is not None
    ):
        raise ValueError(
            "physics_weight > 0 needs nondimensionalized data without normalizers: the "
            "Poisson residual reads rho_hat and log(E_i/L) from x and assumes phi_hat."
        )

    model = FNO(**MODEL).to(device)
    print(f"Model: FNO, {count_model_params(model):,} parameters, device {device}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=TRAINING["learning_rate"],
        weight_decay=TRAINING["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=TRAINING["epochs"]
    )

    if WANDB:
        import wandb

        wandb.init(
            project="space-charge-neural-operator",
            config={"data": DATA, "model": MODEL, "loss": LOSS, "training": TRAINING},
        )

    trainer = HistoryTrainer(
        model=model,
        n_epochs=TRAINING["epochs"],
        device=device,
        data_processor=data_processor,
        eval_interval=EVAL_INTERVAL,
        wandb_log=WANDB,
        verbose=True,
    )
    trainer.train(
        train_loader=train_loader,
        test_loaders=test_loaders,
        optimizer=optimizer,
        scheduler=scheduler,
        training_loss=SobolevLoss(**LOSS),
        eval_losses={
            "h1": SobolevLoss(gradient_weight=1.0),
            "l2": SobolevLoss(gradient_weight=0.0),
        },
    )

    # neuralop's state_dict() carries the init kwargs as `_metadata`, so the model can
    # be rebuilt from this one file (see evaluate_model.load_model).
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Model saved: {MODEL_PATH}")

    plot_losses(trainer.history, save_dir=ROOT / "figures", tag=MODEL_NAME)
    save_csv(history_rows(trainer.history), ROOT / "tables" / f"{MODEL_NAME}_history.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
