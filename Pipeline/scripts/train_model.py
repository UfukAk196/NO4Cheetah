"""
Trains the FNO on the Cheetah space-charge dataset with the neuralop ``Trainer``.

All settings are the constants in ``scno.config``. Missing data files are generated on
the first run.

Every run gets an ID ``<date>_<NN>`` (e.g. ``2026-10-26_01``, ``NN`` counts the runs of
that day) and writes only:
    models/<MODEL_NAME>_<RUN>.pth   state dict of the final model (incl. the neuralop
                                    ``_metadata`` with the init kwargs)
    figures/<RUN>/loss_curves.png   loss curves
    tables/<RUN>/history.csv        losses per epoch

Call: python scripts/train_model.py
"""

import torch

from scno.config import (
    DATA,
    DEVICE,
    EVAL_INTERVAL,
    FIGURES_DIR,
    LOSS,
    MODEL,
    RUN_DIRS,
    SEED,
    TABLES_DIR,
    TRAINING,
    WANDB,
    model_path,
)
from scno.data.dataset import load_cheetah_sc
from scno.evaluate import plot_losses
from scno.train import fit, history_rows
from scno.utils.common import new_run_id, pick_device, save_csv, set_seed


def main() -> int:
    set_seed(SEED)
    run = new_run_id(*RUN_DIRS)
    print(f"Run: {run}")
    device = pick_device(DEVICE)

    if WANDB:
        import wandb

        wandb.init(
            entity="ufuk-akkaya-deutsches-elektronen-synchrotron-desy",
            project="space-charge-neural-operator",
            config={"data": DATA, "model": MODEL, "loss": LOSS, "training": TRAINING,},
        )

    model, trainer = fit(
        load_cheetah_sc(**DATA),
        device,
        MODEL,
        LOSS,
        TRAINING,
        EVAL_INTERVAL,
        nondim=DATA["nondim"],
        wandb_log=WANDB,
    )

    # neuralop's state_dict() carries the init kwargs as `_metadata`, so the model can
    # be rebuilt from this one file (see evaluate_model.load_model).
    path = model_path(run)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"Model saved: {path}")

    plot_losses(trainer.history, save_dir=FIGURES_DIR / run)
    save_csv(history_rows(trainer.history), TABLES_DIR / run / "history.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
