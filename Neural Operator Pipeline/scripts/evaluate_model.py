"""
Evaluates the model trained by ``train_model.py`` with the functions of ``scno.evaluate``.

Loads ``models/<MODEL_NAME>.pth`` with neuralop's ``load_state_dict`` (the model is
rebuilt from the ``_metadata`` neuralop stores in the state dict) and uses the data
settings of ``train_model.py``.

Writes only:
    tables/<MODEL_NAME>_evaluation.csv   accuracy, gradient fidelity, resolution study
    figures/...                          predictions per test resolution

Call: python scripts/evaluate_model.py
"""

from pathlib import Path

import torch
from neuralop.models import FNO
from train_model import DATA, DEVICE, MODEL_NAME, MODEL_PATH, ROOT

from scno.data.dataset import CheetahDataset, load_cheetah_sc
from scno.data.generate import dataset_path
from scno.evaluate import (
    evaluate,
    gradient_consistency,
    plot_normalization_comparison,
    plot_predictions,
    resolution_study,
)
from scno.utils.common import pick_device, save_csv

# Entries of neuralop's `_metadata` that are bookkeeping, not FNO arguments.
_METADATA_ONLY = ("_version", "_name", "args")


def load_model(path: Path, device: torch.device) -> FNO:
    """Rebuilds the FNO from the init kwargs in the state dict and loads the weights."""
    state_dict = torch.load(path, map_location=device, weights_only=False)
    kwargs = {k: v for k, v in state_dict["_metadata"].items() if k not in _METADATA_ONLY}
    model = FNO(**kwargs)
    model.load_state_dict(state_dict)
    return model.to(device)


def flatten(report: dict, prefix: str = "") -> list[dict]:
    """Nested report -> rows ``{"quantity": "a.b.c", "value": ...}``."""
    rows = []
    for key, value in report.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            rows += flatten(value, f"{name}.")
        else:
            rows.append({"quantity": name, "value": value})
    return rows


def main() -> int:
    device = pick_device(DEVICE)
    model = load_model(MODEL_PATH, device)
    nondim = DATA["nondim"]

    # The data processor is refitted on the training data, exactly as in training (it
    # only holds normalizers for nondim=False).
    _, test_loaders, data_processor = load_cheetah_sc(**{**DATA, "generate": False})
    data_processor = data_processor.to(device)

    data_path = dataset_path(DATA["data_root"], "test", DATA["train_resolution"])
    dataset = CheetahDataset(data_path, nondim=nondim)
    report = {
        "accuracy": evaluate(model, dataset, data_processor),
        "gradients": gradient_consistency(model, dataset, data_processor),
        "resolution": resolution_study(model, data_path, nondim, data_processor),
    }
    table = ROOT / "tables" / f"{MODEL_NAME}_evaluation.csv"
    save_csv(flatten(report), table)
    print(f"Evaluation saved: {table}")

    for res, loader in test_loaders.items():
        tag = f"{MODEL_NAME}_res{res}"
        plot_predictions(model, next(iter(loader)), data_processor, nondim, tag=tag)
        plot_normalization_comparison(
            model,
            CheetahDataset(dataset_path(DATA["data_root"], "test", res), nondim=nondim),
            data_processor,
            tag=tag,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
