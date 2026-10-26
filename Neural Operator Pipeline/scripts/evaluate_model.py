"""
Evaluates the model trained by ``train_model.py`` with the functions of ``scno.evaluate``.

Evaluates the run ``<RUN>`` given as argument (default: the latest run, see
``scno.utils.common.existing_runs``). Loads ``models/<MODEL_NAME>_<RUN>.pth`` with
neuralop's ``load_state_dict`` (the model is rebuilt from the ``_metadata`` neuralop
stores in the state dict) and uses the data settings in ``scno.config``.

Writes only:
    tables/<RUN>/evaluation.csv   accuracy, gradient fidelity, resolution study
    figures/<RUN>/beam_<i>.png    phase space of the first test beams
                                  (Cheetah ``ParticleBeam.plot_distribution``)
    figures/<RUN>/phi_<i>.png     phi GT vs. Pred of these samples, same scale
                                  (``plot_field_comparison``, also via Cheetah)

Call: python scripts/evaluate_model.py [RUN]   e.g. RUN = 2026-10-26_01
"""

import sys
from pathlib import Path

import h5py
import torch
from neuralop.models import FNO

from scno.config import DATA, DEVICE, FIGURES_DIR, RUN_DIRS, TABLES_DIR, model_path
from scno.data.dataset import CheetahDataset, load_cheetah_sc
from scno.data.fields import _build_beam
from scno.data.generate import dataset_path
from scno.evaluate import (
    _save_figures,
    evaluate,
    gradient_consistency,
    plot_field_comparison,
    predict,
    resolution_study,
)
from scno.utils.common import existing_runs, pick_device, save_csv

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


def main(run: str | None = None) -> int:
    if run is None:
        runs = [r for r in existing_runs(*RUN_DIRS) if model_path(r).exists()]
        if not runs:
            raise FileNotFoundError("No trained model found, run train_model.py first.")
        run = runs[-1]
    print(f"Run: {run}")
    figures_dir = FIGURES_DIR / run

    device = pick_device(DEVICE)
    model = load_model(model_path(run), device)
    nondim = DATA["nondim"]

    # The data processor is refitted on the training data, exactly as in training (it
    # only holds normalizers for nondim=False).
    _, _, data_processor = load_cheetah_sc(**{**DATA, "generate": False})
    data_processor = data_processor.to(device)

    data_path = dataset_path(DATA["data_root"], "test", DATA["train_resolution"])
    dataset = CheetahDataset(data_path, nondim=nondim)
    report = {
        "accuracy": evaluate(model, dataset, data_processor),
        "gradients": gradient_consistency(model, dataset, data_processor),
        "resolution": resolution_study(model, data_path, nondim, data_processor),
    }
    table = TABLES_DIR / run / "evaluation.csv"
    save_csv(flatten(report), table)
    print(f"Evaluation saved: {table}")

    # The beams behind the first test samples, rebuilt from the stored parameters and
    # seed -- the same particles whose deposition is the model input rho.
    figures = {}
    with h5py.File(data_path, "r") as f:
        for i in range(min(3, len(dataset))):
            params = {k: f["params"][k][i] for k in f["params"]}
            params["seed"] = f["seed"][i]
            # plot_distribution needs an unbatched beam, _build_beam returns batch (1,).
            fig, _ = _build_beam(params)[0].plot_distribution(dimensions=("x", "y", "tau"))
            fig.suptitle(f"Test beam {i}")
            figures[f"beam_{i}"] = fig
    _save_figures(figures, save_dir=figures_dir)

    # Their potential: ground truth vs. prediction, in the units the model was trained on
    # (phi_hat and phi differ per sample only by a constant, i.e. look the same here).
    phi_name = "phi_hat" if nondim else "phi"
    model.eval()
    for i in range(min(3, len(dataset))):
        item = dataset[i]
        with torch.no_grad():
            pred = predict(model, torch.utils.data.default_collate([item]), data_processor)
        plot_field_comparison(
            {f"{phi_name} GT": item["y"][0], f"{phi_name} Pred": pred[0, 0]},
            item["cell_size"],
            title=f"Test sample {i}: {phi_name} projections",
            name=f"phi_{i}",
            save_dir=figures_dir,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else None))
