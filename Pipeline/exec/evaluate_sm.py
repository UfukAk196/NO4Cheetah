"""
Evaluates the model trained by ``train_sm.py`` (the small-model pipeline: raw
``nondim=False`` rho -> phi, 1 channel in / 1 channel out, ``dataset_sm`` .pt data).

Counterpart to ``evaluate_model.py``, which only works with the .h5 pipeline of
``train_model.py``: that one rebuilds the model from neuralop's ``_metadata`` in the
.pth and reads rich metadata (kind, cell_size, gamma, ...) from the HDF5 dataset.
``train_sm.py`` instead saves neuralop ``Trainer`` artifacts (``save_best``) -- a raw
state dict WITHOUT ``_metadata`` -- and its ``dataset_sm`` data holds only ``x``/``y``.
Hence here the architecture is hard-coded (as in ``train_sm.py``), the model is loaded
from ``best_model_state_dict.pt`` and the geometry is recomputed from the stored beam
parameters.

Loads ``outputs/models/best_model_state_dict.pt`` (override with the path argument).

Writes only to ``outputs/``:
    tables/eval_sm/evaluation.csv   accuracy + resolution study
    figures/eval_sm/beam_<i>.png    phase space of the first test beams
                                    (Cheetah ``ParticleBeam.plot_distribution``)
    figures/eval_sm/phi_<i>.png     phi GT vs. Pred of these samples, same scale
                                    (``plot_field_comparison``, also via Cheetah)

Call: python exec/evaluate_sm.py [MODEL.pt]
"""

import sys
from pathlib import Path

import torch
from cheetah.accelerator import SpaceChargeKick
from neuralop.models import TFNO

from scno.config import DATA, FIGURES_DIR, MODELS_DIR, SEED, TABLES_DIR
from scno.data.dataset_sm import dataset_path, load_cheetah_sc, missing_files
from scno.data.fields import _build_beam
from scno.evaluate import _save_figures, plot_field_comparison, predict
from scno.utils.common import pick_device, save_csv
from scno.utils.losses import field_error, gradient_error

# Architecture of ``train_sm.py`` -- the .pt state dict carries no ``_metadata``, so the
# model has to be rebuilt with exactly these kwargs.
TFNO_KWARGS = {
    "n_modes": (5, 5, 5),
    "in_channels": 1,
    "out_channels": 1,
    "hidden_channels": 32,
    "n_layers": 3,
    "domain_padding": 0.125,
}

# Data settings of ``train_sm.py`` (raw rho -> phi). ``generate=False``: evaluation
# never generates data; the test files must already exist.
DATA_SM = {
    "n_train": 1000,
    "batch_size": 64,
    "n_tests": [100, 50],
    "test_resolutions": [32, 64],
    "test_batch_sizes": [32, 32],
    "seed": SEED,
    "ranges": None,
    "data_root": DATA["data_root"],
    "train_resolution": 32,
}

DEFAULT_MODEL = MODELS_DIR / "best_model_state_dict.pt"
OUT_NAME = "eval_sm"


def load_model(path: Path, device: torch.device) -> TFNO:
    """Rebuilds the TFNO from the hard-coded kwargs and loads the raw state dict.

    Unlike ``evaluate_model.load_model`` there is no ``_metadata`` to read: neuralop's
    ``Trainer`` saves a plain ``state_dict``.
    """
    state_dict = torch.load(path, map_location=device, weights_only=False)
    model = TFNO(**TFNO_KWARGS)
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


def stats(values) -> dict:
    t = torch.tensor(values)
    return {
        "mean": float(t.mean()),
        "median": float(t.median()),
        "p95": float(t.quantile(0.95)),
        "max": float(t.max()),
        "n": len(values),
    }


@torch.no_grad()
def evaluate(model, loader, data_processor) -> dict:
    """Relative L2 errors of potential and force on a ``dataset_sm`` test loader.

    Reduced version of ``scno.evaluate.evaluate``: the .pt samples carry no ``kind``
    (and there is only one distribution, ``gaussian`` anyway), so the per-distribution
    breakdown is dropped. Kept otherwise identical (same per-sample metrics, same
    data-processor path through ``predict``).
    """
    field_errors, grad_errors = [], []

    model.eval()
    for batch in loader:
        pred = predict(model, batch, data_processor)
        target = batch["y"].to(pred.device)
        for i in range(pred.shape[0]):
            field_errors.append(float(field_error(pred[i : i + 1], target[i : i + 1])))
            grad_errors.append(float(gradient_error(pred[i : i + 1], target[i : i + 1])))

    return {
        "potential_rel_l2": stats(field_errors),
        "force_rel_l2": stats(grad_errors),
    }


def resolution_study(model, test_loaders, data_processor) -> dict:
    """Trained on one resolution, tested on all test resolutions (zero-shot SR).

    Reimplemented for the .pt pipeline (``scno.evaluate.resolution_study`` needs the
    HDF5 ``CheetahDataset`` with ``downsample``): here every resolution already has its
    own test file/loader, so simply evaluate on each.
    """
    out = {}
    for res, loader in sorted(test_loaders.items()):
        result = evaluate(model, loader, data_processor)
        item = next(iter(loader))["x"]
        out[f"res_{res}"] = {
            "grid": tuple(item.shape[-3:]),
            "potential_rel_l2_mean": result["potential_rel_l2"]["mean"],
            "force_rel_l2_mean": result["force_rel_l2"]["mean"],
        }
    return out


def _cell_size(beam, res: int) -> torch.Tensor:
    """Cell size (3,) [m] exactly as ``generate_sm._simulate`` derives it.

    ``SpaceChargeKick`` grid extents default to Cheetah's 3 sigma per axis, so
    ``grid_dimensions = grid_extent_i * sigma_i`` and ``cell_size = 2 * grid_dimensions
    / grid_shape``.
    """
    sck = SpaceChargeKick(effect_length=torch.tensor(0.0), grid_shape=(res, res, res))
    grid_dimensions = torch.stack(
        [
            sck.grid_extent_x * beam.sigma_x,
            sck.grid_extent_y * beam.sigma_y,
            sck.grid_extent_tau * beam.sigma_tau,
        ],
        dim=-1,
    )
    return (2 * grid_dimensions / torch.tensor([res, res, res])).double()


def main(model_file: Path | None = None) -> int:
    device = pick_device()
    path = DEFAULT_MODEL if model_file is None else Path(model_file)
    if not path.exists():
        raise FileNotFoundError(
            f"No trained model found at {path}, run train_sm.py first."
        )
    print(f"Model: {path}")

    model = load_model(path, device)

    # Evaluation must never (re)generate data: check the required .pt files first and
    # fail loudly instead of silently triggering generation (dataset_sm's loader has no
    # ``generate`` flag and regenerates missing files on its own).
    missing = missing_files(
        DATA_SM["data_root"], DATA_SM["train_resolution"], DATA_SM["test_resolutions"]
    )
    if missing:
        raise FileNotFoundError(
            f"Missing data files {missing} in {DATA_SM['data_root']}; "
            "run train_sm.py first (it generates them)."
        )

    # The data processor is refitted on the training data, exactly as in train_sm.py
    # (input normalizer over the 1-channel raw input; output normalizer for the raw
    # ablation).
    _, test_loaders, data_processor = load_cheetah_sc(**DATA_SM)
    data_processor = data_processor.to(device)

    figures_dir = FIGURES_DIR / OUT_NAME
    table_dir = TABLES_DIR / OUT_NAME

    report = {
        "accuracy": evaluate(model, test_loaders[DATA_SM["test_resolutions"][0]], data_processor),
        "resolution": resolution_study(model, test_loaders, data_processor),
    }
    table = table_dir / "evaluation.csv"
    save_csv(flatten(report), table)
    print(f"Evaluation saved: {table}")

    # Beams behind the first test samples, rebuilt from the stored parameters. The
    # generator seeds each sample with ``seed + i`` where ``i`` is the global index in
    # the file; the test file j uses seed ``SEED + 1 + j`` (see generate_sm). The 32
    # resolution is test_resolutions[0], i.e. j = 0.
    res = DATA_SM["test_resolutions"][0]
    data_path = dataset_path(DATA_SM["data_root"], "test", res)
    data = torch.load(data_path, map_location="cpu")
    params_all = data["params"]
    n_fig = min(3, data["x"].shape[0])

    figures = {}
    beams = []
    for i in range(n_fig):
        params = {k: float(params_all[k][i]) for k in params_all}
        params["seed"] = SEED + 1 + i  # test file j=0
        # plot_distribution needs an unbatched beam, _build_beam returns batch (1,).
        beam = _build_beam(params)[0]
        beams.append(beam)
        fig, _ = beam.plot_distribution(dimensions=("x", "y", "tau"))
        fig.suptitle(f"Test beam {i}")
        figures[f"beam_{i}"] = fig
    _save_figures(figures, save_dir=figures_dir)

    # Their potential: ground truth vs. prediction, both raw phi in V (nondim=False).
    model.eval()
    loader = test_loaders[res]
    for i in range(n_fig):
        item = _sample(loader, i)
        with torch.no_grad():
            pred = predict(model, {"x": item["x"][None], "y": item["y"][None]}, data_processor)
        plot_field_comparison(
            {"phi GT": item["y"][0], "phi Pred": pred[0, 0]},
            _cell_size(beams[i], res),
            title=f"Test sample {i}: phi projections",
            name=f"phi_{i}",
            save_dir=figures_dir,
        )
    return 0


def _sample(loader, index: int) -> dict:
    """``index``-th sample of a ``dataset_sm`` loader (batch dim kept: ``x``/``y`` as in
    ``predict``)."""
    dataset = loader.dataset
    item = dataset[index]
    return {"x": item["x"], "y": item["y"]}


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else None))
