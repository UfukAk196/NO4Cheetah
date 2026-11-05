"""
Evaluates the model trained by ``train_sm.py`` with the functions of ``scno.evaluate``.

Loads the best checkpoint the neuralop ``Trainer`` wrote to ``outputs/models/``
(``best_model_state_dict.pt`` + ``best_model_metadata.pkl``, overwritten by every run of
``train_sm.py``, so there are no run IDs) with ``TFNO.from_checkpoint`` and uses the data
settings of ``train_sm.py`` (``DATA_SM`` below).

Compared to ``evaluate_model.py`` the lean dataset (``scno.data.dataset_sm``) has no meta
fields, so there is no per-distribution breakdown and no gradient consistency check. The
resolution study is the evaluation on the test sets at 32 (trained) and 64 (zero-shot).

Writes only to ``outputs/``:
    tables/sm/evaluation.csv      accuracy per test resolution
    figures/sm/beam_<res>_<i>.png phase space of the first test beams
                                  (Cheetah ``ParticleBeam.plot_distribution``)
    figures/sm/phi_<res>_<i>.png  phi GT vs. Pred of these samples, same scale
                                  (``plot_field_comparison``, also via Cheetah)

Call: python exec/evaluate_sm.py
"""

import h5py
import torch
from cheetah.accelerator import SpaceChargeKick
from neuralop.models import TFNO

from scno.config import DEVICE, FIGURES_DIR, MODELS_DIR, SEED, TABLES_DIR
from scno.data.dataset_sm import load_cheetah_sc
from scno.data.generate_sm import _build_beam, _grid, dataset_path
from scno.evaluate import _save_figures, evaluate, plot_field_comparison, predict
from scno.utils.common import pick_device, save_csv

# Must match train_sm.py: the output normalizer is refitted on the same training data.
DATA_SM = {
    "n_train": 1000,
    "batch_size": 64,
    "n_tests": [100, 50],
    "test_resolutions": [32, 64],
    "test_batch_sizes": [32, 32],
    "seed": SEED,
    "ranges": None,
}
SAVE_NAME = "best_model"  # Trainer(save_best=...) writes best_model_*.pt/.pkl
N_PLOTS = 3


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
    figures_dir = FIGURES_DIR / "sm"
    device = pick_device(DEVICE)
    model = TFNO.from_checkpoint(MODELS_DIR, SAVE_NAME, map_location=device).to(device)
    manifest = MODELS_DIR / "manifest.pt"
    if manifest.exists():
        print(f"Checkpoint epoch: {torch.load(manifest).get('epoch')}")

    _, test_loaders, data_processor = load_cheetah_sc(**DATA_SM)
    data_processor = data_processor.to(device)

    report = {
        f"res_{res}": evaluate(model, loader.dataset, data_processor)
        for res, loader in test_loaders.items()
    }
    table = TABLES_DIR / "sm" / "evaluation.csv"
    save_csv(flatten(report), table)
    print(f"Evaluation saved: {table}")

    model.eval()
    for j, (res, loader) in enumerate(test_loaders.items()):
        dataset = loader.dataset
        sck = SpaceChargeKick(effect_length=torch.tensor(0.0), grid_shape=(res, res, res))
        # Test file j was generated with seed SEED + 1 + j and torch.manual_seed(seed + i)
        # per sample (generate_sm), so the beams behind the model input rho are rebuilt
        # exactly; their cell size is not stored in the file.
        with h5py.File(dataset_path("data", "test", res), "r") as f:
            params = [
                {k: f["params"][k][i] for k in f["params"]}
                for i in range(min(N_PLOTS, len(dataset)))
            ]
        for i, p in enumerate(params):
            torch.manual_seed(SEED + 1 + j + i)
            beam = _build_beam(p)
            cell_size = _grid(beam, sck)[1][0]
            # plot_distribution needs an unbatched beam, _build_beam returns batch (1,).
            fig, _ = beam[0].plot_distribution(dimensions=("x", "y", "tau"))
            fig.suptitle(f"Test beam {i} @ {res}")
            _save_figures({f"beam_{res}_{i}": fig}, save_dir=figures_dir)

            item = dataset[i]
            with torch.no_grad():
                pred = predict(model, torch.utils.data.default_collate([item]), data_processor)
            plot_field_comparison(
                {"phi GT": item["y"][0], "phi Pred": pred[0, 0]},
                cell_size,
                title=f"Test sample {i} @ {res}: phi projections",
                name=f"phi_{res}_{i}",
                save_dir=figures_dir,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
