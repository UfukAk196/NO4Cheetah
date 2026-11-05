"""
Evaluates the model trained by ``train_model.py`` with the functions of ``scno.evaluate``:
trained resolution (32) vs. zero-shot super-resolution (64).

Loads ``outputs/models/final_model_*`` (``TFNO.from_checkpoint``) and the data processor
fitted in training (``final_model_data_processor.pt``), so the normalization is exactly
the one of the training run. ``train_model.py`` calls ``main(run)`` itself at the end.

Writes only to ``outputs/`` (``RUN`` = the given run, default: the latest one):
    tables/<RUN>/evaluation.csv      rel. L2 of phi and force, absolute phi [V]; mean and
                                     max over the test samples, per test resolution
    tables/<RUN>/samples.csv         the same numbers per test sample
    figures/<RUN>/beam_<res>_<i>.png phase space of the first test beams
                                     (Cheetah ``ParticleBeam.plot_distribution``)
    figures/<RUN>/phi_<res>_<i>.png  phi GT vs. Pred of these samples, same scale
                                     (``plot_field_comparison``, also via Cheetah)

Call: python exec/evaluate_model.py [RUN]   e.g. RUN = 2026-11-05_02
"""

import sys

import h5py
import pandas as pd
import torch
from cheetah.accelerator import SpaceChargeKick
from neuralop.models import TFNO

from scno.config import DATA_DIR, FIGURES_DIR, MODELS_DIR, TABLES_DIR ,DATA, SEED
from scno.data.dataset import load_cheetah_sc
from scno.data.generate import _build_beam, _grid, dataset_path
from scno.evaluate import _save_figures, evaluate, plot_field_comparison, predict
from scno.utils.common import existing_runs

N_PLOTS = 3

def main(run: str | None = None) -> int:
    if run is None:
        runs = existing_runs(FIGURES_DIR, TABLES_DIR)
        if not runs:
            raise FileNotFoundError("No run found, run train_model.py first.")
        run = runs[-1]
    print(f"Run: {run}")
    figures_dir, tables_dir = FIGURES_DIR / run, TABLES_DIR / run
    tables_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TFNO.from_checkpoint(MODELS_DIR, "final_model", map_location=device).to(device)
    data_processor = torch.load(
        MODELS_DIR / "final_model_data_processor.pt", map_location=device, weights_only=False
    )
    manifest = MODELS_DIR / "checkpoints" / "manifest.pt"
    if manifest.exists():
        print(f"Best checkpoint at epoch: {torch.load(manifest).get('epoch')}")

    _, test_loaders, _ = load_cheetah_sc(**DATA, encode_output=False)

    report, samples = {}, []
    for res, loader in test_loaders.items():
        rows = []
        report[f"res_{res}"] = evaluate(model, loader.dataset, data_processor, samples=rows)
        samples += [{"res": res, **row} for row in rows]
    # Rows "res_<res>.<quantity>.<mean|max>", value
    pd.json_normalize(report).T.to_csv(
        tables_dir / "evaluation.csv", header=["value"], index_label="quantity"
    )
    pd.DataFrame(samples).to_csv(tables_dir / "samples.csv", index=False)
    print(f"Evaluation saved: {tables_dir / 'evaluation.csv'}, {tables_dir / 'samples.csv'}")
    for row in samples[:N_PLOTS]:
        print(
            f"  res {row['res']} sample {row['sample']}: "
            f"max|phi| GT {row['gt_max_abs']:.4g} V, Pred {row['pred_max_abs']:.4g} V | "
            f"mean GT {row['gt_mean']:.4g} V, Pred {row['pred_mean']:.4g} V | "
            f"max|err| {row['max_abs_error']:.4g} V"
        )

    model.eval()
    for j, (res, loader) in enumerate(test_loaders.items()):
        dataset = loader.dataset
        sck = SpaceChargeKick(effect_length=torch.tensor(0.0), grid_shape=(res, res, res))
        # Test file j was generated with seed SEED + 1 + j and torch.manual_seed(seed + i)
        # per sample (generate.py), so the beams behind the model input rho are rebuilt
        # exactly; their cell size is not stored in the file.
        with h5py.File(dataset_path(DATA_DIR, "test", res), "r") as f:
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
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else None))
