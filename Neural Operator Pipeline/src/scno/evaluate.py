"""
Evaluation -- the numbers that really count in the thesis.

Four questions are answered here:

  1. How well does the model hit the potential?       (relative L2)
  2. How well does it hit the forces?                 (gradient error)
     This is the decisive quantity: the kick is a gradient.
  3. Does it hold across distribution families?       (breakdown by kind)
  4. Does it survive a different resolution?          (zero-shot super-resolution)
     That is the unique selling point of neural operators compared to CNNs -- so
     measure and show it instead of just claiming it.

Call:
    python -m scno.evaluate --run outputs/run_default
    python -m scno.evaluate --run outputs/smoke --data data/cheetah_sc_test_64.h5 --plot
"""

from __future__ import annotations

import argparse
import itertools
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from neuralop.models import FNO

from scno.data.dataset import CheetahDataset, load_cheetah_sc
from scno.data.generate import DISTRIBUTIONS, dataset_path
from scno.models.exact import ExactOperator
from scno.utils.common import load_config, pick_device
from scno.utils.losses import field_error, gradient_error
from scno.utils.normalization import Scaling, decode_prediction, input_channels

ROOT = Path(__file__).resolve().parents[2]


def _device(model: torch.nn.Module) -> torch.device:
    """Device of a model; ``ExactOperator`` has buffers but no parameters."""
    return next(itertools.chain(model.parameters(), model.buffers())).device


def predict(model, batch: dict, data_processor=None) -> torch.Tensor:
    """Model output for ``batch`` (keys ``x``, ``y``) in the units of ``y``.

    The single path through the data processor: in ``eval`` mode ``preprocess``
    normalizes only ``x`` and ``postprocess`` undoes the output normalization. Calling
    ``eval()`` explicitly matters, because the processor is an ``nn.Module`` that starts
    in training mode, in which ``postprocess`` silently does nothing.
    (``DefaultDataProcessor.forward`` is not used: in neuralop 2.0.0 it calls
    ``postprocess`` without ``data_dict``.)
    """
    if data_processor is None:
        return model(batch["x"].to(_device(model)))
    data_processor.eval()
    sample = data_processor.preprocess(dict(batch))
    out, _ = data_processor.postprocess(model(sample["x"]), sample)
    return out


@torch.no_grad()
def evaluate(model, dataset, data_processor=None, batch_size: int = 4) -> dict:
    """Relative L2 errors of potential and force on ``dataset`` (``return_meta=True``),
    in the units the model was trained on (phi_hat or phi [V])."""
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    per_kind = defaultdict(lambda: {"field": [], "grad": []})
    field_errors, grad_errors = [], []

    model.eval()
    for batch in loader:
        pred = predict(model, batch, data_processor)
        target = batch["y"].to(pred.device)
        for i in range(pred.shape[0]):
            fe = float(field_error(pred[i : i + 1], target[i : i + 1]))
            ge = float(gradient_error(pred[i : i + 1], target[i : i + 1]))
            field_errors.append(fe)
            grad_errors.append(ge)
            kind = DISTRIBUTIONS[int(batch["kind"][i])]
            per_kind[kind]["field"].append(fe)
            per_kind[kind]["grad"].append(ge)

    def stats(values):
        t = torch.tensor(values)
        return {
            "mean": float(t.mean()),
            "median": float(t.median()),
            "p95": float(t.quantile(0.95)),
            "max": float(t.max()),
            "n": len(values),
        }

    return {
        "potential_rel_l2": stats(field_errors),
        "force_rel_l2": stats(grad_errors),
        "per_distribution": {
            k: {"potential": stats(v["field"]), "force": stats(v["grad"])}
            for k, v in sorted(per_kind.items())
        },
    }


@torch.no_grad()
def resolution_study(
    model,
    data_path,
    nondim: bool,
    data_processor=None,
    factors=(1, 2),
    batch_size: int = 2,
) -> dict:
    """Trained on one resolution, tested on several.

    factor=2 means half the resolution per axis. An FNO can do this without
    retraining because it works in Fourier space -- a CNN cannot.
    """
    out = {}
    for f in factors:
        ds = CheetahDataset(data_path, downsample=f, nondim=nondim)
        res = evaluate(model, ds, data_processor, batch_size=batch_size)
        out[f"downsample_{f}"] = {
            "grid": tuple(s // f for s in ds.grid_shape),
            "potential_rel_l2_mean": res["potential_rel_l2"]["mean"],
            "force_rel_l2_mean": res["force_rel_l2"]["mean"],
        }
    return out


def gradient_consistency(model, dataset, data_processor=None, num_samples: int = 4) -> dict:
    """Checks whether gradients *through* the model are usable.

    That is exactly what this thesis is about: the surrogate must not only compute
    correctly but also differentiate correctly -- otherwise it is useless for
    gradient-based optimization of beam parameters.

    d(objective)/d(density) is compared once through the network and once through the
    exact solver (``ExactOperator`` in hat units). The cosine similarity of the two
    gradient fields is reported: 1.0 means the same direction, and the direction is
    what an optimizer actually uses. For a raw model (``nondim=False``) the gradient is
    taken w.r.t. rho instead of rho_hat; per sample the two differ only by a constant
    factor, so the cosine stays comparable.
    """
    model.eval()
    exact = ExactOperator().to(_device(model))
    similarities, magnitudes = [], []

    for i in range(min(num_samples, len(dataset))):
        item = dataset[i]
        x = item["x"][None].to(_device(model)).requires_grad_(True)
        # Scalar objective: mean square of the potential -- a stand-in for any quantity
        # one later optimizes for.
        pred = predict(model, {"x": x, "y": item["y"][None]}, data_processor)
        grad_model = torch.autograd.grad(pred.square().mean(), x)[0][:, 0]

        x_ref = input_channels(item["rho_hat"][None], item["normalized_extent"][None])
        x_ref = x_ref.double().to(_device(exact)).requires_grad_(True)
        grad_ref = torch.autograd.grad(exact(x_ref).square().mean(), x_ref)[0][:, 0]

        a = grad_model.flatten().double()
        b = grad_ref.flatten().to(a.device)
        similarities.append(float(torch.nn.functional.cosine_similarity(a, b, dim=0)))
        magnitudes.append(float(a.norm() / (b.norm() + 1e-30)))

    return {
        "cosine_similarity_mean": float(torch.tensor(similarities).mean()),
        "cosine_similarity_min": float(torch.tensor(similarities).min()),
        "magnitude_ratio_mean": float(torch.tensor(magnitudes).mean()),
        "n": len(similarities),
    }


# Central cuts through the grid, in the style of the SciBeam plot:
#   x-y at tau = cen,  x-tau at y = cen,  y-tau at x = cen
CUTS = {
    "xy": ("x-y  (tau = cen)", (0, 1)),
    "xt": ("x-tau (y = cen)", (0, 2)),
    "yt": ("y-tau (x = cen)", (1, 2)),
}
AXIS_NAMES = ["x / idx", "y / idx", "tau / idx"]


def _central_cut(field, cut: str):
    nx, ny, nz = field.shape
    return {
        "xy": field[:, :, nz // 2],
        "xt": field[:, ny // 2, :],
        "yt": field[nx // 2, :, :],
    }[cut]


def _draw_cuts(fig, axes, field, label, cmap, cuts=tuple(CUTS), vlim=None):
    """Draws one central cut (``cuts``) of a 3D field per axis in ``axes``."""
    if torch.is_tensor(field):
        field = field.detach().cpu().numpy()
    vmin, vmax = vlim if vlim is not None else (None, None)
    for ax, cut in zip(axes, cuts):
        title, (ia, ib) = CUTS[cut]
        image = ax.imshow(
            _central_cut(field, cut).T,
            origin="lower",
            aspect="auto",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        fig.colorbar(image, ax=ax)
        ax.set_title(f"{label}   {title}")
        ax.set_xlabel(AXIS_NAMES[ia])
        ax.set_ylabel(AXIS_NAMES[ib])


def _save_figures(figures: dict, save_dir=None, tag=""):
    """Saves ``{file name: figure}`` under ``save_dir`` (default: figures/)."""
    save_dir = Path(save_dir) if save_dir is not None else ROOT / "figures"
    save_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{tag}" if tag else ""
    paths = []
    for name, fig in figures.items():
        path = save_dir / f"{name}{suffix}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        paths.append(str(path))
    print(f"Figures saved: {', '.join(paths)}")


def plot_predictions(model, batch, data_processor, nondim: bool, tag="", save_dir=None):
    """Plots FNO predictions on Cheetah SC data (3D).

    ``batch`` is a raw batch dict from a test loader:
        {"x": (B, 4, nx, ny, nz), "y": (B, 1, nx, ny, nz)}
    The first 3 samples are plotted; the axes are grid indices. The prediction is in
    the units of y (phi_hat for ``nondim=True``, otherwise phi in V). For the raw vs.
    nondimensionalized comparison see ``plot_normalization_comparison``.
    """
    with torch.no_grad():
        out = predict(model, batch, data_processor)
    x, y = batch["x"], batch["y"]

    B = min(3, x.shape[0])  # at most 3 samples
    rho_name, phi_name = ("rho_hat", "phi_hat") if nondim else ("rho [C/m^3]", "phi [V]")
    unit = "nondimensionalized" if nondim else "physical"

    # ----- Figure 1: input, B samples x 3 central cuts -----
    fig1, axs1 = plt.subplots(B, 3, figsize=(11, 3.5 * B), squeeze=False)
    for i in range(B):
        _draw_cuts(fig1, axs1[i], x[i, 0], rho_name, "viridis")
    fig1.suptitle(f"Input {rho_name} ({unit})", y=0.99)
    fig1.tight_layout()

    # ----- Figure 2: ground truth vs. prediction, per sample [GT: 3 | Pred: 3] -----
    fig2, axs2 = plt.subplots(B, 6, figsize=(22, 3.5 * B), squeeze=False)
    for i in range(B):
        _draw_cuts(fig2, axs2[i, :3], y[i, 0], f"{phi_name} GT", "magma")
        _draw_cuts(fig2, axs2[i, 3:], out[i, 0], f"{phi_name} Pred", "magma")
    fig2.suptitle(f"{phi_name} ({unit}): ground truth vs. prediction", y=0.99)
    fig2.tight_layout()

    _save_figures({"rho_hat": fig1, "phi_hat_gt_vs_pred": fig2}, save_dir, tag)
    plt.show()
    return fig1, fig2


def plot_normalization_comparison(
    model, dataset, data_processor=None, indices=(0, 1, 2), cut="xy", tag="", save_dir=None
):
    """Raw vs. nondimensionalized fields for input and output.

    ``dataset`` is a ``CheetahDataset`` with ``return_meta=True`` (the training loaders
    deliberately have it off, because the neuralop ``Trainer`` passes every key to the
    model). Whether the model was trained on rho_hat -> phi_hat or on rho -> phi comes
    from ``dataset.nondim``.

    Figure 1 (input), per sample: rho [C/m^3] (3 cuts) | rho_hat (3 cuts)
    Figure 2 (output), per sample, cut ``cut`` ("xy", "xt", "yt"):
        phi GT [V] | phi Pred [V] | phi_hat GT | phi_hat Pred
    GT and Pred share the colour scale (common range in the cut).
    """
    batch = torch.utils.data.default_collate([dataset[i] for i in indices])
    scaling = Scaling(
        total_charge=batch["total_charge"],
        reference_length=batch["reference_length"],
        normalized_extent=batch["normalized_extent"],
    )
    with torch.no_grad():
        out = predict(model, batch, data_processor)
    phi_hat_pred, phi_pred = decode_prediction(
        out[:, 0].cpu().double(), scaling, nondim=dataset.nondim
    )

    B = len(indices)
    trained_on = "rho_hat -> phi_hat" if dataset.nondim else "rho -> phi"

    # ----- Figure 1: input raw vs. nondimensionalized -----
    fig1, axs1 = plt.subplots(B, 6, figsize=(22, 3.5 * B), squeeze=False)
    for i in range(B):
        _draw_cuts(fig1, axs1[i, :3], batch["rho"][i], "rho [C/m^3]", "viridis")
        _draw_cuts(fig1, axs1[i, 3:], batch["rho_hat"][i], "rho_hat", "viridis")
    fig1.suptitle("Input: physical (left) vs. nondimensionalized (right)", y=0.99)
    fig1.tight_layout()

    # ----- Figure 2: output raw vs. nondimensionalized, GT vs. Pred -----
    fig2, axs2 = plt.subplots(B, 4, figsize=(16, 3.5 * B), squeeze=False)
    for i in range(B):
        pairs = [
            ("phi", "[V]", batch["phi"][i], phi_pred[i]),
            ("phi_hat", "", batch["phi_hat"][i], phi_hat_pred[i]),
        ]
        for j, (name, unit, gt, pred) in enumerate(pairs):
            # Common scale from GT and Pred in the shown cut: otherwise an overshooting
            # prediction is clipped and the cut uses only a fraction of the scale.
            gt_cut, pred_cut = (_central_cut(f.detach().cpu(), cut) for f in (gt, pred))
            vlim = (
                float(min(gt_cut.min(), pred_cut.min())),
                float(max(gt_cut.max(), pred_cut.max())),
            )
            for k, (kind, field) in enumerate((("GT", gt), ("Pred", pred))):
                label = f"{name} {kind} {unit}".strip()
                _draw_cuts(
                    fig2, [axs2[i, 2 * j + k]], field, label, "magma", cuts=(cut,), vlim=vlim
                )
    fig2.suptitle(
        f"Output: physical vs. nondimensionalized (model: {trained_on})", y=0.99
    )
    fig2.tight_layout()

    _save_figures({"norm_input": fig1, "norm_output": fig2}, save_dir, tag)
    plt.show()
    return fig1, fig2


def plot_losses(history, save_dir=None, tag=""):
    """Plots training and evaluation loss curves from ``HistoryTrainer.history``.

    Three panels:
      1. Training loss (avg_loss, train_err) over the epochs (log y).
      2. Evaluation loss per test resolution (h1, l2) over the epochs (log y).
      3. Learning rate over the epochs (log y).

    The evaluation curves only have points every ``eval_interval`` epochs.
    """
    train_hist = history.get("train", {})
    eval_hist = history.get("eval", {})

    fig, axs = plt.subplots(1, 3, figsize=(18, 5))

    # --- Panel 1: training loss ---
    ax = axs[0]
    if train_hist.get("epoch"):
        ax.plot(
            train_hist["epoch"], train_hist["avg_loss"], label="avg_loss", color="tab:blue"
        )
        ax.plot(
            train_hist["epoch"],
            train_hist["train_err"],
            label="train_err",
            color="tab:orange",
            alpha=0.7,
        )
        ax.set_yscale("log")
        ax.legend()
    else:
        ax.text(0.5, 0.5, "no training data", ha="center", va="center")
    ax.set_title("Training loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (log)")
    ax.grid(True, which="both", alpha=0.3)

    # --- Panel 2: evaluation loss per resolution ---
    ax = axs[1]
    eval_epochs = eval_hist.get("epoch", [])
    if eval_epochs:
        # Metric names have the form "<res>_<loss>", e.g. "16_h1"
        metric_names = [k for k in eval_hist if k != "epoch"]
        resolutions = sorted({int(k.split("_")[0]) for k in metric_names})
        cmap = plt.get_cmap("tab10")
        for i, res in enumerate(resolutions):
            for loss_name, style in (("h1", "-"), ("l2", "--")):
                key = f"{res}_{loss_name}"
                if key in eval_hist:
                    ax.plot(
                        eval_epochs,
                        eval_hist[key],
                        style,
                        color=cmap(i % 10),
                        label=f"res {res} {loss_name}",
                    )
        ax.set_yscale("log")
        ax.legend(fontsize=8, ncol=2)
    else:
        ax.text(0.5, 0.5, "no evaluation data", ha="center", va="center")
    ax.set_title("Evaluation loss per resolution")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (log)")
    ax.grid(True, which="both", alpha=0.3)

    # --- Panel 3: learning rate ---
    ax = axs[2]
    if train_hist.get("epoch") and train_hist.get("lr"):
        ax.plot(train_hist["epoch"], train_hist["lr"], color="tab:green")
        ax.set_yscale("log")
    else:
        ax.text(0.5, 0.5, "no learning rate data", ha="center", va="center")
    ax.set_title("Learning rate")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("lr (log)")
    ax.grid(True, which="both", alpha=0.3)

    fig.suptitle("Training history", y=1.0)
    fig.tight_layout()

    _save_figures({"loss_curves": fig}, save_dir, tag)
    plt.show()
    return fig


def load_run(run_dir: str | Path, device: torch.device | None = None):
    """Model and config of a training run written by ``scno.train``.

    Prefers the best checkpoint (``best_model_*``), falls back to the final one
    (``model_*``). Returns ``(model, cfg)``.
    """
    run_dir = Path(run_dir)
    cfg = load_config(run_dir / "config.yaml")
    name = "best_model" if (run_dir / "best_model_metadata.pkl").exists() else "model"
    model = FNO.from_checkpoint(run_dir, name, map_location=device)
    return model.to(device) if device is not None else model, cfg


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="outputs/run_default", help="output_dir of scno.train")
    p.add_argument("--data", default=None, help="test file (default: train resolution)")
    p.add_argument("--out", default=None, help="report (default: <run>/evaluation.json)")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--skip-resolution", action="store_true")
    p.add_argument("--plot", action="store_true", help="plot predictions per resolution")
    args = p.parse_args()

    device = pick_device("auto")
    model, cfg = load_run(args.run, device)
    data_cfg = cfg["data"]
    nondim = data_cfg["nondim"]
    # The data processor is refitted on the training data, exactly as in training.
    _, test_loaders, data_processor = load_cheetah_sc(**{**data_cfg, "generate": False})
    data_processor = data_processor.to(device)

    data_path = args.data or dataset_path(
        data_cfg["data_root"], "test", data_cfg["train_resolution"]
    )
    dataset = CheetahDataset(data_path, nondim=nondim)
    report = {"accuracy": evaluate(model, dataset, data_processor, args.batch_size)}
    report["gradients"] = gradient_consistency(model, dataset, data_processor)
    if not args.skip_resolution:
        report["resolution"] = resolution_study(model, data_path, nondim, data_processor)

    out = Path(args.out) if args.out else Path(args.run) / "evaluation.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))

    if args.plot:
        for res, loader in test_loaders.items():
            tag = f"{Path(args.run).name}_res{res}"
            plot_predictions(model, next(iter(loader)), data_processor, nondim, tag=tag)
            plot_normalization_comparison(
                model,
                CheetahDataset(dataset_path(data_cfg["data_root"], "test", res), nondim=nondim),
                data_processor,
                tag=tag,
            )


if __name__ == "__main__":
    main()
