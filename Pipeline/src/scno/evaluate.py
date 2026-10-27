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

The evaluation run itself (loading the model, writing the table) is
``scripts/evaluate_model.py``.
"""

from __future__ import annotations

import itertools
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from cheetah.particles import ParticleBeam
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from scno.config import FIGURES_DIR
from scno.data.dataset import CheetahDataset
from scno.data.generate import DISTRIBUTIONS
from scno.models.exact import ExactOperator
from scno.utils.losses import field_error, gradient_error
from scno.utils.normalization import input_channels


def _device(model: torch.nn.Module) -> torch.device:
    """Device of a model; CPU for one without tensors (``ExactOperator``)."""
    tensor = next(itertools.chain(model.parameters(), model.buffers()), None)
    return tensor.device if tensor is not None else torch.device("cpu")


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


def _save_figures(figures: dict, save_dir=None, tag=""):
    """Saves ``{file name: figure}`` under ``save_dir`` (default: figures/)."""
    save_dir = Path(save_dir) if save_dir is not None else FIGURES_DIR
    save_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{tag}" if tag else ""
    paths = []
    for name, fig in figures.items():
        path = save_dir / f"{name}{suffix}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        paths.append(str(path))
    print(f"Figures saved: {', '.join(paths)}")


def field_to_beam(field, cell_size, k: float) -> ParticleBeam | None:
    """Grid field -> ``ParticleBeam`` whose histogram is the field, for Cheetah's plots.

    ``field`` (nx, ny, nz) must be >= 0 (one sign part of a field), ``cell_size`` (3,)
    in m. Each cell gets ``round(k * field)`` particles, all exactly at the cell centre
    (x, y, tau; momenta 0). Histogrammed with one bin per cell (``bins = n``,
    ``bin_ranges = +-n * cell_size / 2``) this reproduces the projected field up to the
    rounding. Returns ``None`` if no particle remains.
    """
    counts = torch.round(k * field).long()
    if counts.sum() == 0:
        return None
    shape = torch.tensor(field.shape, dtype=torch.float64)
    cells = counts.nonzero()
    centres = (cells + 0.5) * cell_size - shape * cell_size / 2
    centres = centres.repeat_interleave(counts[tuple(cells.T)], dim=0)
    particles = torch.zeros(len(centres), 7)
    particles[:, [0, 2, 4]] = centres.float()
    particles[:, 6] = 1.0
    return ParticleBeam(particles, energy=torch.tensor(1e9))


def plot_field_comparison(
    fields: dict, cell_size, title="", name="fields", tag="", save_dir=None,
    max_particles: int = 2_000_000,
):
    """Grid fields (e.g. ``{"phi GT": gt, "phi Pred": pred}``) side by side through
    Cheetah's ``ParticleBeam.plot_distribution``, one 3x3 block (x, y, tau) per field.

    ``plot_distribution`` only shows particle counts, each panel scaled to its own
    maximum. To compare the fields, every field is split into its positive and negative
    part, each part becomes a beam (``field_to_beam``, same particle density ``k`` for
    all fields) and the drawn curves/meshes are rescaled afterwards: every panel is in
    units of the largest |projection| of the *first* field in that panel, negative parts
    count negative. So the first field peaks at +-1 and the others are directly
    comparable to it (shared colour scale and y-range).

    The panels are projections (sums along the hidden axis), not central cuts. Where
    both signs project onto the same pixel, the negative part (drawn last) covers the
    positive one. Needs a cubic grid, since ``plot_distribution`` takes one ``bins``.
    """
    fields = {k: f.detach().cpu().double() for k, f in fields.items()}
    ref_name, ref = next(iter(fields.items()))
    n = ref.shape[0]
    assert ref.shape == (n, n, n), f"cubic grid needed, got {tuple(ref.shape)}"
    cell_size = torch.as_tensor(cell_size, dtype=torch.float64).cpu()
    bin_ranges = [(-h, h) for h in (n * cell_size / 2).tolist()]
    # One particle density for all fields, so the counts share units; capped so that a
    # large grid (or a strongly overshooting prediction) stays in memory.
    k = min(
        500 / float(ref.abs().max()),
        max_particles / max(float(f.abs().sum()) for f in fields.values()),
    )

    def summed(field, *keep):
        """Projection of ``field`` onto the axes ``keep``."""
        return field.sum(tuple(a for a in range(3) if a not in keep))

    pairs = list(itertools.combinations(range(3), 2))
    ref_1d = [float(summed(ref, i).abs().max()) for i in range(3)]
    ref_2d = {(i, j): float(summed(ref, i, j).abs().max()) for i, j in pairs}

    cmap = "RdBu_r"
    fig, axs = plt.subplots(
        3, 3 * len(fields), figsize=(7 * len(fields) + 1, 6), gridspec_kw={"wspace": 0.45}
    )
    meshes, lines = [], []
    for b, (label, field) in enumerate(fields.items()):
        block = axs[:, 3 * b : 3 * b + 3]
        for sign, part in ((1, field.clamp(min=0)), (-1, (-field).clamp(min=0))):
            beam = field_to_beam(part, cell_size, k)
            if beam is None:
                continue
            beam.plot_distribution(
                ("x", "y", "tau"),
                bins=n,
                bin_ranges=bin_ranges,
                plot_2d_kws={"pcolormesh_kws": {"cmap": cmap}},
                axs=block,
            )
            # Undo the per-panel normalization of Cheetah (division by the maximum).
            counts = torch.round(k * part)
            for i in range(3):
                line = block[i, i].get_lines()[-1]
                scale = sign * float(summed(counts, i).max()) / k / ref_1d[i]
                line.set_ydata(line.get_ydata() * scale)
                lines.append(line)
            for i, j in pairs:
                mesh = block[j, i].collections[-1]
                scale = sign * float(summed(counts, i, j).max()) / k / ref_2d[i, j]
                mesh.set_array(mesh.get_array() * scale)
                meshes.append(mesh)
        block[0, 0].set_title(label, loc="left")

    # Shared scales: symmetric colour range, common y-range of the 1D curves.
    values = [np.ma.masked_invalid(m.get_array()) for m in meshes]
    vmax = max([1.0] + [float(np.abs(v).max()) for v in values if v.count()])
    for mesh in meshes:
        mesh.set_clim(-vmax, vmax)
    ys = np.concatenate([line.get_ydata() for line in lines])
    lo, hi = min(0.0, float(ys.min())), max(0.0, float(ys.max()))
    for b in range(len(fields)):
        for i in range(3):
            ax = axs[i, 3 * b + i]
            ax.set_ylim(lo - 0.05 * (hi - lo), hi + 0.05 * (hi - lo))
            ax.axhline(0.0, color="grey", lw=0.5)

    fig.colorbar(
        ScalarMappable(Normalize(-vmax, vmax), cmap),
        ax=axs.ravel().tolist(),
        shrink=0.8,
        label=f"projection / max |projection of {ref_name}|",
    )
    fig.suptitle(title)
    _save_figures({name: fig}, save_dir, tag)
    return fig


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

