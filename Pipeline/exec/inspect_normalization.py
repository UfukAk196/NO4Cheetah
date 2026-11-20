"""
Shows the data before and after normalization and the model's in- and outputs in
training and evaluation mode, to check that the ``DefaultDataProcessor`` normalizes
what it should.

What neuralop's ``DefaultDataProcessor`` does (2.0.0):
    preprocess   x -> in_normalizer(x) in both modes;
                 y -> out_normalizer(y) only in training mode (loss on normalized phi)
    postprocess  out -> out_normalizer.inverse(out) only in eval mode (loss on phi [V])

So after a correct normalization the model input channels and, in training, the target
and the model output have mean ~0 and std ~1 (exactly so only over the full training
set; this batch only gives an estimate), while in eval mode output and target are in V.

Default: the model and data processor saved by ``train_model.py`` (what the last run
really used). ``--fresh N``: instead fit a processor with the current ``DATA`` settings
on N training samples, without a model, to check a configuration before training it.

Prints a table (mean, std, min, max per channel) and writes to
``figures/<RUN>/`` (latest run) or ``figures/normalization/`` (``--fresh``):
    normalization_hist.png   histograms per channel and stage
    normalization_cuts.png   central cuts of sample 0 per stage

Call: python exec/inspect_normalization.py [--fresh N]
"""

import argparse

import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.data import default_collate

from scno.config import DATA, DATA_DIR, FIGURES_DIR, MODELS_DIR, TABLES_DIR
from scno.data.dataset import CheetahDataset
from scno.evaluate import _save_figures, plot_central_cut
from scno.utils.common import existing_runs

N_SHOW = 64  # samples of the training set that are inspected

parser = argparse.ArgumentParser()
parser.add_argument("--fresh", type=int, metavar="N", help="fit a new processor on N samples")
args = parser.parse_args()

input_params = DATA["input_params"]
model = None
if not args.fresh:
    from neuralop.models import TFNO

    model = TFNO.from_checkpoint(MODELS_DIR, "final_model", map_location="cpu")
    if model.in_channels == 1:
        input_params = ()  # a model from before the beam-parameter channels: rho only
    elif model.in_channels != 1 + len(input_params):
        raise ValueError(f"model has {model.in_channels} input channels, DATA['input_params'] "
                         f"{input_params} gives {1 + len(input_params)}")
x_names = ["rho", *input_params]

# The fitted normalizers depend on n_train; without --fresh nothing is fitted here.
dataset = CheetahDataset(
    DATA_DIR,
    n_train=max(args.fresh or 0, N_SHOW),
    n_tests=DATA["n_tests"],
    train_resolution=DATA["train_resolution"],
    test_resolutions=DATA["test_resolutions"],
    input_params=input_params,
    encode_input=bool(args.fresh),
    encode_output=bool(args.fresh),
    seed=DATA["seed"],
)
batch = default_collate([dataset.train_db[i] for i in range(N_SHOW)])

if args.fresh:
    data_processor = dataset.data_processor
    save_dir = FIGURES_DIR / "normalization"
else:
    data_processor = torch.load(
        MODELS_DIR / "final_model_data_processor.pt", map_location="cpu", weights_only=False
    )
    save_dir = FIGURES_DIR / existing_runs(FIGURES_DIR, TABLES_DIR)[-1]
# .to sets the processor's own ``device`` attribute (still "cuda" after GPU training)
data_processor = data_processor.to("cpu")

for kind in ("in", "out"):
    norm = getattr(data_processor, f"{kind}_normalizer")
    print(f"{kind}_normalizer: " + ("None (no normalization!)" if norm is None else
          f"mean {norm.mean.flatten()[:len(x_names)].tolist()}, "
          f"std {norm.std.flatten()[:len(x_names)].tolist()}"
          + (" (pixel-wise: first grid points only)" if norm.mean[0, 0].numel() > 1 else "")))

# Every stage as {label: (channel names, (B, C, nx, ny, nz) tensor)}.
stages = {"x raw": (x_names, batch["x"]), "y raw [V]": (["phi"], batch["y"])}
with torch.no_grad():
    data_processor.train()
    sample = data_processor.preprocess(dict(batch))
    stages["x model input"] = (x_names, sample["x"])
    stages["y train target"] = (["phi"], sample["y"])
    if model is not None:
        out, _ = data_processor.postprocess(model(sample["x"]), sample)  # no-op in train
        stages["out train"] = (["phi"], out)

    data_processor.eval()
    sample = data_processor.preprocess(dict(batch))
    stages["y eval target [V]"] = (["phi"], sample["y"])
    if model is not None:
        out, _ = data_processor.postprocess(model(sample["x"]), sample)
        stages["out eval [V]"] = (["phi"], out)

rows = [
    {"stage": stage, "channel": name, "mean": float(c.mean()), "std": float(c.std()),
     "min": float(c.min()), "max": float(c.max())}
    for stage, (names, t) in stages.items()
    for name, c in zip(names, t.unbind(1))
]
with pd.option_context("display.float_format", "{:.4g}".format, "display.width", 120):
    print(pd.DataFrame(rows).to_string(index=False))

# Histograms over the batch and the grid; log y, since rho is zero on most grid points.
n_cols = len(x_names)
fig, axs = plt.subplots(len(stages), n_cols, figsize=(3.2 * n_cols, 2.4 * len(stages)),
                        squeeze=False)
for r, (stage, (names, t)) in enumerate(stages.items()):
    for c in range(n_cols):
        ax = axs[r, c]
        if c >= len(names):
            ax.axis("off")
            continue
        ax.hist(t[:, c].flatten().numpy(), bins=100, log=True)
        ax.set_title(f"{stage}: {names[c]}", fontsize=9)
fig.suptitle(f"Normalization check, {N_SHOW} training samples")
fig.tight_layout()
_save_figures({"normalization_hist": fig}, save_dir)

# Central cuts of sample 0. Only keys starting with "phi" share a colour scale, so the
# ones in V do and the normalized ones get their own.
cuts = {"rho raw": batch["x"][0, 0], "rho model input": stages["x model input"][1][0, 0],
        "phi GT [V]": batch["y"][0, 0], "norm. phi train target": stages["y train target"][1][0, 0]}
if model is not None:
    cuts["norm. phi out train"] = stages["out train"][1][0, 0]
    cuts["phi out eval [V]"] = stages["out eval [V]"][1][0, 0]
plot_central_cut(cuts, title="Sample 0 before / after normalization",
                 name="normalization_cuts", save_dir=save_dir)
plt.show()
