"""
Auswertung -- und zwar die Zahlen, die in der Thesis wirklich zaehlen.

Vier Fragen werden hier beantwortet:

  1. Wie gut trifft das Modell das Potential?           (relativer L2)
  2. Wie gut trifft es die Kraefte?                     (Gradientenfehler)
     Das ist die entscheidende Groesse: der Kick ist ein Gradient.
  3. Haelt es ueber Verteilungsfamilien hinweg?         (Aufschluesselung nach Typ)
  4. Ueberlebt es eine andere Aufloesung?               (Zero-Shot-Super-Resolution)
     Das ist das Alleinstellungsmerkmal von Neural Operators gegenueber CNNs --
     also messen und zeigen, statt nur behaupten.

Aufruf:
    python -m evaluate --checkpoint outputs/best.pt --data data/val.h5
"""

from __future__ import annotations

import argparse
import json
import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

import torch

from scno.data.dataset import CheetahDataset
from scno.data.generate import DISTRIBUTIONS
from scno.utils.losses import central_gradient, relative_l2
from scno.models.fno import build_model
from scno.utils.normalization import Scaling, build_input
from scno.train import batch_to_input
from scno.utils.common import load_checkpoint, pick_device


@torch.no_grad()
def evaluate(model, dataset, device, batch_size: int = 4) -> dict:
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    per_kind = defaultdict(lambda: {"field": [], "grad": []})
    field_errors, grad_errors = [], []

    model.eval()
    for batch in loader:
        x, target, _ = batch_to_input(batch, device)
        pred = model(x)
        for i in range(x.shape[0]):
            fe = float(relative_l2(pred[i : i + 1], target[i : i + 1]))
            ge = float(
                relative_l2(
                    central_gradient(pred[i : i + 1]), central_gradient(target[i : i + 1])
                )
            )
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
def resolution_study(model, data_path, device, factors=(1, 2), batch_size: int = 2) -> dict:
    """Trainiert auf einer Aufloesung, getestet auf mehreren.

    factor=2 bedeutet halbe Aufloesung pro Achse. Ein FNO kann das ohne
    Nachtraining, weil er im Fourierraum arbeitet -- ein CNN kann es nicht.
    """
    out = {}
    for f in factors:
        ds = CheetahDataset(data_path, downsample=f)
        res = evaluate(model, ds, device, batch_size=batch_size)
        out[f"downsample_{f}"] = {
            "grid": tuple(s // f for s in ds.grid_shape),
            "potential_rel_l2_mean": res["potential_rel_l2"]["mean"],
            "force_rel_l2_mean": res["force_rel_l2"]["mean"],
        }
    return out


def gradient_consistency(model, dataset, device, num_samples: int = 4) -> dict:
    """Prueft, ob Gradienten *durch* das Modell brauchbar sind.

    Genau darum geht es in dieser Arbeit: der Surrogat muss nicht nur richtig
    rechnen, sondern auch richtig ableiten -- sonst taugt er nicht fuer
    gradientenbasierte Optimierung von Strahlparametern.

    Verglichen wird d(Zielgroesse)/d(rho) einmal durch das Netz und einmal
    durch den exakten Solver. Berichtet wird die Kosinusaehnlichkeit der
    beiden Gradientenfelder: 1.0 heisst gleiche Richtung, und die Richtung
    ist das, was ein Optimierer tatsaechlich verwendet.
    """
    from scno.utils.physics import solve_poisson
    from scno.utils.normalization import potential_to_hat

    model.eval()
    similarities, magnitudes = [], []

    for i in range(min(num_samples, len(dataset))):
        item = dataset[i]
        rho = item["rho_hat"][None].to(device).float().requires_grad_(True)
        scaling = Scaling(
            total_charge=item["total_charge"][None].to(device).float(),
            reference_length=item["reference_length"][None].to(device).float(),
            normalized_extent=item["normalized_extent"][None].to(device).float(),
        )

        # Skalare Zielgroesse: mittleres Quadrat des Potentials -- steht
        # stellvertretend fuer jede Groesse, nach der man spaeter optimiert.
        pred = model(build_input(rho, scaling))
        objective_model = pred.square().mean()
        grad_model = torch.autograd.grad(objective_model, rho)[0]

        rho_ref = item["rho_hat"][None].to(device).double().requires_grad_(True)
        cell = item["cell_size"][None].to(device).double().clone()
        # Die longitudinale Zelle ist im Datensatz bereits ohne gamma
        # gespeichert; der Solver braucht sie gestreckt (modifizierte
        # Poisson-Gleichung im Laborsystem).
        cell[:, 2] = cell[:, 2] * item["gamma"].to(device).double()
        # rho_hat ist die dimensionslose Dichte rho*L^3/Q -> zurueck auf C/m^3
        ref_len = item["reference_length"].to(device).double()
        density = rho_ref * item["total_charge"].to(device).double() / ref_len.pow(3)
        phi_ref = solve_poisson(density, cell)
        phi_hat_ref = potential_to_hat(phi_ref, scaling.to(device))
        objective_ref = phi_hat_ref.square().mean()
        grad_ref = torch.autograd.grad(objective_ref, rho_ref)[0].float()

        a = grad_model.flatten()
        b = grad_ref.flatten()
        similarities.append(float(torch.nn.functional.cosine_similarity(a, b, dim=0)))
        magnitudes.append(float(a.norm() / (b.norm() + 1e-30)))

    return {
        "cosine_similarity_mean": float(torch.tensor(similarities).mean()),
        "cosine_similarity_min": float(torch.tensor(similarities).min()),
        "magnitude_ratio_mean": float(torch.tensor(magnitudes).mean()),
        "n": len(similarities),
    }

def plot_predictions(model, test_samples, data_processor, tag="", save_dir=None):
    """Plot FNO-Predictions auf Cheetah-SC-Daten (3D, 32^3).

    `test_samples` ist ein Roh-Batch-Dict aus einem test_loader:
        {"x": (B,4,32,32,32), "y": (B,1,32,32,32)}
    (oder eine Liste solcher Batches). Es werden die ersten 3 Proben des
    ersten Batches geplottet.

    Daten sind ENT-DIMENSIONALISIERT (rho_hat, phi_hat); die Achsen sind
    Gitter-Indizes (keine mm-Extent, kein grid_dimensions im Batch).
    y und die Vorhersage werden NICHT rücktransformiert (direkt gerawht).

    Für jede Probe werden drei Zentralschnitte durch das Gitter gezeigt,
    im Stil des SciBeam-Plots:
        x-y  bei tau = cen,  x-tau bei y = cen,  y-tau bei x = cen

    Kanal 0 von x ist rho_hat (Eingabe); y/out zeigen phi_hat (GT/Pred).
    """
    if isinstance(test_samples, dict):
        test_samples = [test_samples]
    batch = test_samples[0]

    # Vorverarbeitung (eval-Modus: x/y bleiben entdimensionalisiert,
    # da encode_input=False und out_normalizer nur im training greift).
    data = data_processor.preprocess(batch, batched=True)
    x = data["x"]          # (B, 4, 32, 32, 32)
    y = data["y"]          # (B, 1, 32, 32, 32)
    with torch.no_grad():
        out = model(x)     # (B, 1, 32, 32, 32)

    B = min(3, x.shape[0])  # hoechstens 3 Proben

    rho = x[:, 0]            # Kanal 0 = rho_hat: (B, nx, ny, nz)
    phi_gt = y[:, 0]         # (B, nx, ny, nz)
    phi_pred = out[:, 0]     # (B, nx, ny, nz)

    nx, ny, nz = rho.shape[1], rho.shape[2], rho.shape[3]
    cx, cy, ct = nx // 2, ny // 2, nz // 2

    cuts = [
        ("x-y  (tau = cen)", (slice(None), slice(None), ct), (0, 1)),
        ("x-tau (y = cen)", (slice(None), cy, slice(None)), (0, 2)),
        ("y-tau (x = cen)", (cx, slice(None), slice(None)), (1, 2)),
    ]
    axis_names = ["x / idx", "y / idx", "tau / idx"]

    rho_list = [rho[i].detach().cpu().numpy() for i in range(B)]
    gt_list = [phi_gt[i].detach().cpu().numpy() for i in range(B)]
    pred_list = [phi_pred[i].detach().cpu().numpy() for i in range(B)]

    # ----- Abbildung 1: Eingabe (rho_hat), 3 Proben x 3 Zentralschnitte -----
    fig1, axs1 = plt.subplots(B, 3, figsize=(11, 3.5 * B))
    if B == 1:
        axs1 = axs1[None, :]
    for i in range(B):
        field = rho_list[i]
        for j, (title, index, (ia, ib)) in enumerate(cuts):
            ax = axs1[i, j]
            image = ax.imshow(field[index].T, origin="lower", aspect="auto", cmap="viridis")
            fig1.colorbar(image, ax=ax)
            ax.set_title(f"rho_hat   {title}" if i == 0 or True else title)
            ax.set_xlabel(axis_names[ia])
            ax.set_ylabel(axis_names[ib])
    fig1.suptitle("Input rho_hat (entdimensionalisiert)", y=0.99)
    fig1.tight_layout()

    # ----- Abbildung 2: phi_hat, Ground-Truth vs. Prediction -----
    # Layout: eine Zeile je Probe; je Probe [GT: 3 Cuts | Pred: 3 Cuts]
    fig2, axs2 = plt.subplots(B, 6, figsize=(22, 3.5 * B))
    if B == 1:
        axs2 = axs2[None, :]
    for i in range(B):
        for j, (title, index, (ia, ib)) in enumerate(cuts):
            # Ground-Truth (Spalten 0..2)
            ax_gt = axs2[i, j]
            image = ax_gt.imshow(gt_list[i][index].T, origin="lower", aspect="auto", cmap="magma")
            fig2.colorbar(image, ax=ax_gt)
            ax_gt.set_title(f"phi_hat GT   {title}")
            ax_gt.set_xlabel(axis_names[ia])
            ax_gt.set_ylabel(axis_names[ib])

            # Prediction (Spalten 3..5)
            ax_pr = axs2[i, j + 3]
            image = ax_pr.imshow(pred_list[i][index].T, origin="lower", aspect="auto", cmap="magma")
            fig2.colorbar(image, ax=ax_pr)
            ax_pr.set_title(f"phi_hat Pred {title}")
            ax_pr.set_xlabel(axis_names[ia])
            ax_pr.set_ylabel(axis_names[ib])
    fig2.suptitle("phi_hat (entdimensionalisiert): Ground-Truth vs. Prediction", y=0.99)
    fig2.tight_layout()

    # ----- Figuren speichern -----
    if save_dir is None:
        save_dir = ROOT / "figures"
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{tag}" if tag else ""
    fig1_path = save_dir / f"rho_hat{suffix}.png"
    fig2_path = save_dir / f"phi_hat_gt_vs_pred{suffix}.png"
    fig1.savefig(fig1_path, dpi=150, bbox_inches="tight")
    fig2.savefig(fig2_path, dpi=150, bbox_inches="tight")
    print(f"Figuren gespeichert: {fig1_path}, {fig2_path}")

    plt.show()
    return fig1, fig2



def plot_losses(history, save_dir=None, tag=""):
    """Plottet Trainings- und Eval-Lossverlaeufe aus ``HistoryTrainer.history``.

    Drei Panels:
      1. Trainings-Loss (avg_loss, train_err) ueber die Epochen (log-y).
      2. Eval-Loss je Test-Aufloesung (h1, l2) ueber die Epochen (log-y).
      3. Lernrate ueber die Epochen (log-y).

    Die Eval-Kurven haben nur alle ``eval_interval`` Epochen Punkte.
    """
    train_hist = history.get("train", {})
    eval_hist = history.get("eval", {})

    fig, axs = plt.subplots(1, 3, figsize=(18, 5))

    # --- Panel 1: Trainings-Loss ---
    ax = axs[0]
    if train_hist.get("epoch"):
        ax.plot(train_hist["epoch"], train_hist["avg_loss"], label="avg_loss", color="tab:blue")
        ax.plot(train_hist["epoch"], train_hist["train_err"], label="train_err", color="tab:orange", alpha=0.7)
        ax.set_yscale("log")
        ax.legend()
    else:
        ax.text(0.5, 0.5, "keine Trainingsdaten", ha="center", va="center")
    ax.set_title("Trainings-Loss")
    ax.set_xlabel("Epoche")
    ax.set_ylabel("Loss (log)")
    ax.grid(True, which="both", alpha=0.3)

    # --- Panel 2: Eval-Loss je Aufloesung ---
    ax = axs[1]
    eval_epochs = eval_hist.get("epoch", [])
    if eval_epochs:
        # Metriknamen haben die Form "<res>_<loss>", z.B. "16_h1"
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
        ax.text(0.5, 0.5, "keine Eval-Daten", ha="center", va="center")
    ax.set_title("Eval-Loss je Aufloesung")
    ax.set_xlabel("Epoche")
    ax.set_ylabel("Loss (log)")
    ax.grid(True, which="both", alpha=0.3)

    # --- Panel 3: Lernrate ---
    ax = axs[2]
    if train_hist.get("epoch") and train_hist.get("lr"):
        ax.plot(train_hist["epoch"], train_hist["lr"], color="tab:green")
        ax.set_yscale("log")
    else:
        ax.text(0.5, 0.5, "keine LR-Daten", ha="center", va="center")
    ax.set_title("Lernrate")
    ax.set_xlabel("Epoche")
    ax.set_ylabel("lr (log)")
    ax.grid(True, which="both", alpha=0.3)

    fig.suptitle("Trainingsverlauf (Smoke-Test)", y=1.0)
    fig.tight_layout()

    if save_dir is None:
        save_dir = ROOT / "figures"
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{tag}" if tag else ""
    path = save_dir / f"loss_curves{suffix}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Figur gespeichert: {path}")

    plt.show()
    return fig


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="outputs/best.pt")
    p.add_argument("--data", default="data/val.h5")
    p.add_argument("--out", default="outputs/evaluation.json")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--skip-resolution", action="store_true")
    args = p.parse_args()

    device = pick_device("auto")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = build_model(ckpt["config"]["model"]).to(device)
    load_checkpoint(args.checkpoint, model, map_location=device)

    dataset = CheetahDataset(args.data)
    report = {"accuracy": evaluate(model, dataset, device, args.batch_size)}
    report["gradients"] = gradient_consistency(model, dataset, device)
    if not args.skip_resolution:
        report["resolution"] = resolution_study(model, args.data, device)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
