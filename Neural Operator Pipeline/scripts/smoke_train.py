"""
G6-Smoke-Test: Datengenerator (G1-G5) + neuralop-Trainer einmal durchlaufen.

Nutzt ``load_cheetah_sc`` (unser Ersatz fuer ``load_darcy_flow_small``) und
trahiert 2 Epochen einen 3D-FNO mit dem neuralop ``Trainer`` - Zeile fuer
Zeile wie im Darcy-Tutorial. Danach werden Train-Loss pro Epoche und die
Eval-Werte (h1, l2) pro Aufloesung ausgegeben.

Aufruf: python scripts/smoke_train.py
"""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from neuralop import Trainer  # noqa: E402
from neuralop.losses import H1Loss, LpLoss  # noqa: E402
from neuralop.models import FNO  # noqa: E402

from cheetah_sc.dataset import load_cheetah_sc  # noqa: E402


# Kleine Parameter-Grenzen, damit der Smoke-Test auf CPU schnell bleibt.
SMOKE_RANGES = {
    "sigma_x": {"min": 5.0e-4, "max": 1.5e-3},
    "sigma_y": {"min": 5.0e-4, "max": 1.5e-3},
    "sigma_tau": {"min": 1.0e-3, "max": 5.0e-3},
    "energy": {"min": 5.0e8, "max": 1.5e9},
    "total_charge": {"min": 1.0e-10, "max": 2.0e-9},
    "n_particles": {"min": 1.0e4, "max": 1.0e5},
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


class HistoryTrainer(Trainer):
    """``Trainer``, der die Metriken jeder Epoche mitschreibt.

    Der neuralop-``Trainer`` gibt die Werte nur nach stdout aus und speichert
    sie nicht. Hier werden ``log_training`` (jede Epoche) und ``log_eval``
    (alle ``eval_interval`` Epochen) ueberschrieben und die Werte zusaetzlich
    in ``self.history`` abgelegt:

        history["train"] = {"epoch": [...], "avg_loss": [...], "train_err": [...]}
        history["eval"]  = {"epoch": [...], "<res>_h1": [...], "<res>_l2": [...]}
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.history = {"train": {}, "eval": {}}

    def log_training(self, epoch, time, avg_loss, train_err, avg_lasso_loss=None, lr=None):
        super().log_training(
            epoch=epoch,
            time=time,
            avg_loss=avg_loss,
            train_err=train_err,
            avg_lasso_loss=avg_lasso_loss,
            lr=lr,
        )
        train_hist = self.history["train"]
        train_hist.setdefault("epoch", []).append(epoch)
        train_hist.setdefault("avg_loss", []).append(float(avg_loss))
        train_hist.setdefault("train_err", []).append(float(train_err))
        train_hist.setdefault("lr", []).append(float(lr) if lr is not None else float("nan"))

    def log_eval(self, epoch, eval_metrics):
        super().log_eval(epoch=epoch, eval_metrics=eval_metrics)
        eval_hist = self.history["eval"]
        eval_hist.setdefault("epoch", []).append(epoch)
        for metric, value in eval_metrics.items():
            if isinstance(value, torch.Tensor):
                value = value.item()
            if isinstance(value, (int, float)):
                eval_hist.setdefault(metric, []).append(float(value))


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


def main() -> int:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Persistenter Datenordner im Projekt (statt tempfile.mkdtemp), damit die
    # generierten .pt-Dateien zwischen Laeufen erhalten bleiben. generate=True
    # erzeugt dann nur noch die fehlenden Dateien (s. _missing_files).
    data_root = ROOT / "data" / "cheetah_sc_smoke"

    # --- Daten: kleines Setup, wird bei Bedarf generiert (generate=True) ---
    n_train, batch_size = 16, 4
    n_tests, test_batch_sizes = [4, 4, 4], [4, 4, 4]
    train_resolution, test_resolutions = 16, [16, 32, 64]

    print("== load_cheetah_sc (Daten laden/generieren) ==")
    train_loader, test_loaders, data_processor = load_cheetah_sc(
        data_root=data_root,
        n_train=n_train,
        n_tests=n_tests,
        batch_size=batch_size,
        test_batch_sizes=test_batch_sizes,
        train_resolution=train_resolution,
        test_resolutions=test_resolutions,
        generate=True,
        seed=0,
        ranges=SMOKE_RANGES,
        num_workers=0,
    )
    data_processor = data_processor.to(device)

    # --- Modell: 3D-FNO ---, tag="before_training_res16"
    model = FNO(
        n_modes=(8, 8, 8),      # max. lowest fourier modes per dimension (k_max)
        #in_channels=4,          # rho_hat + 3x E_hat
        in_channels=1,          # rho_hat only
        out_channels=1,         # phi_hat
        hidden_channels=16,
        n_layers=3,
        domain_padding=0.25
    ).to(device)
    print(f"FNO Parameter: {sum(p.numel() for p in model.parameters()):,}")

    # --- Plot predictions before training ---
    batch = next(iter(test_loaders[16]))   # ein Roh-Batch: {"x","y"}
    plot_predictions(model, batch, data_processor, tag=f"before_training")

    # --- Optimizer/Scheduler wie im Darcy-Tutorial ---
    n_epochs = 1000
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=1e-6)

    # --- Verluste ---
    training_loss = H1Loss(d=3)
    eval_losses = {"h1": H1Loss(d=3), "l2": LpLoss(d=3, p=2)}

    # --- Trainer (Darcy-Tutorial-Muster, mit Loss-Historie) ---
    trainer = HistoryTrainer(
        model=model,
        n_epochs=n_epochs,
        device=device,
        data_processor=data_processor,
        eval_interval=100,
        verbose=True,   # druckt avg_loss je Epoche + Eval-Werte je Aufloesung
    )

    print(f"== Training ({n_epochs} Epochen) ==")
    trainer.train(
        train_loader=train_loader,
        test_loaders=test_loaders,
        optimizer=optimizer,
        scheduler=scheduler,
        training_loss=training_loss,
        eval_losses=eval_losses,
    )

    # --- Explizite Zusammenfassung pro Aufloesung ---
    print("\n== Eval-Werte pro Aufloesung ==")
    model.eval()
    for res, loader in test_loaders.items():
        metrics = {name: 0.0 for name in eval_losses}
        count = 0
        with torch.no_grad():
            for data_dict in loader:
                data_dict = data_processor.preprocess(data_dict)
                out = model(data_dict["x"])
                y = data_dict["y"]
                for name, fn in eval_losses.items():
                    metrics[name] += fn(out, y).item() * out.size(0)
                count += out.size(0)
        line = ", ".join(f"{name}={v / count:.4f}" for name, v in metrics.items())
        print(f"  Aufloesung {res}: {line}")

    # --- Loss-Kurven aus der Trainings-Historie ---
    plot_losses(trainer.history, tag="smoke")

    # --- Plot predictions after training for each resolution ---
    model.eval()
    for res, loader in test_loaders.items():
        batch = next(iter(loader))
        plot_predictions(model, batch, data_processor, tag=f"after_training_res{res}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
