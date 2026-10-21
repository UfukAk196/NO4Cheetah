"""
G6-Smoke-Test: Datengenerator (G1-G5) + neuralop-Trainer einmal durchlaufen.

Nutzt ``load_cheetah_sc`` (unser Ersatz fuer ``load_darcy_flow_small``) und
trahiert 2 Epochen einen 3D-FNO mit dem neuralop ``Trainer`` - Zeile fuer
Zeile wie im Darcy-Tutorial. Danach werden Train-Loss pro Epoche und die
Eval-Werte (h1, l2) pro Aufloesung ausgegeben.

Aufruf: python scripts/smoke_train.py
"""

from __future__ import annotations

from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]

from neuralop.losses import H1Loss, LpLoss  # noqa: E402
from neuralop.models import FNO  # noqa: E402

from scno.data.dataset import load_cheetah_sc  # noqa: E402
from scno.train import HistoryTrainer  # noqa: E402
from scno.evaluate import plot_predictions, plot_losses  # noqa: E402


# Kleine Parameter-Grenzen, damit der Smoke-Test auf CPU schnell bleibt.
SMOKE_RANGES = {
    "sigma_x": {"min": 5.0e-6, "max": 1.5e-3},
    "sigma_y": {"min": 5.0e-6, "max": 1.5e-3},
    "sigma_tau": {"min": 1.0e-3, "max": 5.0e-3},
    "energy": {"min": 5.0e7, "max": 1.5e8}, # ggf. min noch weiter runter
    "total_charge": {"min": 1.0e-10, "max": 2.0e-9}, # ggf. auf XFEL oder ARES-Panel nachschauen oder Space Charge Paper vom CERN (auch für energy)
    "n_particles": {"min": 1.0e4, "max": 1.0e6},
}
# ggf. Ranges wieder enger machen für den Anfang
# In Zukunft für Evaluierung schauen, ab welchen Ranges der FNO scheitert.

# Create models directory (if it doesn't already exist), see: https://docs.python.org/3/library/pathlib.html#pathlib.Path.mkdir
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, # create parent directories if needed
                 exist_ok=True # if models directory already exists, don't error
)

# Create model save path
MODEL_NAME = "fno_cheetah_space_charge.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

def main() -> int:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Persistenter Datenordner im Projekt (statt tempfile.mkdtemp), damit die
    # generierten .h5-Dateien zwischen Laeufen erhalten bleiben. generate=True
    # erzeugt dann nur noch die fehlenden Dateien (s. missing_files).
    data_root = ROOT / "data"

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
        nondim=False,
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
        in_channels=4,          # rho_hat + 3x log(E_i/L), E_i = Boxkanten
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
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=1e-4)
    
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

    # Save the model state dict
    print(f"Saving model to: {MODEL_SAVE_PATH}")
    torch.save(obj=model.state_dict(), # only saving the state_dict() only saves the learned parameters
            f=MODEL_SAVE_PATH)

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
