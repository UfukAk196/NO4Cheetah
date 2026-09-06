"""
Die ganze Pipeline einmal in Miniaturgroesse durchlaufen lassen.

Winziges Gitter, wenige Proben, drei Epochen. Das Ziel ist nicht Genauigkeit,
sondern die Antwort auf die Frage: laeuft von Datengenerierung bis Auswertung
alles durch, ohne dass eine Achse verrutscht oder ein Tensor die falsche Form
hat? Das erst zu klaeren, bevor man eine GPU-Nacht verbrennt, spart Tage.

Aufruf: python scripts/smoke_test.py
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402
import yaml  # noqa: E402

from scno.data.dataset import SpaceChargeDataset  # noqa: E402
from scno.data.generate import write_hdf5  # noqa: E402
from scno.evaluate import evaluate, gradient_consistency, resolution_study  # noqa: E402
from scno.models.exact import ExactOperator  # noqa: E402
from scno.models.fno import build_model  # noqa: E402
from scno.utils import count_parameters  # noqa: E402


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="scno_smoke_"))
    data_path = tmp / "smoke.h5"
    grid = (16, 16, 16)

    print("[1/5] Daten erzeugen")
    write_hdf5(
        data_path,
        num_samples=8,
        grid_shape=grid,
        num_particles=20_000,
        seed=1,
        batch_size=4,
    )
    ds = SpaceChargeDataset(data_path)
    item = ds[0]
    print(f"      rho_hat {tuple(item['rho_hat'].shape)}, "
          f"Mittelwert {item['rho_hat'].mean():.4f} (Soll 1.0)")
    print(f"      phi_hat Spanne [{item['phi_hat'].min():.3f}, {item['phi_hat'].max():.3f}]")

    print("[2/5] Kontrolle: exakter Solver als Modell (muss Fehler 0 liefern)")
    control = evaluate(ExactOperator(), ds, torch.device("cpu"), batch_size=2)
    control_grad = gradient_consistency(ExactOperator(), ds, torch.device("cpu"), 2)
    print(f"      Potentialfehler {control['potential_rel_l2']['mean']:.2e} (Soll 0)")
    print(f"      Gradienten-Kosinus {control_grad['cosine_similarity_mean']:.6f} (Soll 1)")
    if control["potential_rel_l2"]["mean"] > 1e-4 or control_grad["cosine_similarity_mean"] < 0.999:
        print("      FEHLER: Normierung oder Auswertung stimmt nicht. "
              "Hier weitersuchen, nicht im Netz.")
        return 1

    print("[3/5] Modell bauen und drei Epochen trainieren")
    cfg = yaml.safe_load((ROOT / "configs" / "default.yaml").read_text())
    cfg["data"]["path"] = str(data_path)
    cfg["model"]["width"] = 8
    cfg["model"]["modes"] = [6, 6, 6]
    cfg["model"]["num_layers"] = 2
    cfg["model"]["lifting_channels"] = 16
    cfg["training"]["epochs"] = 3
    cfg["training"]["batch_size"] = 2
    cfg["output_dir"] = str(tmp / "run")
    cfg_path = tmp / "smoke.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))

    result = subprocess.run(
        [sys.executable, "-m", "scno.train", "--config", str(cfg_path)],
        cwd=ROOT, env={**__import__("os").environ, "PYTHONPATH": str(ROOT / "src")},
        capture_output=True, text=True,
    )
    print("      " + "\n      ".join(result.stdout.strip().splitlines()[-4:]))
    if result.returncode != 0:
        print(result.stderr[-3000:])
        return 1

    print("[4/5] Auswerten")
    device = torch.device("cpu")
    model = build_model(cfg["model"]).to(device)
    ckpt = torch.load(tmp / "run" / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    print(f"      {count_parameters(model):,} Parameter")

    acc = evaluate(model, ds, device, batch_size=2)
    print(f"      Potentialfehler {acc['potential_rel_l2']['mean']:.3f}, "
          f"Kraftfehler {acc['force_rel_l2']['mean']:.3f} "
          "(nach drei Epochen erwartungsgemaess gross)")

    res = resolution_study(model, data_path, device, factors=(1, 2))
    for key, val in res.items():
        print(f"      {key}: Gitter {val['grid']}, phi {val['potential_rel_l2_mean']:.3f}")

    print("[5/5] Differenzierbarkeit pruefen")
    grads = gradient_consistency(model, ds, device, num_samples=2)
    print(f"      Kosinusaehnlichkeit Netz vs. exakter Solver: "
          f"{grads['cosine_similarity_mean']:.3f} "
          "(bei untrainiertem Modell beliebig -- der Test prueft nur, dass "
          "Gradienten ueberhaupt durchlaufen)")

    print(f"\nAlles durchgelaufen. Zwischenstaende in {tmp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
