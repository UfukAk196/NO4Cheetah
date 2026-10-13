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
from collections import defaultdict
from pathlib import Path

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
            aspect=item["aspect"][None].to(device).float(),
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
