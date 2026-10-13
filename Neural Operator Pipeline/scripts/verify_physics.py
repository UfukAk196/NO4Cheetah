"""
Prueft den klassischen Solver gegen eine Loesung, die man von Hand hinschreiben kann.

Eine homogen geladene Kugel mit Radius R und Gesamtladung Q hat aussen das
Potential Q/(4*pi*eps0*r) und innen Q*(3R^2 - r^2)/(8*pi*eps0*R^3).

Diesen Test zuerst laufen lassen, bevor irgendein Netz trainiert wird. Wenn der
Lehrer falsch rechnet, lernt der Schueler zuverlaessig Unsinn -- und man sucht
den Fehler tagelang im Netz.

Aufruf: python scripts/verify_physics.py
"""

import math
import torch
from scno.utils.physics import EPSILON_0, solve_poisson  # noqa: E402


def main(n: int = 64, radius: float = 1.0, total_charge: float = 1.0) -> int:
    half_extent = 2.0 * radius
    cell = torch.full((1, 3), 2 * half_extent / n, dtype=torch.float64)

    axis = (torch.arange(n, dtype=torch.float64) + 0.5) * cell[0, 0] - half_extent
    X, Y, Z = torch.meshgrid(axis, axis, axis, indexing="ij")
    r = (X.square() + Y.square() + Z.square()).sqrt()

    inside = r <= radius
    volume = 4 / 3 * math.pi * radius**3
    density = torch.where(inside, total_charge / volume, torch.zeros(())).to(torch.float64)

    phi_num = solve_poisson(density[None], cell)[0]

    k = 1.0 / (4 * math.pi * EPSILON_0)
    phi_exact = torch.where(
        inside,
        k * total_charge * (3 * radius**2 - r.square()) / (2 * radius**3),
        k * total_charge / r.clamp(min=1e-12),
    )

    # Randzellen weglassen: dort schneidet das endliche Gitter die Loesung ab
    sl = slice(4, -4)
    a = phi_num[sl, sl, sl]
    b = phi_exact[sl, sl, sl]
    rel = ((a - b).norm() / b.norm()).item()
    max_rel = ((a - b).abs() / b.abs()).max().item()

    print(f"Gitter {n}^3, Kugelradius {radius}, Ladung {total_charge} C")
    print(f"  relativer L2-Fehler : {rel:.3e}")
    print(f"  groesster Punktfehler: {max_rel:.3e}")
    print(f"  phi im Zentrum      : numerisch {phi_num[n//2, n//2, n//2]:.6e} V, "
          f"exakt {phi_exact[n//2, n//2, n//2]:.6e} V")

    ok = rel < 0.02
    print("BESTANDEN" if ok else "DURCHGEFALLEN -- Solver oder Gitteraufloesung pruefen")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
