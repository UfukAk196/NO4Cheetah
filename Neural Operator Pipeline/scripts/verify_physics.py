"""
Checks the classical solver against a solution one can write down by hand.

A homogeneously charged sphere of radius R and total charge Q has the potential
Q/(4*pi*eps0*r) outside and Q*(3R^2 - r^2)/(8*pi*eps0*R^3) inside.

Run this test first, before any network is trained. If the teacher computes wrongly,
the student reliably learns nonsense -- and one searches for the bug in the network
for days.

The solver checked is Cheetah's own grid-level solve
(``SpaceChargeKick._solve_poisson_on_grid``), i.e. exactly what produces the labels.

Call: python scripts/verify_physics.py
"""

import math

import torch
from cheetah.accelerator import SpaceChargeKick
from scipy.constants import epsilon_0


def main(n: int = 64, radius: float = 1.0, total_charge: float = 1.0) -> int:
    half_extent = 2.0 * radius
    cell = torch.full((1, 3), 2 * half_extent / n, dtype=torch.float64)

    axis = (torch.arange(n, dtype=torch.float64) + 0.5) * cell[0, 0] - half_extent
    X, Y, Z = torch.meshgrid(axis, axis, axis, indexing="ij")
    r = (X.square() + Y.square() + Z.square()).sqrt()

    inside = r <= radius
    volume = 4 / 3 * math.pi * radius**3
    density = torch.where(inside, total_charge / volume, torch.zeros(())).to(
        torch.float64
    )

    solver = SpaceChargeKick(effect_length=torch.tensor(0.0))
    phi_num = solver._solve_poisson_on_grid(density[None], cell)[0]

    k = 1.0 / (4 * math.pi * epsilon_0)
    phi_exact = torch.where(
        inside,
        k * total_charge * (3 * radius**2 - r.square()) / (2 * radius**3),
        k * total_charge / r.clamp(min=1e-12),
    )

    # Leave out the boundary cells: there the finite grid cuts off the solution
    sl = slice(4, -4)
    a = phi_num[sl, sl, sl]
    b = phi_exact[sl, sl, sl]
    rel = ((a - b).norm() / b.norm()).item()
    max_rel = ((a - b).abs() / b.abs()).max().item()

    print(f"Grid {n}^3, sphere radius {radius}, charge {total_charge} C")
    print(f"  relative L2 error : {rel:.3e}")
    print(f"  largest point error: {max_rel:.3e}")
    print(
        f"  phi at the centre  : numerical {phi_num[n//2, n//2, n//2]:.6e} V, "
        f"exact {phi_exact[n//2, n//2, n//2]:.6e} V"
    )

    ok = rel < 0.02
    print("PASSED" if ok else "FAILED -- check the solver or the grid resolution")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
