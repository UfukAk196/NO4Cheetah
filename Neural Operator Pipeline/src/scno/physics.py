"""
Klassischer Space-Charge-Solver, standalone in reinem PyTorch.

Herausgeloest aus `cheetah.accelerator.SpaceChargeKick`, damit die Pipeline
ohne Cheetah-Installation laeuft. Das ist bewusst so gebaut: dieses Modul ist
der *Lehrer*, der die Trainingsdaten erzeugt, und gleichzeitig die
Referenzloesung, gegen die der Neural Operator spaeter gemessen wird.

Methode (Integrated Green Function / Hockney):
  1. Ladung per Cloud-in-Cell auf ein Gitter legen  -> rho
  2. Gitter in allen Richtungen verdoppeln (offene Randbedingungen)
  3. rho mit der integrierten Greenschen Funktion falten (per FFT) -> phi
  4. Zentrale Differenzen auf phi -> Feldgradienten

Alles differenzierbar, alles batched: Achse 0 ist immer der Batch.
"""

from __future__ import annotations

import math

import torch

EPSILON_0 = 8.8541878128e-12  # F/m
ELEMENTARY_CHARGE = 1.602176634e-19  # C
SPEED_OF_LIGHT = 299792458.0  # m/s


# --------------------------------------------------------------------------- #
# Ladungsdeposition
# --------------------------------------------------------------------------- #
def cloud_in_cell(
    positions: torch.Tensor,  # (B, N, 3) in Metern, zentriert um 0
    charges: torch.Tensor,  # (B, N) in Coulomb
    grid_shape: tuple[int, int, int],
    half_extent: torch.Tensor,  # (B, 3) halbe Kantenlaenge des Gitters
) -> torch.Tensor:
    """Legt Punktladungen per trilinearer Gewichtung auf ein regulaeres Gitter.

    Rueckgabe: (B, nx, ny, nz) Ladung pro Zelle (nicht Dichte!).
    """
    batch, num_particles, _ = positions.shape
    device, dtype = positions.device, positions.dtype
    shape_t = torch.tensor(grid_shape, device=device, dtype=dtype)

    cell_size = 2.0 * half_extent / shape_t  # (B, 3)
    # Auf Zellkoordinaten umrechnen; Zellzentren liegen bei 0.5, 1.5, ...
    normalized = (positions + half_extent[:, None, :]) / cell_size[:, None, :] - 0.5

    base = normalized.floor()
    frac = normalized - base
    base = base.to(torch.long)  # (B, N, 3)

    grid = torch.zeros(batch, *grid_shape, device=device, dtype=dtype)
    flat = grid.view(batch, -1)
    nx, ny, nz = grid_shape

    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                offset = torch.tensor([dx, dy, dz], device=device)
                idx = base + offset  # (B, N, 3)
                w = (
                    torch.where(torch.tensor(dx, device=device) == 1, frac[..., 0], 1 - frac[..., 0])
                    * torch.where(torch.tensor(dy, device=device) == 1, frac[..., 1], 1 - frac[..., 1])
                    * torch.where(torch.tensor(dz, device=device) == 1, frac[..., 2], 1 - frac[..., 2])
                )
                valid = (
                    (idx[..., 0] >= 0) & (idx[..., 0] < nx)
                    & (idx[..., 1] >= 0) & (idx[..., 1] < ny)
                    & (idx[..., 2] >= 0) & (idx[..., 2] < nz)
                )
                idx_c = idx.clamp(min=0)
                idx_c[..., 0] = idx_c[..., 0].clamp(max=nx - 1)
                idx_c[..., 1] = idx_c[..., 1].clamp(max=ny - 1)
                idx_c[..., 2] = idx_c[..., 2].clamp(max=nz - 1)
                lin = idx_c[..., 0] * ny * nz + idx_c[..., 1] * nz + idx_c[..., 2]
                contrib = (w * charges * valid).to(dtype)
                flat = flat.scatter_add(1, lin, contrib)

    return flat.view(batch, *grid_shape)


# --------------------------------------------------------------------------- #
# Integrated Green Function
# --------------------------------------------------------------------------- #
def _integrated_potential(x, y, tau):
    """Stammfunktion des 1/r-Kerns ueber eine Zelle (numerisch gutmuetige Form)."""
    r = (x.square() + y.square() + tau.square()).sqrt()
    return (
        -0.5 * tau.square() * (x * y / (tau * r)).atan()
        - 0.5 * y.square() * (x * tau / (y * r)).atan()
        - 0.5 * x.square() * (y * tau / (x * r)).atan()
        + y * tau * (x / (y.square() + tau.square()).sqrt()).asinh()
        + x * tau * (y / (x.square() + tau.square()).sqrt()).asinh()
        + x * y * (tau / (x.square() + y.square()).sqrt()).asinh()
    )


def integrated_green_function(
    grid_shape: tuple[int, int, int],
    cell_size: torch.Tensor,  # (B, 3), dritte Komponente bereits mit gamma skaliert
) -> torch.Tensor:
    """Greensche Funktion auf dem verdoppelten Gitter, periodisch gespiegelt."""
    device, dtype = cell_size.device, cell_size.dtype
    nx, ny, nz = grid_shape
    dx, dy, dtau = cell_size[:, 0], cell_size[:, 1], cell_size[:, 2]

    ix, iy, itau = torch.meshgrid(
        torch.arange(nx, device=device, dtype=dtype),
        torch.arange(ny, device=device, dtype=dtype),
        torch.arange(nz, device=device, dtype=dtype),
        indexing="ij",
    )
    x = ix[None] * dx[:, None, None, None]
    y = iy[None] * dy[:, None, None, None]
    t = itau[None] * dtau[:, None, None, None]
    hx = 0.5 * dx[:, None, None, None]
    hy = 0.5 * dy[:, None, None, None]
    ht = 0.5 * dtau[:, None, None, None]

    g = (
        _integrated_potential(x + hx, y + hy, t + ht)
        - _integrated_potential(x - hx, y + hy, t + ht)
        - _integrated_potential(x + hx, y - hy, t + ht)
        - _integrated_potential(x + hx, y + hy, t - ht)
        + _integrated_potential(x + hx, y - hy, t - ht)
        + _integrated_potential(x - hx, y + hy, t - ht)
        + _integrated_potential(x - hx, y - hy, t + ht)
        - _integrated_potential(x - hx, y - hy, t - ht)
    )

    out = torch.zeros(cell_size.shape[0], 2 * nx, 2 * ny, 2 * nz, device=device, dtype=dtype)
    out[:, :nx, :ny, :nz] = g
    out[:, nx + 1 :, :ny, :nz] = g[:, 1:, :, :].flip(dims=[1])
    out[:, :nx, ny + 1 :, :nz] = g[:, :, 1:, :].flip(dims=[2])
    out[:, :nx, :ny, nz + 1 :] = g[:, :, :, 1:].flip(dims=[3])
    out[:, nx + 1 :, ny + 1 :, :nz] = g[:, 1:, 1:, :].flip(dims=[1, 2])
    out[:, :nx, ny + 1 :, nz + 1 :] = g[:, :, 1:, 1:].flip(dims=[2, 3])
    out[:, nx + 1 :, :ny, nz + 1 :] = g[:, 1:, :, 1:].flip(dims=[1, 3])
    out[:, nx + 1 :, ny + 1 :, nz + 1 :] = g[:, 1:, 1:, 1:].flip(dims=[1, 2, 3])
    return out


def solve_poisson(
    charge_density: torch.Tensor,  # (B, nx, ny, nz) in C/m^3
    cell_size: torch.Tensor,  # (B, 3), tau-Komponente mit gamma skaliert
) -> torch.Tensor:
    """Loest die Poisson-Gleichung per FFT-Faltung (Hockney, offene Raender).

    Rueckgabe: Potential (B, nx, ny, nz) in Volt.
    """
    batch, nx, ny, nz = charge_density.shape
    padded = charge_density.new_zeros(batch, 2 * nx, 2 * ny, 2 * nz)
    padded[:, :nx, :ny, :nz] = charge_density

    rho_ft = torch.fft.rfftn(padded, dim=[1, 2, 3])
    g_ft = torch.fft.rfftn(integrated_green_function((nx, ny, nz), cell_size), dim=[1, 2, 3])
    phi = torch.fft.irfftn(rho_ft * g_ft, dim=[1, 2, 3]).real
    return phi[:, :nx, :ny, :nz] / (4 * math.pi * EPSILON_0)


def potential_gradient(
    potential: torch.Tensor,  # (B, nx, ny, nz)
    cell_size: torch.Tensor,  # (B, 3)
) -> torch.Tensor:
    """Zentrale Differenzen, Null am Rand. Rueckgabe: (B, 3, nx, ny, nz)."""
    inv = cell_size.reciprocal()
    gx = torch.zeros_like(potential)
    gy = torch.zeros_like(potential)
    gt = torch.zeros_like(potential)
    gx[:, 1:-1] = (potential[:, 2:] - potential[:, :-2]) * (0.5 * inv[:, 0, None, None, None])
    gy[:, :, 1:-1] = (potential[:, :, 2:] - potential[:, :, :-2]) * (0.5 * inv[:, 1, None, None, None])
    gt[:, :, :, 1:-1] = (potential[:, :, :, 2:] - potential[:, :, :, :-2]) * (
        0.5 * inv[:, 2, None, None, None]
    )
    return torch.stack([gx, gy, gt], dim=1)


def space_charge_field(
    charge_density: torch.Tensor,
    cell_size: torch.Tensor,
    gamma: torch.Tensor,  # (B,)
) -> torch.Tensor:
    """E + v x B auf dem Gitter, so wie SpaceChargeKick._E_plus_vB_field es rechnet."""
    scaled = cell_size.clone()
    scaled[:, 2] = scaled[:, 2] * gamma
    phi = solve_poisson(charge_density, scaled)
    igamma2 = torch.where(gamma != 0, gamma.square().reciprocal(), torch.zeros_like(gamma))
    return -igamma2[:, None, None, None, None] * potential_gradient(phi, cell_size)
