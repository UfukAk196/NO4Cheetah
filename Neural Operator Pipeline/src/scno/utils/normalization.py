"""
Nondimensionalization -- the most important piece of the whole pipeline.

Why at all? Raw charge densities are around 1e-3 C/m^3, potentials around 1e5 V,
cell sizes around 1e-5 m. A network fed such numbers directly mostly learns orders
of magnitude instead of physics. So we clear that away beforehand.

The Poisson equation gives us two properties for this:

  Linearity  phi is linear in rho  -> factor out the total charge Q
  Scaling    length x L scales phi with L^2 (at fixed total charge: 1/L)

Concretely, with Q the total charge on the grid and L a reference length:

    rho_hat = rho * L^3 / Q            (dimensionless density, integral = 1)
    phi_hat = phi * 4*pi*eps0*L / Q    (dimensionless, order of magnitude 1)

The equation then simply reads:  laplace(phi_hat) = -4*pi*rho_hat.

The choice of L is where one can silently shoot oneself in the foot. The obvious
choice would be the cell size -- and that is exactly wrong. A neural operator is
supposed to learn a map between *functions*, not between grids. If L depends on the
cell size, the same physical density maps to a different phi_hat depending on the
resolution, and zero-shot super-resolution falls apart although nothing is wrong
with the network. Hence: L = geometric mean of the three *box edges*. That only
depends on the geometry, not on the discretization.

The geometric mean instead of a single edge is deliberate as well: the gamma
stretching of the longitudinal axis spreads the aspect ratios over several orders of
magnitude, and a symmetric reference noticeably compresses the spread of phi_hat
(measure it with scripts/inspect_scaling.py).

The remaining true parameters are the aspect ratios log(E_i / L). By construction
the three values sum to zero, and they enter the network as constant extra
channels. The operator the network learns is therefore universal: one model covers
all beam sizes, charges and energies, because the scaling was removed analytically
instead of being learned from data.

Check this experimentally: train once without nondimensionalization
(``nondim=False``). The difference in error is typically one to two orders of
magnitude -- a clean ablation result for the thesis.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from scipy.constants import epsilon_0


@dataclass
class Scaling:
    """Keeps the factors of a batch so that the fields can be transformed back."""

    total_charge: torch.Tensor  # (B,) in C
    reference_length: torch.Tensor  # (B,) in m: (Ex*Ey*Ez)^(1/3) of the box edges
    normalized_extent: torch.Tensor  # (B, 3): E_i / L, product = 1, resolution independent

    def to(self, device):
        return Scaling(
            self.total_charge.to(device),
            self.reference_length.to(device),
            self.normalized_extent.to(device),
        )


def nondimensionalize(
    charge_density: torch.Tensor,  # (B, nx, ny, nz) in C/m^3
    cell_size: torch.Tensor,  # (B, 3) in m, tau component already scaled by gamma
    eps: float = 1e-30,
) -> tuple[torch.Tensor, Scaling]:
    grid_shape = torch.tensor(
        charge_density.shape[1:], device=cell_size.device, dtype=cell_size.dtype
    )
    cell_volume = cell_size.prod(dim=1)  # (B,)
    charge_per_cell = charge_density * cell_volume[:, None, None, None]
    total_charge = charge_per_cell.flatten(1).sum(dim=1)  # (B,)
    total_charge = torch.where(
        total_charge.abs() < eps, torch.full_like(total_charge, eps), total_charge
    )

    extent = cell_size * grid_shape[None, :]  # (B, 3) box edges
    reference_length = extent.prod(dim=1).pow(1.0 / 3.0)  # geometric mean of the box edges
    normalized_extent = extent / reference_length[:, None]  # product of the three is 1

    # As a density, not as a charge fraction: this is the quantity that converges to
    # a fixed function when the grid is refined.
    volume_ratio = reference_length.pow(3) / cell_volume  # = number of cells
    rho_hat = (
        charge_per_cell
        / total_charge[:, None, None, None]
        * volume_ratio[:, None, None, None]
    )
    return rho_hat, Scaling(total_charge, reference_length, normalized_extent)


def potential_to_hat(potential: torch.Tensor, scaling: Scaling) -> torch.Tensor:
    """phi (B, nx, ny, nz) in V -> phi_hat (B, nx, ny, nz), dimensionless."""
    factor = 4 * math.pi * epsilon_0 * scaling.reference_length / scaling.total_charge
    return potential * factor[:, None, None, None]


def potential_from_hat(potential_hat: torch.Tensor, scaling: Scaling) -> torch.Tensor:
    """phi_hat (B, nx, ny, nz), dimensionless -> phi (B, nx, ny, nz) in V."""
    factor = scaling.total_charge / (4 * math.pi * epsilon_0 * scaling.reference_length)
    return potential_hat * factor[:, None, None, None]


def stretched_cell_size(cell_size: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
    """Cell size in the rest frame, (B, 3) in m: tau axis stretched by gamma.

    Only needed for cell sizes stored without gamma (the datasets). Inside the lattice,
    ``SpaceChargeKick._solve_poisson_equation`` stretches before the grid-level solve.
    """
    stretched = cell_size.clone()
    stretched[:, 2] = stretched[:, 2] * gamma
    return stretched


def nondimensionalize_fields(
    charge_density: torch.Tensor,  # (B, nx, ny, nz) in C/m^3
    potential: torch.Tensor,  # (B, nx, ny, nz) in V
    cell_size: torch.Tensor,  # (B, 3) in m, WITHOUT gamma (as stored)
    gamma: torch.Tensor,  # (B,)
) -> tuple[torch.Tensor, torch.Tensor, Scaling]:
    """Physical field pair -> (rho_hat, phi_hat, Scaling).

    The files hold rho and phi in physical units; this step happens on loading.
    """
    rho_hat, scaling = nondimensionalize(
        charge_density, stretched_cell_size(cell_size, gamma)
    )
    return rho_hat, potential_to_hat(potential, scaling), scaling


def decode_prediction(
    field: torch.Tensor,  # (B, nx, ny, nz), model output in the units of y
    scaling: Scaling,
    nondim: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Model output -> (phi_hat, phi [V]), each (B, nx, ny, nz).

    ``field`` must already be in the units of ``y`` (see ``evaluate.predict``, which
    undoes the data processor). With ``nondim=False`` the network learns phi in V
    directly; phi_hat is then computed from it for comparison.
    """
    if nondim:
        return field, potential_from_hat(field, scaling)
    return potential_to_hat(field, scaling), field


INPUT_CHANNELS = 4  # rho_hat + 3x log(E_i/L)


def input_channels(rho_hat: torch.Tensor, normalized_extent: torch.Tensor) -> torch.Tensor:
    """Model input x: rho_hat + three constant channels log(E_i/L).

    The only place that defines the input format -- dataset, training, evaluation and
    the Cheetah element all go through here.

    E_i are the (gamma-stretched) box edges, not the electric field. Without them the
    network only sees grid indices and cannot distinguish equal rho_hat at different
    aspect ratios. Grid coordinates are not part of the input: neuralop's ``FNO``
    appends them itself (``positional_embedding="grid"``).

    rho_hat (..., nx, ny, nz), normalized_extent (..., 3) -> x (..., 4, nx, ny, nz)
    """
    grid = rho_hat.shape[-3:]
    log_aspect = normalized_extent.log().to(rho_hat.dtype)[..., :, None, None, None]
    return torch.cat(
        [rho_hat.unsqueeze(-4), log_aspect.expand(*log_aspect.shape[:-3], *grid)], dim=-4
    )
