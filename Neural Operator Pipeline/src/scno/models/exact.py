"""
The exact solver, wrapped as a model.

Sounds like a gimmick, but it is the most valuable debugging tool in the whole
project: this "model" must pass every evaluation with zero error and a gradient
cosine of 1.0. If it does not, the bug is in the normalization, the axis order or the
evaluation -- not in the network. Only once this control is clean do the numbers of
the trained model mean anything.

For the thesis it is also the upper bound: everything a surrogate can achieve is here.
"""

from __future__ import annotations

import math

import torch
from cheetah.accelerator import SpaceChargeKick
from cheetah.particles import ParticleBeam, Species
from neuralop.models.base_model import BaseModel
from scipy.constants import epsilon_0


def solve_poisson_on_grid(
    charge_density: torch.Tensor,  # (B, nx, ny, nz) in C/m^3
    cell_size: torch.Tensor,  # (B, 3) in m, tau component already scaled by gamma
) -> torch.Tensor:
    """Cheetah's Poisson solve for a charge density given on the grid, (B, nx, ny, nz) in V.

    ``SpaceChargeKick`` only solves for a beam (``_solve_poisson_equation`` deposits the
    particles itself and takes gamma from the beam), so this runs its integrated Green
    function on a stand-in beam with gamma = 1 -- the cell size is already stretched.

    ponytail: the zero padding and FFT convolution below mirror
    ``SpaceChargeKick._solve_poisson_equation`` (cheetah 0.8.x) line by line; keep them
    in sync. ``tests/test_fields.py`` fails if they drift apart.
    """
    batch = charge_density.shape[0]
    grid_shape = tuple(charge_density.shape[-3:])
    nx, ny, nz = grid_shape
    factory_kwargs = {"device": cell_size.device, "dtype": cell_size.dtype}

    species = Species("electron", **factory_kwargs)
    unit_gamma_beam = ParticleBeam(
        particles=cell_size.new_zeros(batch, 1, 7),
        energy=species.mass_eV.expand(batch),  # gamma = energy / mass = 1
        species=species,
        **factory_kwargs,
    )
    solver = SpaceChargeKick(effect_length=torch.tensor(0.0), grid_shape=grid_shape)
    green = solver._integrated_green_function(unit_gamma_beam, cell_size)

    padded = charge_density.new_zeros(batch, 2 * nx, 2 * ny, 2 * nz)
    padded[:, :nx, :ny, :nz] = charge_density
    potential = torch.fft.irfftn(
        torch.fft.rfftn(padded, dim=[1, 2, 3]) * torch.fft.rfftn(green, dim=[1, 2, 3]),
        dim=[1, 2, 3],
    ).real / (4 * torch.pi * epsilon_0)
    return potential[:, :nx, :ny, :nz]


class ExactOperator(BaseModel, name="ExactOperator"):
    """Takes the same input tensor as the network and solves exactly.

    It is a neuralop ``BaseModel`` so that it can be passed wherever a trained network
    is expected, in particular to ``NeuralSpaceChargeKick``.

    Input (B, 4, nx, ny, nz): rho_hat, log(E_i/L)
    Output (B, 1, nx, ny, nz): phi_hat
    """

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        rho_hat = x[:, 0].double()
        grid = torch.tensor(rho_hat.shape[1:], dtype=torch.float64, device=x.device)
        aspect = x[:, 1:4, 0, 0, 0].double().exp()  # E_i / L
        cell = aspect / grid[None, :]  # cell size in units of L

        # The solver brings the factor 1/(4*pi*eps0); in hat units the equation reads
        # laplace(phi_hat) = -4*pi*rho_hat, so it is removed again.
        phi_hat = solve_poisson_on_grid(rho_hat, cell) * (4 * math.pi * epsilon_0)
        return phi_hat[:, None].to(x.dtype)
