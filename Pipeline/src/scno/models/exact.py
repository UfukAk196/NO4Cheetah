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

from scno.utils.normalization import INPUT_CHANNELS


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

    The solver needs the cell size, which only the ``"normalized_extents"`` variant
    carries in the input (``log(E_i/L)``). For ``"gamma"`` and ``"only rho"`` the
    geometry is missing from ``x``, so it is passed out of band via ``cell_size``:

        ExactOperator()                  # "normalized_extents": geometry from x
        ExactOperator(cell_size=cell)    # "gamma" / "only rho": geometry from cell_size

    ``cell_size`` is in units of the reference length L, shape (B, 3) or (3,), and is
    broadcast over the batch. It is a plain attribute, not a buffer, so the operator
    stays a pure function of ``x`` when it is not needed.

    Input (B, C, nx, ny, nz): rho_hat plus the variant's geometry channels
    Output (B, 1, nx, ny, nz): phi_hat
    """

    def __init__(self, cell_size: torch.Tensor | None = None, **kwargs):
        super().__init__(**kwargs)
        self.cell_size = cell_size

    def _cell_size(self, x: torch.Tensor) -> torch.Tensor:
        """Cell size in units of L, (B, 3), from the input or from ``self.cell_size``."""
        grid = torch.tensor(x.shape[-3:], dtype=torch.float64, device=x.device)
        if x.shape[1] == INPUT_CHANNELS["normalized_extents"]:
            aspect = x[:, 1:4, 0, 0, 0].double().exp()  # E_i / L
            return aspect / grid[None, :]
        if self.cell_size is None:
            raise ValueError(
                f"ExactOperator got {x.shape[1]} input channels, so the cell size is "
                "not in x; pass it as ExactOperator(cell_size=...) (in units of L)."
            )
        # Already in units of L (E_i/L / grid), so no second division by the grid.
        cell = self.cell_size.to(device=x.device, dtype=torch.float64)
        return cell.expand(x.shape[0], 3)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        rho_hat = x[:, 0].double()
        cell = self._cell_size(x)

        # The solver brings the factor 1/(4*pi*eps0); in hat units the equation reads
        # laplace(phi_hat) = -4*pi*rho_hat, so it is removed again.
        phi_hat = solve_poisson_on_grid(rho_hat, cell) * (4 * math.pi * epsilon_0)
        return phi_hat[:, None].to(x.dtype)
