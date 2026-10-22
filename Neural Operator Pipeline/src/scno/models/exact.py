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
from neuralop.models.base_model import BaseModel
from scipy.constants import epsilon_0


class ExactOperator(BaseModel, name="ExactOperator"):
    """Takes the same input tensor as the network and solves exactly.

    It is a neuralop ``BaseModel`` so that it can be passed wherever a trained network
    is expected, in particular to ``NeuralSpaceChargeKick``.

    Input (B, 4, nx, ny, nz): rho_hat, log(E_i/L)
    Output (B, 1, nx, ny, nz): phi_hat
    """

    def __init__(self):
        super().__init__()
        # Only its grid-level solve is used, which depends on no element parameter.
        self.solver = SpaceChargeKick(effect_length=torch.tensor(0.0))

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        rho_hat = x[:, 0].double()
        grid = torch.tensor(rho_hat.shape[1:], dtype=torch.float64, device=x.device)
        aspect = x[:, 1:4, 0, 0, 0].double().exp()  # E_i / L
        cell = aspect / grid[None, :]  # cell size in units of L

        # The solver brings the factor 1/(4*pi*eps0); in hat units the equation reads
        # laplace(phi_hat) = -4*pi*rho_hat, so it is removed again.
        phi_hat = self.solver._solve_poisson_on_grid(rho_hat, cell) * (
            4 * math.pi * epsilon_0
        )
        return phi_hat[:, None].to(x.dtype)
