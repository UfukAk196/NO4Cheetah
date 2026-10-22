"""
Loss functions, built on neuraloperator's ``LpLoss`` and ``FiniteDiff``.

The most important point for this project: the kick does not depend on the potential
but on its gradient. A model with a tiny L2 error on phi can still produce miserable
forces, because differentiation amplifies high frequencies. Hence the Sobolev loss
here also penalizes the gradient error with its own weight (neuralop's ``H1Loss``
has no such weight) -- and the evaluation reports the force error separately.

All losses are summed over the batch, as the neuralop ``Trainer`` expects (it divides
the accumulated evaluation losses by the number of samples).
"""

from __future__ import annotations

import torch
from neuralop.losses import LpLoss
from neuralop.losses.differentiation import FiniteDiff

# Open boundaries: one-sided differences at the edges instead of periodic wrapping.
# Grid spacing 1: the physical spacing drops out of the relative error, and in
# nondimensionalized coordinates it is constant per axis.
_FINITE_DIFF = FiniteDiff(
    dim=3, h=1.0, periodic_in_x=False, periodic_in_y=False, periodic_in_z=False
)
_FIELD_L2 = LpLoss(d=3, p=2, reduction="sum")  # per sample over (nx, ny, nz)
_GRADIENT_L2 = LpLoss(d=4, p=2, reduction="sum")  # per sample over (3, nx, ny, nz)


def field_error(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Relative L2 error of (B, 1, nx, ny, nz) fields, summed over the batch -- the
    usual quantity in neural operator papers, so that numbers are comparable across
    datasets."""
    return _FIELD_L2.rel(pred, target)


def gradient_error(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Relative L2 error of the gradients of (B, 1, nx, ny, nz) fields, over all three
    components together and summed over the batch -- the error of the forces."""
    return _GRADIENT_L2.rel(_FINITE_DIFF.gradient(pred), _FINITE_DIFF.gradient(target))


def poisson_residual(
    potential_hat: torch.Tensor,  # (B, 1, nx, ny, nz)
    rho_hat: torch.Tensor,  # (B, 1, nx, ny, nz)
    normalized_extent: torch.Tensor,  # (B, 3)
) -> torch.Tensor:
    """Residual of laplace(phi_hat) = -4*pi*rho_hat, mean over the interior points of
    each sample, summed over the batch.

    In nondimensionalized form (derivation in utils/normalization.py) every material
    constant drops out, only the factor 4*pi remains. The grid spacings in these units
    are h_i = E_i / (L * N_i), i.e. normalized_extent_i / N_i. Interior points only --
    the difference stencil does not hold at the open boundary.

    This is the optional PINN part: it pulls the solution towards where the physics
    wants it, also at places without training data. Meant as an additional term with a
    small weight, not as a replacement for data.

    Honest warning: the training data come from a Green's function solver, while this
    residual measures against the 7-point difference stencil. The two discretizations
    do not agree exactly, so the residual never reaches zero. The remaining floor is
    discretization error, not model error -- measure it once with the exact solution
    as input and report it as a reference value.
    """
    n = torch.tensor(potential_hat.shape[-3:], device=potential_hat.device)
    h = normalized_extent.to(potential_hat.dtype) / n[None, :]  # (B, 3) in units of L
    second_differences = (
        _FINITE_DIFF.dx(potential_hat, order=2),
        _FINITE_DIFF.dy(potential_hat, order=2),
        _FINITE_DIFF.dz(potential_hat, order=2),
    )
    laplacian = sum(
        d2 / h[:, i].square().view(-1, 1, 1, 1, 1)
        for i, d2 in enumerate(second_differences)
    )
    residual = laplacian + 4 * torch.pi * rho_hat
    return residual[..., 1:-1, 1:-1, 1:-1].abs().flatten(1).mean(dim=1).sum()


class SobolevLoss:
    """Relative L2 of the field plus weighted relative L2 of its gradient.

    Callable as a neuralop ``Trainer`` loss: ``loss(y_pred, y, x=x)``. With
    ``gradient_weight=0`` this is the plain relative L2 loss.

    :param gradient_weight: Weight of the gradient error -- the kick is a gradient.
    :param physics_weight: Weight of the Poisson residual (PINN part). Requires
        nondimensionalized data without output normalizer, since rho_hat and
        log(E_i/L) are read from the model input ``x``.
    """

    def __init__(self, gradient_weight: float = 1.0, physics_weight: float = 0.0):
        self.gradient_weight = float(gradient_weight)
        self.physics_weight = float(physics_weight)

    def __call__(
        self, y_pred: torch.Tensor, y: torch.Tensor, x: torch.Tensor = None, **kwargs
    ) -> torch.Tensor:
        loss = field_error(y_pred, y)
        if self.gradient_weight > 0.0:
            loss = loss + self.gradient_weight * gradient_error(y_pred, y)
        if self.physics_weight > 0.0:
            normalized_extent = x[:, 1:4, 0, 0, 0].exp()
            loss = loss + self.physics_weight * poisson_residual(
                y_pred, x[:, :1], normalized_extent
            )
        return loss
