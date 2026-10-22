"""G2 - Field pair (rho, phi) from Cheetah for one sample.

Builds a ``ParticleBeam`` from a set of beam parameters and calls the methods of
``cheetah.accelerator.SpaceChargeKick`` directly (instead of copying them):

- ``_array_rho``              (blocks 4+5: deposition + density, zero padded to 2x)
- ``_solve_poisson_equation`` (blocks 6+7: IGF convolution -> potential, cropped)

Only the grid geometry (block 2) is not a method but written inline in
``SpaceChargeKick.track``; ``_compute_quantities`` mirrors those lines.

The *physical* fields are returned (rho in C/m^3, phi in V). Nondimensionalization
happens only on loading, with ``normalization.nondimensionalize_fields`` -- exactly
the convention ``NeuralSpaceChargeKick`` uses in the lattice. The reference
quantities come from the gamma-stretched box (rest frame):

    cell_s  = (dx, dy, gamma*dtau)            cell size, tau stretched
    L       = (Ex*Ey*Ez)^(1/3)                Ei = full box edges from cell_s
    Q_s     = sum(rho * prod(cell_s))         = gamma * charge on the grid
    rho_hat = rho * L^3 / Q_s                 (mean 1)
    phi_hat = phi * 4*pi*eps0 * L / Q_s

In hat units this gives ``phi_hat = ExactOperator(input_channels(rho_hat, E_i/L))``.
All quantities are the *unpadded* fields on the original grid.
"""

from __future__ import annotations

import torch
from cheetah.accelerator import SpaceChargeKick
from cheetah.particles import ParticleBeam


def _build_beam(params: dict) -> ParticleBeam:
    """Builds the ``ParticleBeam`` reproducibly from ``params``.

    Shapes/units of the inputs (from ``params``):
        sigma_x, sigma_y, sigma_tau [m] ; energy [eV] ; total_charge [C]
        n_particles [1] ; seed [1] (optional, default 0)

    Returns a *vectorized* beam with batch dimension ``(1,)``, so that the
    SpaceChargeKick methods (``dim=[1, 2, 3]`` in the FFT) work exactly as in the
    ``track`` path.

    Shapes of the beam:
        particles (1, n_particles, 7) ; energy (1,) ; sigma_* (1,)
    """
    torch.manual_seed(int(params.get("seed", 0)))
    return ParticleBeam.from_parameters(
        num_particles=int(params["n_particles"]),
        sigma_x=torch.tensor([float(params["sigma_x"])]),
        sigma_y=torch.tensor([float(params["sigma_y"])]),
        sigma_tau=torch.tensor([float(params["sigma_tau"])]),
        energy=torch.tensor([float(params["energy"])]),
        total_charge=torch.tensor([float(params["total_charge"])]),
    )


def _compute_quantities(
    params: dict, sck: SpaceChargeKick
) -> tuple[ParticleBeam, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Builds the beam and the intermediate quantities of ``SpaceChargeKick.track``.

    The tests need exactly these quantities to run the reference
    ``_solve_poisson_equation`` on the same beam, without duplicating the beam
    construction.

    Returns (all with batch dimension (1,)):
        beam
        grid_dimensions (1, 3) [m]  half edges = grid_extent * sigma
        cell_size       (1, 3) [m]  = 2 * grid_dimensions / grid_shape (without gamma)
        xp_coordinates  (1, N, 7)   = to_xyz_pxpypz()
    """
    beam = _build_beam(params)
    # ponytail: mirrors block 2 of `SpaceChargeKick.track` (cheetah 0.8.x), which has no
    # method of its own; `tests/test_fields.py::test_grid_geometry_matches_track` fails
    # if they drift apart.
    grid_dimensions = torch.stack(
        [
            sck.grid_extent_x * beam.sigma_x,
            sck.grid_extent_y * beam.sigma_y,
            sck.grid_extent_tau * beam.sigma_tau,
        ],
        dim=-1,
    )
    cell_size = (
        2
        * grid_dimensions
        / torch.tensor(
            sck.grid_shape, device=grid_dimensions.device, dtype=grid_dimensions.dtype
        )
    )
    return beam, grid_dimensions, cell_size, beam.to_xyz_pxpypz()


def _as_grid_shape(grid_shape: int | tuple[int, int, int]) -> tuple[int, int, int]:
    if isinstance(grid_shape, int):
        return (grid_shape, grid_shape, grid_shape)
    return tuple(int(n) for n in grid_shape)


def make_space_charge_kick(grid_shape: int | tuple[int, int, int]) -> SpaceChargeKick:
    """The ``SpaceChargeKick`` element whose methods ``make_sample`` uses.

    Grid extents are Cheetah's defaults (3 sigma per axis).
    """
    return SpaceChargeKick(
        effect_length=torch.tensor(0.0), grid_shape=_as_grid_shape(grid_shape)
    )


def make_sample(params: dict, grid_shape: int | tuple[int, int, int]) -> dict:
    """Generates the physical field pair (rho, phi) for one sample.

    Deliberately without nondimensionalization: that happens only on loading
    (``normalization.nondimensionalize_fields``), so that raw and scaled fields can
    always be compared.

    Parameters
    ----------
    params : dict
        Beam parameters (keys and units as in ``_build_beam``), optionally ``seed``
        for reproducible beam generation.
    grid_shape : int | (nx, ny, nz)
        Grid resolution (``SpaceChargeKick.grid_shape``); an int means res^3.

    Returns
    -------
    dict with
        rho        (nx, ny, nz) float32  [C/m^3]  charge density
        phi        (nx, ny, nz) float32  [V]      potential (modified Poisson equation)
        cell_size  (3,) float64  [m]  cell size WITHOUT gamma
        gamma      float  relativistic gamma
    """
    sck = make_space_charge_kick(grid_shape)
    beam, grid_dimensions, cell_size, xp_coordinates = _compute_quantities(params, sck)

    # Blocks 4+5: deposition (cloud in cell) + density, zero padded to 2x; the physical
    # density is the first octant, (1, nx, ny, nz) in C/m^3.
    nx, ny, nz = sck.grid_shape
    rho = sck._array_rho(beam, xp_coordinates, cell_size, grid_dimensions)[
        ..., :nx, :ny, :nz
    ]
    # Blocks 6+7: potential on the original grid, (1, nx, ny, nz) in V.
    phi = sck._solve_poisson_equation(beam, xp_coordinates, cell_size, grid_dimensions)

    return {
        "rho": rho[0].float(),
        "phi": phi[0].float(),
        "cell_size": cell_size[0].double(),
        "gamma": float(beam.relativistic_gamma.reshape(-1)[0]),
    }
