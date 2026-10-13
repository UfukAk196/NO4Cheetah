"""G2 - Feld-Paar (rho_hat, phi_hat) aus Cheetah fuer eine Probe.

Baut zu einem Satz Strahlparameter einen ``ParticleBeam`` und ruft die
Methoden von ``cheetah.accelerator.SpaceChargeKick`` direkt auf (statt sie
zu kopieren), um auf dem Gitter zu loesen:

- ``_array_rho``             (Block 4+5: Deposition + Dichte, nullgepolstert auf 2x)
- ``_solve_poisson_equation``(Block 6+7: IGF-Faltung -> Potential, zurueckgeschnitten)

Die Gitterausdehnung (``grid_dimensions``) und ``cell_size`` stehen in
``SpaceChargeKick.track`` (Block 2) und werden hier identisch nachgebildet;
sie sind keine Methoden, sondern im ``track``-Koerper kodiert.

Entdimensionalisierung: exakt die Konvention aus ``utils/normalization.py``, die
auch ``NeuralSpaceChargeKick`` beim Einsatz im Lattice verwendet. Nur so
passt das trainierte Modell spaeter zu dem, was es im Element sieht.
Bezugsgroessen kommen aus der gamma-gestreckten Box (Ruhesystem):

    cell_s  = (dx, dy, gamma*dtau)            Zellgroesse, tau gestreckt
    L       = (Ex*Ey*Ez)^(1/3)                Ei = volle Boxkanten aus cell_s
    Q_s     = sum(rho * prod(cell_s))         = gamma * Ladung im Gitter
    rho_hat = rho * L^3 / Q_s                 (Mittelwert 1)
    phi_hat = phi * 4*pi*eps0 * L / Q_s

Damit gilt in Hut-Einheiten ``phi_hat = ExactOperator(rho_hat, aspect)``.
Alle Groessen sind die *ungepolsterten* Felder auf dem Originalgitter.
"""

from __future__ import annotations

import torch
from cheetah.accelerator import SpaceChargeKick
from cheetah.particles import ParticleBeam

from utils.normalization import nondimensionalize, potential_to_hat

# SpaceChargeKick-Default: Boxgroesse in Vielfachen von sigma je Achse.
DEFAULT_GRID_EXTENT = 3.0


def _build_beam(params: dict) -> ParticleBeam:
    """Erzeugt den ``ParticleBeam`` reproduzierbar aus ``params``.

    Formen/Einheiten der Eingaben (aus ``params``):
        sigma_x, sigma_y, sigma_tau [m] ; energy [eV] ; total_charge [C]
        n_particles [1] ; seed [1] (optional, default 0)

    Gibt einen *vektorisierten* Beam mit Batch-Dimension ``(1,)`` zurueck,
    damit die SpaceChargeKick-Methoden (``dim=[1,2,3]`` in der FFT) exakt wie
    im ``track``-Pfad arbeiten.

    Formen des Beams:
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


def _prepare(
    sck: SpaceChargeKick, beam: ParticleBeam
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Block 2+3 von ``SpaceChargeKick.track``: Gitterausdehnung, cell_size, xyz.

    Rueckgabe (alle mit Batch-Dimension (1,)):
        grid_dimensions (1, 3) [m]  Halbkanten = grid_extent * sigma
        cell_size       (1, 3) [m]  = 2 * grid_dimensions / grid_shape (ohne gamma)
        xp_coordinates  (1, N, 7)    = to_xyz_pxpypz()
    """
    grid_dimensions = torch.stack(
        [
            sck.grid_extent_x * beam.sigma_x,
            sck.grid_extent_y * beam.sigma_y,
            sck.grid_extent_tau * beam.sigma_tau,
        ],
        dim=-1,
    )  # (1, 3) m
    cell_size = (
        2
        * grid_dimensions
        / torch.tensor(
            sck.grid_shape, device=grid_dimensions.device, dtype=grid_dimensions.dtype
        )
    )  # (1, 3) m
    xp_coordinates = beam.to_xyz_pxpypz()  # (1, N, 7)
    return grid_dimensions, cell_size, xp_coordinates


def _compute_quantities(
    params: dict, sck: SpaceChargeKick
) -> tuple[ParticleBeam, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Baut Beam und leitet die fuer den Test noetigen Zwischengroessen ab.

    Ruft ``_build_beam`` und ``_prepare`` auf. Der Test braucht exakt diese
    Groessen, um die Referenz ``_solve_poisson_equation`` auf demselben Beam
    nachzuziehen - ohne die Beam-Konstruktion duplizieren zu muessen.

    Rueckgabe:
        beam, grid_dimensions, cell_size, xp_coordinates
    """
    beam = _build_beam(params)
    grid_dimensions, cell_size, xp_coordinates = _prepare(sck, beam)
    return beam, grid_dimensions, cell_size, xp_coordinates


def _as_grid_shape(grid_shape: int | tuple[int, int, int]) -> tuple[int, int, int]:
    if isinstance(grid_shape, int):
        return (grid_shape, grid_shape, grid_shape)
    return tuple(int(n) for n in grid_shape)


def make_space_charge_kick(grid_shape: int | tuple[int, int, int]) -> SpaceChargeKick:
    """Das ``SpaceChargeKick``-Element, dessen Methoden ``make_sample`` nutzt."""
    return SpaceChargeKick(
        effect_length=torch.tensor(0.0),
        grid_shape=_as_grid_shape(grid_shape),
        grid_extent_x=torch.tensor(DEFAULT_GRID_EXTENT),
        grid_extent_y=torch.tensor(DEFAULT_GRID_EXTENT),
        grid_extent_tau=torch.tensor(DEFAULT_GRID_EXTENT),
    )


def make_sample(params: dict, grid_shape: int | tuple[int, int, int]) -> dict:
    """Erzeugt das entdimensionalisierte Feld-Paar samt Skalierung fuer eine Probe.

    Parameters
    ----------
    params : dict
        Strahlparameter (Keys und Einheiten wie in ``_build_beam``),
        optional ``seed`` fuer reproduzierbare Beam-Erzeugung.
    grid_shape : int | (nx, ny, nz)
        Gitteraufloesung (``SpaceChargeKick.grid_shape``); ein int meint res^3.

    Returns
    -------
    dict mit
        rho_hat          (nx, ny, nz) float32   dimensionslose Dichte
        phi_hat          (nx, ny, nz) float32   dimensionsloses Potential
        total_charge     float  [C]  Q_s = gamma * Ladung im Gitter (Skalierung)
        reference_length float  [m]  L der gestreckten Box
        aspect           (3,) float64  Ei / L (gestreckt), Produkt = 1
        cell_size        (3,) float64  [m]  Zellgroesse OHNE gamma
        gamma            float  relativistisches gamma
    """
    nx, ny, nz = _as_grid_shape(grid_shape)
    sck = make_space_charge_kick((nx, ny, nz))

    beam, grid_dimensions, cell_size, xp_coordinates = _compute_quantities(params, sck)

    # Block 4+5: Deposition (cloud_in_cell) + Dichte, nullgepolstert auf 2x. Die
    # *ungepolsterte* Dichte steht im ersten Quadranten der Rueckgabe.
    rho_padded = sck._array_rho(beam, xp_coordinates, cell_size, grid_dimensions)
    rho = rho_padded[..., :nx, :ny, :nz]  # (1, nx, ny, nz) C/m^3

    # Block 6+7: Potential auf dem Originalgitter.
    phi = sck._solve_poisson_equation(beam, xp_coordinates, cell_size, grid_dimensions)

    gamma = beam.relativistic_gamma.reshape(1).double()  # (1,)
    stretched = cell_size.double().clone()
    stretched[:, 2] = stretched[:, 2] * gamma  # (1, 3) m, wie in SpaceChargeKick

    rho_hat, scaling = nondimensionalize(rho.double(), stretched)
    phi_hat = potential_to_hat(phi.double(), scaling)

    return {
        "rho_hat": rho_hat[0].float(),
        "phi_hat": phi_hat[0].float(),
        "total_charge": float(scaling.total_charge[0]),
        "reference_length": float(scaling.reference_length[0]),
        "aspect": scaling.aspect[0],
        "cell_size": cell_size[0].double(),
        "gamma": float(gamma[0]),
    }
