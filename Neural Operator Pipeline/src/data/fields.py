"""G2 - Feld-Paar (rho_hat, phi_hat, E_hat) aus Cheetah fuer eine Probe.

Baut zu einem Satz Strahlparameter einen ``ParticleBeam`` und ruft die
Methoden von ``cheetah.accelerator.SpaceChargeKick`` direkt auf (statt sie
zu kopieren), um auf dem Gitter zu loesen:

- ``_array_rho``             (Block 4+5: Deposition + Dichte, nullgepolstert auf 2x)
- ``_solve_poisson_equation``(Block 6+7: IGF-Faltung -> Potential, zurueckgeschnitten)
- ``_E_plus_vB_field``       (Block 8: Feldgradienten inkl. -1/gamma^2)

Die Gitterausdehnung (``grid_dimensions``) und ``cell_size`` stehen in
``SpaceChargeKick.track`` (Block 2) und werden hier identisch nachgebildet;
sie sind keine Methoden, sondern im ``track``-Koerper kodiert.

Entdimensionalisierung (Plan, "L aus der Boxgroesse, nicht der Zellgroesse"):
    rho_hat = rho * L^3 / Q          mit L^3 = gx*gy*gz   (Box-Halbkanten-Wuerfel)
    phi_hat = phi * 4*pi*eps0 * L / Q      , L = (gx*gy*gz)^(1/3)
    E_hat   = E   * 4*pi*eps0 * L^2 / Q
    rho [C/m^3], phi [V], E [V/m], Q [C], gx,gy,gz [m], L [m].
    rho ist ein einziges Skalarfeld (Ladungsdichte ueber alle 3 Dimensionen),
    deshalb ist das Volumen intrinsisch 3D -> L^3 = gx*gy*gz (nicht vektoriell).
    E_hat ist der Kanal inkl. der -1/gamma^2-Skalierung aus _E_plus_vB_field.

Alle drei Formen geben die *ungepolsterten* Groessen auf dem Originalgitter
res^3 zurueck (FNO-Ein-/Ausgabe auf demselben Gitter).
"""

from __future__ import annotations

import math

import torch
from cheetah.accelerator import SpaceChargeKick
from cheetah.particles import ParticleBeam

EPSILON_0 = 8.8541878128e-12  # F/m (= C/(V m))

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
        cell_size       (1, 3) [m]  = 2 * grid_dimensions / res (ohne gamma)
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


def _nondimensionalize(
    rho: torch.Tensor,
    phi: torch.Tensor,
    e_fields: torch.Tensor,
    grid_dimensions: torch.Tensor,
    total_charge: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Wendet die Plan-Entdimensionalisierung auf Gittergroessen an.

    Eingaben (Batch-Dim (1,)):
        rho   (1, res, res, res) [C/m^3]      ungepolstert
        phi   (1, res, res, res) [V]
        e_fields (1, 3, res, res, res) [V/m]  (gx, gy, gtau gestapelt)
        grid_dimensions (1, 3) [m]            Halbkanten gx,gy,gz
        total_charge [C]                       Gesamtladung Q

    Rueckgabe (alle dimensionslos, Batch-Dim (1,)):
        rho_hat (1, 1, res, res, res)
        phi_hat (1, 1, res, res, res)
        e_hat   (1, 3, res, res, res)
    """
    dtype = rho.dtype
    gx, gy, gz = (
        grid_dimensions[..., 0],
        grid_dimensions[..., 1],
        grid_dimensions[..., 2],
    )  # (1,)
    # L^3 = gx*gy*gz (Box-Halbkanten-Wuerfel); L = (L^3)^(1/3)
    box_volume = gx * gy * gz  # (1,) m^3
    L = box_volume ** (1.0 / 3.0)  # (1,) m (geometrisches Mittel der Halbkanten)

    Q = torch.tensor(float(total_charge), dtype=dtype)
    coulomb = torch.tensor(4 * math.pi * EPSILON_0, dtype=dtype)  # F/m

    rho_hat = (rho * box_volume / Q).unsqueeze(1)  # (1, 1, res, res, res)
    phi_hat = (phi * coulomb * L / Q).unsqueeze(1)  # (1, 1, res, res, res)
    e_hat = e_fields * coulomb * L.square() / Q  # (1, 3, res, res, res)

    return rho_hat, phi_hat, e_hat


def make_sample(params: dict, res: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Erzeugt das entdimensionalisierte Feld-Paar (x, y) fuer eine Probe.

    Parameters
    ----------
    params : dict
        Strahlparameter (Keys und Einheiten wie in ``_build_beam``),
        optional ``seed`` fuer reproduzierbare Beam-Erzeugung.
    res : int
        Gitteraufloesung je Achse (``SpaceChargeKick.grid_shape=(res,res,res)``).

    Returns
    -------
    x : torch.Tensor  (4, res, res, res) float32
        Kanal 0 = rho_hat (dimensionslos), Kanaele 1..3 = E_hat (dimensionslos,
        inkl. -1/gamma^2-Skalierung aus _E_plus_vB_field).
    y : torch.Tensor  (1, res, res, res) float32
        Kanal 0 = phi_hat (dimensionslos).
    """
    sck = SpaceChargeKick(
        effect_length=torch.tensor(0.0),
        grid_shape=(res, res, res),
        grid_extent_x=torch.tensor(DEFAULT_GRID_EXTENT),
        grid_extent_y=torch.tensor(DEFAULT_GRID_EXTENT),
        grid_extent_tau=torch.tensor(DEFAULT_GRID_EXTENT),
    )

    beam, grid_dimensions, cell_size, xp_coordinates = _compute_quantities(params, sck)

    # Block 4+5: Deposition (cloud_in_cell) + Dichte, nullgepolstert auf 2x. Die
    # *ungepolsterte* Dichte steht im ersten Quadranten der Rueckgabe.
    rho_padded = sck._array_rho(beam, xp_coordinates, cell_size, grid_dimensions)
    rho = rho_padded[..., :res, :res, :res]  # (1, res, res, res) C/m^3

    # Block 6+7: Potential auf dem Originalgitter res^3.
    phi = sck._solve_poisson_equation(beam, xp_coordinates, cell_size, grid_dimensions)

    # Block 8: Feldgradienten (E+vB, inkl. -1/gamma^2), je (1, res, res, res).
    grad_x, grad_y, grad_tau = sck._E_plus_vB_field(
        beam, xp_coordinates, cell_size, grid_dimensions
    )
    e_fields = torch.stack([grad_x, grad_y, grad_tau], dim=1)  # (1, 3, res, res, res)

    rho_hat, phi_hat, e_hat = _nondimensionalize(
        rho, phi, e_fields, grid_dimensions, float(params["total_charge"])
    )

    x = torch.cat([rho_hat, e_hat], dim=1).float()  # (1, 4, res, res, res)
    x = rho_hat.float()  # (1, 1, res, res, res)
    y = phi_hat.float()  # (1, 1, res, res, res)

    return x.squeeze(0), y.squeeze(0)  # (4, res, res, res), (1, res, res, res)
