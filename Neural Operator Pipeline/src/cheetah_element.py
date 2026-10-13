"""
Der Zielpunkt der ganzen Arbeit: ein Cheetah-Element, das den FFT-Solver
durch den trainierten Neural Operator ersetzt.

Struktur ist absichtlich eins zu eins an `SpaceChargeKick` angelehnt --
Ladungsdeposition, Potential, Gradient, Interpolation auf die Teilchen. Nur
Schritt 2 wechselt den Besitzer: statt FFT-Faltung fragt das Element das Netz.

Alles bleibt differenzierbar, weil nichts anderes passiert als Torch-Ops.
Damit ueberlebt der Gradient von Strahlparametern bis zum Ende der Strecke,
und genau darauf zielt das Vorhaben.

Braucht Cheetah. Ohne installiertes Cheetah bleibt der Rest des Projekts
lauffaehig, dieses Modul importiert dann eben nicht.
"""

from __future__ import annotations

import torch

from utils.normalization import build_input, nondimensionalize, potential_from_hat
from utils.physics import cloud_in_cell, potential_gradient

try:
    from cheetah.accelerator.element import Element
    from cheetah.particles import ParticleBeam

    CHEETAH_AVAILABLE = True
except ImportError:  # pragma: no cover
    Element = object  # type: ignore
    ParticleBeam = object  # type: ignore
    CHEETAH_AVAILABLE = False

ELEMENTARY_CHARGE = 1.602176634e-19
SPEED_OF_LIGHT = 299792458.0


class NeuralSpaceChargeKick(Element if CHEETAH_AVAILABLE else torch.nn.Module):
    """Drop-in-Ersatz fuer `SpaceChargeKick` mit gelerntem Poisson-Solver.

    :param operator: trainiertes Netz, Eingabe (B, 7, nx, ny, nz), Ausgabe (B, 1, ...)
    :param effect_length: Laenge, ueber die der Kick wirkt, in Metern
    :param grid_shape: Gitteraufloesung -- darf von der Trainingsaufloesung
        abweichen, das ist ja gerade der Witz an Neural Operators
    """

    def __init__(
        self,
        operator: torch.nn.Module,
        effect_length: torch.Tensor,
        grid_shape: tuple[int, int, int] = (32, 32, 32),
        grid_extent_sigma: float = 3.0,
        name: str | None = None,
        **kwargs,
    ):
        if CHEETAH_AVAILABLE:
            super().__init__(name=name, **kwargs)
            self.register_buffer_or_parameter("effect_length", effect_length)
        else:
            super().__init__()
            self.register_buffer("effect_length", effect_length)

        self.operator = operator
        self.grid_shape = grid_shape
        self.grid_extent_sigma = grid_extent_sigma

    # ------------------------------------------------------------------ #
    def predict_potential(
        self,
        charge_density: torch.Tensor,  # (B, nx, ny, nz)
        cell_size: torch.Tensor,  # (B, 3), tau bereits mit gamma skaliert
    ) -> torch.Tensor:
        """Der eigentliche Austausch: Netz statt FFT."""
        rho_hat, scaling = nondimensionalize(charge_density, cell_size)
        x = build_input(rho_hat.float(), scaling)
        phi_hat = self.operator(x)[:, 0]
        return potential_from_hat(phi_hat.to(charge_density.dtype), scaling)

    def field_on_grid(
        self, charge_density: torch.Tensor, cell_size: torch.Tensor, gamma: torch.Tensor
    ) -> torch.Tensor:
        scaled = cell_size.clone()
        scaled[:, 2] = scaled[:, 2] * gamma
        phi = self.predict_potential(charge_density, scaled)
        igamma2 = torch.where(gamma != 0, gamma.square().reciprocal(), torch.zeros_like(gamma))
        return -igamma2[:, None, None, None, None] * potential_gradient(phi, cell_size)

    # ------------------------------------------------------------------ #
    def track(self, incoming):
        """Vereinfachte Variante ohne die Vektor-Dimensionen von Cheetah.

        TODO fuer die Thesis: die Batch-/Broadcast-Logik aus dem Original
        uebernehmen, damit vektorisierte Lattices durchlaufen. Fuer erste
        Tests mit einem einzelnen Strahl genuegt das hier.
        """
        if not CHEETAH_AVAILABLE:
            raise RuntimeError("Cheetah fehlt -- `pip install cheetah-accelerator`")

        xp = incoming.to_xyz_pxpypz()
        if xp.dim() == 2:
            xp = xp[None]
        positions = xp[..., [0, 2, 4]]

        half_extent = self.grid_extent_sigma * positions.std(dim=1)
        cell_size = 2.0 * half_extent / torch.tensor(
            self.grid_shape, device=xp.device, dtype=xp.dtype
        )
        charges = incoming.particle_charges.reshape(positions.shape[0], -1)
        charge_grid = cloud_in_cell(positions, charges, self.grid_shape, half_extent)
        charge_density = charge_grid / cell_size.prod(dim=1)[:, None, None, None]

        gamma = incoming.relativistic_gamma.reshape(-1)
        field = self.field_on_grid(charge_density, cell_size, gamma)
        forces = _interpolate_to_particles(field, positions, half_extent, cell_size)

        dt = self.effect_length / (SPEED_OF_LIGHT * incoming.relativistic_beta)
        xp = xp.clone()
        for axis, comp in enumerate((1, 3, 5)):
            xp[..., comp] = xp[..., comp] + ELEMENTARY_CHARGE * forces[..., axis] * dt

        return ParticleBeam.from_xyz_pxpypz(
            xp_coordinates=xp.reshape(*incoming.particles.shape[:-1], 7),
            energy=incoming.energy,
            particle_charges=incoming.particle_charges,
            survival_probabilities=incoming.survival_probabilities,
            s=incoming.s,
            species=incoming.species,
        )

    @property
    def is_skippable(self) -> bool:
        return False

    @property
    def defining_features(self) -> list[str]:
        base = super().defining_features if CHEETAH_AVAILABLE else []
        return base + ["effect_length", "grid_shape", "grid_extent_sigma"]


def _interpolate_to_particles(
    field: torch.Tensor,  # (B, 3, nx, ny, nz)
    positions: torch.Tensor,  # (B, N, 3)
    half_extent: torch.Tensor,  # (B, 3)
    cell_size: torch.Tensor,  # (B, 3)
) -> torch.Tensor:
    """Trilineare Interpolation via grid_sample -- differenzierbar und schnell.

    Achtung bei den Achsen: grid_sample erwartet die Koordinaten in der
    Reihenfolge (x, y, z) = (letzte, mittlere, erste) Tensorachse. Genau hier
    baut man sich die haesslichsten stillen Fehler ein, also einmal gegen die
    Schleifenvariante im Original gegentesten.
    """
    normalized = positions / half_extent[:, None, :]  # -> [-1, 1]
    grid = normalized[..., [2, 1, 0]][:, :, None, None, :]  # (B, N, 1, 1, 3)
    sampled = torch.nn.functional.grid_sample(
        field, grid, mode="bilinear", padding_mode="zeros", align_corners=False
    )  # (B, 3, N, 1, 1)
    return sampled[..., 0, 0].transpose(1, 2)  # (B, N, 3)
