"""
Entdimensionalisierung -- das wichtigste Stueck der ganzen Pipeline.

Warum ueberhaupt? Rohe Ladungsdichten liegen bei 1e-3 C/m^3, Potentiale bei
1e5 V, Zellgroessen bei 1e-5 m. Ein Netz, das solche Zahlen direkt frisst,
lernt vor allem Groessenordnungen statt Physik. Also raeumen wir das vorher weg.

Die Poisson-Gleichung schenkt uns dafuer zwei Eigenschaften:

  Linearitaet  phi ist linear in rho  -> Gesamtladung Q rausziehen
  Skalierung   Laenge x L skaliert phi mit L^2 (bei fester Gesamtladung: 1/L)

Konkret, mit Q als Gesamtladung im Gitter und L als Referenzlaenge:

    rho_hat = rho * L^3 / Q            (dimensionslose Dichte, Integral = 1)
    phi_hat = phi * 4*pi*eps0*L / Q    (dimensionslos, Groessenordnung 1)

Damit lautet die Gleichung schlicht:  laplace(phi_hat) = -4*pi*rho_hat.

Die Wahl von L ist der Punkt, an dem man sich still ein Bein stellen kann.
Naheliegend waere die Zellgroesse -- und genau das ist falsch. Ein
Neural Operator soll eine Abbildung zwischen *Funktionen* lernen, nicht
zwischen Gittern. Haengt L an der Zellgroesse, dann bildet dieselbe
physikalische Dichte je nach Aufloesung auf ein anderes phi_hat ab, und die
Zero-Shot-Super-Resolution faellt auseinander, obwohl am Netz nichts falsch
ist. Also: L = geometrisches Mittel der drei *Boxkanten*. Das haengt nur an
der Geometrie, nicht an der Diskretisierung.

Das geometrische Mittel statt einer einzelnen Kante ist ebenfalls Absicht:
durch die gamma-Streckung der longitudinalen Achse laufen die
Achsenverhaeltnisse ueber mehrere Groessenordnungen auseinander, und ein
symmetrischer Bezug drueckt die Streuung von phi_hat spuerbar zusammen
(nachmessen mit scripts/inspect_scaling.py).

Uebrig bleiben als echte Parameter die Achsenverhaeltnisse log(E_i / L). Die
drei Werte summieren sich per Konstruktion zu null und gehen als konstante
Extra-Kanaele ins Netz. Der Operator, den das Netz lernt, ist damit
universell: ein Modell deckt alle Strahlgroessen, Ladungen und Energien ab,
weil die Skalierung analytisch abgezogen wurde statt aus Daten gelernt.

Kontrolliere das im Experiment: trainiere einmal ohne Entdimensionalisierung
gegen. Der Unterschied im Fehler ist typischerweise eine bis zwei
Groessenordnungen -- ein sauberes Ablationsergebnis fuer die Thesis.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .physics import EPSILON_0


@dataclass
class Scaling:
    """Merkt sich die Faktoren eines Batches, damit man zurueckrechnen kann."""

    total_charge: torch.Tensor  # (B,) in Coulomb
    reference_length: torch.Tensor  # (B,) in Meter: (Ex*Ey*Ez)^(1/3), Boxkanten
    aspect: torch.Tensor  # (B, 3) -> E_i / L, Produkt = 1, aufloesungsunabhaengig

    def to(self, device):
        return Scaling(
            self.total_charge.to(device),
            self.reference_length.to(device),
            self.aspect.to(device),
        )


def nondimensionalize(
    charge_density: torch.Tensor,  # (B, nx, ny, nz) in C/m^3
    cell_size: torch.Tensor,  # (B, 3) in m
    eps: float = 1e-30,
) -> tuple[torch.Tensor, Scaling]:
    grid_shape = torch.tensor(
        charge_density.shape[1:], device=cell_size.device, dtype=cell_size.dtype
    )
    cell_volume = cell_size.prod(dim=1)  # (B,)
    charge_per_cell = charge_density * cell_volume[:, None, None, None]
    total = charge_per_cell.flatten(1).sum(dim=1)  # (B,)
    total = torch.where(total.abs() < eps, torch.full_like(total, eps), total)

    extent = cell_size * grid_shape[None, :]  # (B, 3) Boxkanten
    ref = extent.prod(dim=1).pow(1.0 / 3.0)  # geometrisches Mittel der Boxkanten
    aspect = extent / ref[:, None]  # Produkt der drei Werte ist 1

    # Als Dichte, nicht als Ladungsanteil: das ist die Groesse, die beim
    # Verfeinern des Gitters gegen eine feste Funktion konvergiert.
    volume_ratio = ref.pow(3) / cell_volume  # = Anzahl Zellen
    rho_hat = charge_per_cell / total[:, None, None, None] * volume_ratio[:, None, None, None]
    return rho_hat, Scaling(total, ref, aspect)


def potential_to_hat(potential: torch.Tensor, scaling: Scaling) -> torch.Tensor:
    factor = 4 * math.pi * EPSILON_0 * scaling.reference_length / scaling.total_charge
    return potential * factor[:, None, None, None]


def potential_from_hat(potential_hat: torch.Tensor, scaling: Scaling) -> torch.Tensor:
    factor = scaling.total_charge / (4 * math.pi * EPSILON_0 * scaling.reference_length)
    return potential_hat * factor[:, None, None, None]


def coordinate_channels(
    grid_shape: tuple[int, int, int], device=None, dtype=torch.float32
) -> torch.Tensor:
    """Drei Gitterkoordinaten in [-1, 1], Form (3, nx, ny, nz).

    FNOs brauchen das als Positionsinformation -- ohne sie ist der erste Lift
    translationsblind und das Modell tut sich mit den offenen Raendern schwer.
    """
    axes = [
        torch.linspace(-1.0, 1.0, n, device=device, dtype=dtype) for n in grid_shape
    ]
    return torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=0)


def build_input(rho_hat: torch.Tensor, scaling: Scaling) -> torch.Tensor:
    """Setzt den Eingabetensor fuer das Netz zusammen: (B, 7, nx, ny, nz).

    Kanaele: rho_hat, x, y, z, log(Ex/L), log(Ey/L), log(Ez/L)
    """
    batch = rho_hat.shape[0]
    grid_shape = tuple(rho_hat.shape[1:])
    coords = coordinate_channels(grid_shape, rho_hat.device, rho_hat.dtype)
    coords = coords[None].expand(batch, -1, -1, -1, -1)

    log_aspect = scaling.aspect.log()[:, :, None, None, None].expand(
        -1, -1, *grid_shape
    )
    return torch.cat([rho_hat[:, None], coords, log_aspect], dim=1)
