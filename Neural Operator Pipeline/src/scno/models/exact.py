"""
Der exakte Solver, verpackt als Modell.

Klingt nach Spielerei, ist aber das wertvollste Debugging-Werkzeug im ganzen
Projekt: dieses "Modell" muss durch jede Auswertung mit Fehler null und
Gradienten-Kosinus 1.0 laufen. Tut es das nicht, steckt der Fehler in der
Normierung, in der Achsenreihenfolge oder in der Auswertung -- und nicht im
Netz. Erst wenn diese Kontrolle sauber ist, sagen die Zahlen des trainierten
Modells ueberhaupt etwas aus.

Fuer die Thesis ist das ausserdem die obere Schranke: alles, was ein
Surrogat erreichen kann, steht hier.
"""

from __future__ import annotations

import math

import torch

from scno.utils.physics import EPSILON_0, solve_poisson


class ExactOperator(torch.nn.Module):
    """Nimmt denselben Eingabetensor wie das Netz und rechnet exakt.

    Eingabe (B, 7, nx, ny, nz): rho_hat, x, y, z, log(E_i/L)
    Ausgabe (B, 1, nx, ny, nz): phi_hat
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rho_hat = x[:, 0].double()
        grid = torch.tensor(rho_hat.shape[1:], dtype=torch.float64, device=x.device)
        aspect = x[:, 4:7, 0, 0, 0].double().exp()  # E_i / L
        cell = aspect / grid[None, :]  # Zellgroesse in Einheiten von L

        # solve_poisson bringt den Faktor 1/(4*pi*eps0) mit; in Hut-Einheiten
        # lautet die Gleichung laplace(phi_hat) = -4*pi*rho_hat, also wieder raus damit.
        phi_hat = solve_poisson(rho_hat, cell) * (4 * math.pi * EPSILON_0)
        return phi_hat[:, None].to(x.dtype)
