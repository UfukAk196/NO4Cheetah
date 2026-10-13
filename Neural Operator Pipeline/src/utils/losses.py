"""
Verlustfunktionen.

Wichtigster Punkt fuer dieses Projekt: der Kick haengt nicht vom Potential ab,
sondern von dessen Gradienten. Ein Modell mit winzigem L2-Fehler auf phi kann
trotzdem miserable Kraefte liefern, weil Ableiten hohe Frequenzen verstaerkt.
Deshalb steht hier eine H1-artige Variante bereit, die den Gradientenfehler
mitbestraft -- und in der Auswertung wird der Kraftfehler getrennt berichtet.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def relative_l2(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Mittlerer relativer L2-Fehler pro Probe -- die uebliche Groesse in
    Neural-Operator-Papern, damit Zahlen ueber Datensaetze hinweg vergleichbar sind."""
    p = pred.flatten(1)
    t = target.flatten(1)
    return ((p - t).norm(dim=1) / (t.norm(dim=1) + eps)).mean()


def central_gradient(field: torch.Tensor) -> torch.Tensor:
    """Zentrale Differenzen auf (B, 1, nx, ny, nz), Gitterabstand 1.

    Der physikalische Abstand faellt beim relativen Fehler ohnehin raus, und in
    entdimensionalisierten Koordinaten ist er pro Achse konstant.
    """
    f = field
    gx = torch.zeros_like(f)
    gy = torch.zeros_like(f)
    gz = torch.zeros_like(f)
    gx[..., 1:-1, :, :] = 0.5 * (f[..., 2:, :, :] - f[..., :-2, :, :])
    gy[..., :, 1:-1, :] = 0.5 * (f[..., :, 2:, :] - f[..., :, :-2, :])
    gz[..., :, :, 1:-1] = 0.5 * (f[..., :, :, 2:] - f[..., :, :, :-2])
    return torch.cat([gx, gy, gz], dim=1)


def h1_loss(
    pred: torch.Tensor, target: torch.Tensor, gradient_weight: float = 1.0
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """L2 auf dem Feld plus gewichteter L2 auf dem Gradienten."""
    field = relative_l2(pred, target)
    grad = relative_l2(central_gradient(pred), central_gradient(target))
    total = field + gradient_weight * grad
    return total, {"loss_field": field.detach(), "loss_grad": grad.detach()}


def poisson_residual(
    potential_hat: torch.Tensor, rho_hat: torch.Tensor, aspect: torch.Tensor
) -> torch.Tensor:
    """Residuum von laplace(phi_hat) = -4*pi*rho_hat.

    In entdimensionalisierter Form (Herleitung in normalization.py) faellt
    jede Materialkonstante raus, uebrig bleibt der Faktor 4*pi. Die
    Gitterabstaende in diesen Einheiten sind h_i = E_i / (L * N_i), also
    aspect_i / N_i. Nur Innenpunkte -- am offenen Rand gilt der
    Differenzenstern nicht.

    Das ist der optionale PiNN-Anteil: er zieht die Loesung dorthin, wo die
    Physik sie haben will, auch an Stellen ohne Trainingsdaten. Als Zusatzterm
    mit kleinem Gewicht gedacht, nicht als Ersatz fuer Daten.

    Ehrliche Warnung: die Trainingsdaten kommen aus einem Greenschen-Funktions-
    Solver, dieses Residuum misst gegen den 7-Punkt-Differenzenstern. Die
    beiden Diskretisierungen stimmen nicht exakt ueberein, das Residuum geht
    also nie ganz auf null. Der verbleibende Sockel ist Diskretisierungsfehler,
    kein Modellfehler -- miss ihn einmal mit der exakten Loesung als Eingabe
    und berichte ihn als Referenzwert.
    """
    p = potential_hat
    n = torch.tensor(p.shape[2:], device=p.device, dtype=p.dtype)
    h = aspect / n[None, :]  # (B, 3) Gitterabstand in Einheiten von L
    hx, hy, hz = (h[:, i].view(-1, 1, 1, 1, 1) for i in range(3))

    core = p[:, :, 1:-1, 1:-1, 1:-1]
    d2x = p[:, :, 2:, 1:-1, 1:-1] - 2 * core + p[:, :, :-2, 1:-1, 1:-1]
    d2y = p[:, :, 1:-1, 2:, 1:-1] - 2 * core + p[:, :, 1:-1, :-2, 1:-1]
    d2z = p[:, :, 1:-1, 1:-1, 2:] - 2 * core + p[:, :, 1:-1, 1:-1, :-2]

    lap = d2x / hx.square() + d2y / hy.square() + d2z / hz.square()
    source = rho_hat[:, :, 1:-1, 1:-1, 1:-1]
    return (lap + 4 * torch.pi * source).abs().mean()


def build_loss(cfg: dict):
    """Gibt eine Funktion (pred, batch_dict) -> (loss, metriken) zurueck."""
    kind = cfg.get("kind", "h1")
    gw = float(cfg.get("gradient_weight", 1.0))
    pw = float(cfg.get("physics_weight", 0.0))

    def fn(pred, target, rho_hat=None, aspect=None):
        if kind == "l2":
            loss = relative_l2(pred, target)
            metrics = {"loss_field": loss.detach()}
        elif kind == "h1":
            loss, metrics = h1_loss(pred, target, gw)
        else:
            raise ValueError(f"Unbekannter Verlust: {kind}")

        if pw > 0.0 and rho_hat is not None and aspect is not None:
            res = poisson_residual(pred, rho_hat, aspect)
            loss = loss + pw * res
            metrics["loss_physics"] = res.detach()
        return loss, metrics

    return fn
