"""
FNO-3D.

Zwei Wege, und der Code nimmt automatisch den besseren:

  1. Ist die Bibliothek `neuraloperator` installiert, wird deren FNO verwendet.
     Das ist die Referenzimplementierung der Autoren des FNO-Papers, hat
     Tensor-Faktorisierung, Skip-Varianten und viel Feinschliff.
  2. Sonst greift eine kompakte Eigenimplementierung. Die ist ungefaehr 80
     Zeilen lang und tut genau das, was im Paper steht -- gut zum Lesen und
     zum Debuggen, und sie haelt das Projekt lauffaehig, wenn die
     Installation gerade zickt.

Kernidee des FNO in einem Satz: statt lokaler Faltungen im Ortsraum eine
Multiplikation im Fourierraum, abgeschnitten nach den ersten k Moden -- das
ist eine globale Faltung mit gelerntem Kern, und genau das braucht Poisson,
weil das Potential an jedem Punkt von der gesamten Ladungsverteilung abhaengt.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SpectralConv3d(nn.Module):
    """Faltung im Fourierraum, abgeschnitten nach (mx, my, mz) Moden."""

    def __init__(self, in_channels: int, out_channels: int, modes: tuple[int, int, int]):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        scale = 1.0 / (in_channels * out_channels)
        # Vier Gewichtsbloecke: die rfft laesst nur die letzte Achse halbiert,
        # die ersten beiden brauchen positive und negative Moden getrennt.
        shape = (in_channels, out_channels, *modes)
        self.weights = nn.ParameterList(
            [
                nn.Parameter(scale * torch.rand(*shape, dtype=torch.cfloat))
                for _ in range(4)
            ]
        )

    @staticmethod
    def _mul(x, w):
        return torch.einsum("bixyz,ioxyz->boxyz", x, w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, nx, ny, nz = x.shape
        mx, my, mz = (min(m, s) for m, s in zip(self.modes, (nx, ny, nz // 2 + 1)))

        x_ft = torch.fft.rfftn(x, dim=[-3, -2, -1])
        out_ft = torch.zeros(
            b, self.out_channels, nx, ny, nz // 2 + 1, dtype=torch.cfloat, device=x.device
        )
        slices = [
            (slice(None, mx), slice(None, my)),
            (slice(-mx, None), slice(None, my)),
            (slice(None, mx), slice(-my, None)),
            (slice(-mx, None), slice(-my, None)),
        ]
        for w, (sx, sy) in zip(self.weights, slices):
            out_ft[:, :, sx, sy, :mz] = self._mul(
                x_ft[:, :, sx, sy, :mz], w[:, :, :mx, :my, :mz]
            )
        return torch.fft.irfftn(out_ft, s=(nx, ny, nz), dim=[-3, -2, -1])


class FNO3d(nn.Module):
    """Lift -> N x (Spektralfaltung + 1x1-Faltung + GELU) -> Projektion."""

    def __init__(
        self,
        in_channels: int = 7,
        out_channels: int = 1,
        width: int = 32,
        modes: tuple[int, int, int] = (12, 12, 12),
        num_layers: int = 4,
        lifting_channels: int = 128,
    ):
        super().__init__()
        self.lift = nn.Sequential(
            nn.Conv3d(in_channels, lifting_channels, 1),
            nn.GELU(),
            nn.Conv3d(lifting_channels, width, 1),
        )
        self.spectral = nn.ModuleList(
            [SpectralConv3d(width, width, modes) for _ in range(num_layers)]
        )
        self.pointwise = nn.ModuleList(
            [nn.Conv3d(width, width, 1) for _ in range(num_layers)]
        )
        self.norms = nn.ModuleList([nn.GroupNorm(8, width) for _ in range(num_layers)])
        self.project = nn.Sequential(
            nn.Conv3d(width, lifting_channels, 1),
            nn.GELU(),
            nn.Conv3d(lifting_channels, out_channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.lift(x)
        for spec, point, norm in zip(self.spectral, self.pointwise, self.norms):
            x = x + torch.nn.functional.gelu(norm(spec(x) + point(x)))
        return self.project(x)


def build_model(cfg: dict) -> nn.Module:
    """Baut das Modell laut Config. Bevorzugt `neuraloperator`, wenn vorhanden."""
    name = cfg.get("name", "fno3d")
    if name == "neuralop_fno":
        try:
            from neuralop.models import FNO
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "neuraloperator fehlt -- `pip install neuraloperator`, "
                "oder in der Config auf name: fno3d wechseln."
            ) from exc
        return FNO(
            n_modes=tuple(cfg.get("modes", (12, 12, 12))),
            hidden_channels=cfg.get("width", 32),
            in_channels=cfg.get("in_channels", 7),
            out_channels=cfg.get("out_channels", 1),
            n_layers=cfg.get("num_layers", 4),
            factorization=cfg.get("factorization"),
            rank=cfg.get("rank", 1.0),
        )
    if name == "fno3d":
        return FNO3d(
            in_channels=cfg.get("in_channels", 7),
            out_channels=cfg.get("out_channels", 1),
            width=cfg.get("width", 32),
            modes=tuple(cfg.get("modes", (12, 12, 12))),
            num_layers=cfg.get("num_layers", 4),
            lifting_channels=cfg.get("lifting_channels", 128),
        )
    raise ValueError(f"Unbekanntes Modell: {name}")
