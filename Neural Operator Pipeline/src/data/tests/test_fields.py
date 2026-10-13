"""G2-Abnahmetest: Feld-Paar aus Cheetah gegen die SpaceChargeKick-Referenz.

Abnahmekriterium (Plan G2):
- phi aus ``make_sample`` (VOR Entdimensionalisierung, aus y zurueckgerechnet)
  stimmt mit ``SpaceChargeKick._solve_poisson_equation`` auf demselben Beam
  mit relativem Fehler < 1e-6 ueberein.
Zusaetzlich: Formen (4,res,res,res)/(1,res,res,res), dtype float32, endlich.
"""

from __future__ import annotations

import math

import torch
from cheetah.accelerator import SpaceChargeKick

from cheetah_sc.fields import (
    DEFAULT_GRID_EXTENT,
    EPSILON_0,
    _compute_quantities,
    _prepare,
    make_sample,
)


def _sample_params(seed: int = 0) -> dict:
    return {
        "sigma_x": 1.2e-3,
        "sigma_y": 0.8e-3,
        "sigma_tau": 3.0e-3,
        "energy": 1.2e9,
        "total_charge": 2.5e-9,
        "n_particles": 30000,
        "seed": seed,
    }


def test_phi_matches_spacechargekick_reference():
    res = 16
    params = _sample_params(seed=3)
    sck = SpaceChargeKick(  # identisches Element wie in make_sample
        effect_length=torch.tensor(0.0),
        grid_shape=(res, res, res),
        grid_extent_x=torch.tensor(DEFAULT_GRID_EXTENT),
        grid_extent_y=torch.tensor(DEFAULT_GRID_EXTENT),
        grid_extent_tau=torch.tensor(DEFAULT_GRID_EXTENT),
    )

    # Referenz: phi direkt aus SpaceChargeKick auf dem von make_sample
    # verwendeten Beam (deterministisch ueber params['seed']).
    beam, grid_dimensions, cell_size, xp = _compute_quantities(params, sck)
    phi_ref = sck._solve_poisson_equation(beam, xp, cell_size, grid_dimensions)  # (1,res,res,res)

    # Aus make_sample das y (phi_hat) holen und zurueck entdimensionalisieren.
    x_sample, y_sample = make_sample(params, res)
    assert x_sample.dtype == torch.float32
    assert y_sample.dtype == torch.float32

    # phi_hat = phi * 4*pi*eps0 * L / Q  ->  phi = phi_hat / (4*pi*eps0*L/Q)
    gx, gy, gz = (
        grid_dimensions[..., 0],
        grid_dimensions[..., 1],
        grid_dimensions[..., 2],
    )
    L = (gx * gy * gz) ** (1.0 / 3.0)
    Q = torch.tensor(float(params["total_charge"]), dtype=y_sample.dtype)
    coulomb = torch.tensor(4 * math.pi * EPSILON_0, dtype=y_sample.dtype)
    phi_make = y_sample[0] / (coulomb * L / Q)  # (res,res,res)

    # Relativer Fehler (normiert auf Referenznorm), muss < 1e-6 sein.
    denom = phi_ref.norm()
    rel_err = (phi_make - phi_ref.squeeze(0)).norm() / denom
    assert rel_err < 1e-6, f"relativer phi-Fehler zu gross: {rel_err:.2e}"


def test_sample_shapes_and_dtypes():
    res = 16
    params = _sample_params(seed=1)
    x, y = make_sample(params, res)

    assert tuple(x.shape) == (4, res, res, res), f"x-Form falsch: {x.shape}"
    assert tuple(y.shape) == (1, res, res, res), f"y-Form falsch: {y.shape}"
    assert x.dtype == torch.float32
    assert y.dtype == torch.float32


def test_sample_finite():
    res = 16
    params = _sample_params(seed=5)
    x, y = make_sample(params, res)

    assert torch.isfinite(x).all(), "x enthaelt NaN/Inf"
    assert torch.isfinite(y).all(), "y enthaelt NaN/Inf"


def test_sample_deterministic_same_params():
    res = 16
    params = _sample_params(seed=9)
    x1, y1 = make_sample(params, res)
    x2, y2 = make_sample(params, res)
    assert torch.equal(x1, x2)
    assert torch.equal(y1, y2)
