"""G2-Abnahmetest: Feld-Paar aus Cheetah gegen die SpaceChargeKick-Referenz.

Abnahmekriterien (Plan G2):
- phi aus ``make_sample`` (aus phi_hat zurueckgerechnet) stimmt mit
  ``SpaceChargeKick._solve_poisson_equation`` auf demselben Beam mit
  relativem Fehler < 1e-6 ueberein.
- Die Entdimensionalisierung passt zu ``utils/normalization.py``: der exakte
  Solver in Hut-Einheiten (``ExactOperator``) reproduziert phi_hat aus rho_hat.
Zusaetzlich: Formen, dtype float32, endlich, deterministisch.
"""

from __future__ import annotations

import torch

from data.fields import _compute_quantities, make_sample, make_space_charge_kick
from models.exact import ExactOperator
from utils.normalization import Scaling, build_input, potential_from_hat


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


def _scaling(sample: dict) -> Scaling:
    return Scaling(
        total_charge=torch.tensor([sample["total_charge"]], dtype=torch.float64),
        reference_length=torch.tensor([sample["reference_length"]], dtype=torch.float64),
        aspect=sample["aspect"][None].double(),
    )


def _rel_err(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).norm() / b.norm())


def test_phi_matches_spacechargekick_reference():
    res = 16
    params = _sample_params(seed=3)
    sck = make_space_charge_kick(res)  # identisches Element wie in make_sample

    # Referenz: phi direkt aus SpaceChargeKick auf dem von make_sample
    # verwendeten Beam (deterministisch ueber params['seed']).
    beam, grid_dimensions, cell_size, xp = _compute_quantities(params, sck)
    phi_ref = sck._solve_poisson_equation(beam, xp, cell_size, grid_dimensions)  # (1,res,res,res)

    sample = make_sample(params, res)
    phi_make = potential_from_hat(sample["phi_hat"][None].double(), _scaling(sample))

    rel_err = _rel_err(phi_make, phi_ref.double())
    assert rel_err < 1e-6, f"relativer phi-Fehler zu gross: {rel_err:.2e}"


def test_exact_operator_reproduces_phi_hat():
    """phi_hat = ExactOperator(rho_hat): Daten und Normierung sind konsistent."""
    sample = make_sample(_sample_params(seed=4), 16)
    x = build_input(sample["rho_hat"][None], _scaling(sample))
    phi_exact = ExactOperator()(x.double())[:, 0]

    rel_err = _rel_err(phi_exact, sample["phi_hat"][None].double())
    assert rel_err < 1e-4, f"ExactOperator weicht ab: {rel_err:.2e}"


def test_rho_hat_mean_is_one():
    sample = make_sample(_sample_params(seed=2), 16)
    assert abs(float(sample["rho_hat"].double().mean()) - 1.0) < 1e-5
    assert abs(float(sample["aspect"].prod()) - 1.0) < 1e-9


def test_sample_shapes_and_dtypes():
    sample = make_sample(_sample_params(seed=1), (16, 12, 8))

    assert tuple(sample["rho_hat"].shape) == (16, 12, 8), sample["rho_hat"].shape
    assert tuple(sample["phi_hat"].shape) == (16, 12, 8), sample["phi_hat"].shape
    assert sample["rho_hat"].dtype == torch.float32
    assert sample["phi_hat"].dtype == torch.float32
    assert tuple(sample["aspect"].shape) == (3,)
    assert tuple(sample["cell_size"].shape) == (3,)
    assert sample["gamma"] > 1.0


def test_sample_finite():
    sample = make_sample(_sample_params(seed=5), 16)

    assert torch.isfinite(sample["rho_hat"]).all(), "rho_hat enthaelt NaN/Inf"
    assert torch.isfinite(sample["phi_hat"]).all(), "phi_hat enthaelt NaN/Inf"


def test_sample_deterministic_same_params():
    params = _sample_params(seed=9)
    s1 = make_sample(params, 16)
    s2 = make_sample(params, 16)
    assert torch.equal(s1["rho_hat"], s2["rho_hat"])
    assert torch.equal(s1["phi_hat"], s2["phi_hat"])
