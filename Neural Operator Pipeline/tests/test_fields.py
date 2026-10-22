"""G2 acceptance test: field pair from Cheetah against the SpaceChargeKick reference.

Acceptance criteria (plan G2):
- phi from ``make_sample`` (physical, in V) agrees with
  ``SpaceChargeKick._solve_poisson_equation`` on the same beam with a relative error
  < 1e-6.
- The nondimensionalization fits ``utils/normalization.py``: the exact solver in hat
  units (``ExactOperator``) reproduces phi_hat from rho_hat.
Additionally: shapes, dtype float32, finite, deterministic.
"""

from __future__ import annotations

import torch

from scno.data.fields import _compute_quantities, make_sample, make_space_charge_kick
from scno.models.exact import ExactOperator
from scno.utils.normalization import input_channels, nondimensionalize_fields


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


def _nondim(sample: dict):
    """make_sample output (without batch) -> (rho_hat, phi_hat, Scaling) with batch 1."""
    return nondimensionalize_fields(
        sample["rho"][None].double(),
        sample["phi"][None].double(),
        sample["cell_size"][None],
        torch.tensor([sample["gamma"]], dtype=torch.float64),
    )


def _rel_err(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).norm() / b.norm())


def test_phi_matches_spacechargekick_reference():
    res = 16
    params = _sample_params(seed=3)
    sck = make_space_charge_kick(res)  # identical element as in make_sample

    # Reference: phi directly from SpaceChargeKick on the beam used by make_sample
    # (deterministic via params['seed']).
    beam, grid_dimensions, cell_size, xp = _compute_quantities(params, sck)
    phi_ref = sck._solve_poisson_equation(beam, xp, cell_size, grid_dimensions)

    sample = make_sample(params, res)

    rel_err = _rel_err(sample["phi"][None].double(), phi_ref.double())
    assert rel_err < 1e-6, f"relative phi error too large: {rel_err:.2e}"


def test_exact_operator_reproduces_phi_hat():
    """phi_hat = ExactOperator(rho_hat): data and normalization are consistent."""
    rho_hat, phi_hat, scaling = _nondim(make_sample(_sample_params(seed=4), 16))
    phi_exact = ExactOperator()(input_channels(rho_hat, scaling.normalized_extent))[:, 0]

    rel_err = _rel_err(phi_exact, phi_hat)
    assert rel_err < 1e-4, f"ExactOperator deviates: {rel_err:.2e}"


def test_rho_hat_mean_is_one():
    rho_hat, _, scaling = _nondim(make_sample(_sample_params(seed=2), 16))
    assert abs(float(rho_hat.mean()) - 1.0) < 1e-5
    assert abs(float(scaling.normalized_extent.prod()) - 1.0) < 1e-9


def test_sample_shapes_and_dtypes():
    sample = make_sample(_sample_params(seed=1), (16, 12, 8))

    assert set(sample) == {"rho", "phi", "cell_size", "gamma"}
    assert tuple(sample["rho"].shape) == (16, 12, 8), sample["rho"].shape
    assert tuple(sample["phi"].shape) == (16, 12, 8), sample["phi"].shape
    assert sample["rho"].dtype == torch.float32
    assert sample["phi"].dtype == torch.float32
    assert tuple(sample["cell_size"].shape) == (3,)
    assert sample["gamma"] > 1.0


def test_sample_is_physical():
    """Raw fields, not nondimensionalized: charge on the grid ~ beam charge."""
    params = _sample_params(seed=6)
    sample = make_sample(params, 16)
    charge = abs(float(sample["rho"].double().sum() * sample["cell_size"].prod()))

    assert 0.5 * params["total_charge"] < charge <= 1.001 * params["total_charge"]
    assert float(sample["phi"].abs().max()) > 1.0  # volts, not O(1) hat units


def test_sample_finite():
    sample = make_sample(_sample_params(seed=5), 16)

    assert torch.isfinite(sample["rho"]).all(), "rho contains NaN/Inf"
    assert torch.isfinite(sample["phi"]).all(), "phi contains NaN/Inf"


def test_sample_deterministic_same_params():
    params = _sample_params(seed=9)
    s1 = make_sample(params, 16)
    s2 = make_sample(params, 16)
    assert torch.equal(s1["rho"], s2["rho"])
    assert torch.equal(s1["phi"], s2["phi"])
