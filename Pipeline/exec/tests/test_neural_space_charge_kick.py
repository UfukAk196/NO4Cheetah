import pytest
import torch
from cheetah.accelerator import SpaceChargeKick
from cheetah.particles import ParticleBeam
from neuralop.models import FNO

from scno.accelerator import NeuralSpaceChargeKick
from scno.models.exact import ExactOperator


def _beam(**overrides) -> ParticleBeam:
    """A vectorized beam of two bunches with different sizes, energies and charges."""
    torch.manual_seed(0)
    parameters = dict(
        num_particles=10_000,
        sigma_x=torch.tensor([1e-3, 2e-3]),
        sigma_y=torch.tensor([5e-4, 1e-3]),
        sigma_tau=torch.tensor([2e-3, 3e-3]),
        energy=torch.tensor([1e8, 5e8]),
        total_charge=torch.tensor([1e-9, 2e-9]),
        dtype=torch.float64,
    )
    parameters.update(overrides)
    return ParticleBeam.from_parameters(**parameters)


def _kicks(grid_shape=(16, 12, 8)) -> tuple[SpaceChargeKick, NeuralSpaceChargeKick]:
    effect_length = torch.tensor(0.5, dtype=torch.float64)
    reference = SpaceChargeKick(effect_length=effect_length, grid_shape=grid_shape)
    neural = NeuralSpaceChargeKick(
        operator=ExactOperator(), effect_length=effect_length, grid_shape=grid_shape
    )
    return reference, neural


def test_exact_operator_reproduces_space_charge_kick():
    """With the exact operator, the element must track like `SpaceChargeKick`."""
    beam = _beam()
    reference, neural = _kicks()

    expected = reference.track(beam).particles
    actual = neural.track(beam).particles

    momentum_change = (expected - beam.particles)[..., [1, 3, 5]]
    assert momentum_change.abs().max() > 0.0  # the kick is not trivially zero
    assert torch.allclose(actual, expected, rtol=1e-9, atol=1e-15)


def test_gradients_match_space_charge_kick():
    """Gradients w.r.t. the beam size through the element equal the exact path."""

    def objective(kick, sigma_x):
        outgoing = kick.track(_beam(sigma_x=sigma_x))
        return outgoing.particles[..., [1, 3]].square().sum()

    reference, neural = _kicks()
    sigma_ref = torch.tensor([1e-3, 2e-3], dtype=torch.float64, requires_grad=True)
    sigma_neural = sigma_ref.detach().clone().requires_grad_(True)

    (grad_ref,) = torch.autograd.grad(objective(reference, sigma_ref), sigma_ref)
    (grad_neural,) = torch.autograd.grad(objective(neural, sigma_neural), sigma_neural)

    assert torch.isfinite(grad_neural).all()
    assert grad_neural.abs().max() > 0.0
    assert torch.allclose(grad_neural, grad_ref, rtol=1e-6)


def test_trained_operator_is_differentiable():
    """A float32 neuralop FNO runs inside the float64 lattice and lets gradients flow."""
    fno = FNO(n_modes=(4, 4, 4), in_channels=4, out_channels=1, hidden_channels=4)
    neural = NeuralSpaceChargeKick(
        operator=fno, effect_length=torch.tensor(0.5, dtype=torch.float64), grid_shape=(8, 8, 8)
    )
    energy = torch.tensor([1e8, 5e8], dtype=torch.float64, requires_grad=True)

    outgoing = neural.track(_beam(energy=energy))
    (grad,) = torch.autograd.grad(outgoing.particles[..., 5].sum(), energy)

    assert outgoing.particles.dtype == torch.float64
    assert torch.isfinite(grad).all()


@pytest.mark.parametrize("grid_shape", [(8, 8, 8), (16, 16, 16)])
def test_clone_keeps_operator(grid_shape):
    _, neural = _kicks(grid_shape)
    clone = neural.clone()

    assert isinstance(clone, NeuralSpaceChargeKick)
    assert isinstance(clone.operator, ExactOperator)
    assert clone.operator is not neural.operator
    assert clone.grid_shape == grid_shape
