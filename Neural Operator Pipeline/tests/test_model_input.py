"""Input format: dataset, training, evaluation and the Cheetah element feed the same model.

The input format is defined once (``input_channels``); these tests keep the model
settings of ``scno.config``, saving/loading, the decoding of predictions and
the loss interface consistent with it.
"""

from __future__ import annotations

import torch
from evaluate_model import load_model
from neuralop.data.transforms.data_processors import DefaultDataProcessor
from neuralop.data.transforms.normalizers import UnitGaussianNormalizer
from neuralop.models import FNO

from scno.config import MODEL
from scno.evaluate import predict
from scno.models.exact import ExactOperator, solve_poisson_on_grid
from scno.utils.losses import SobolevLoss
from scno.utils.normalization import (
    INPUT_CHANNELS,
    decode_prediction,
    input_channels,
    nondimensionalize_fields,
    stretched_cell_size,
)


def _physical_fields(batch: int = 2, grid=(8, 8, 8)):
    """Coarse Gaussian bunch in physical units (C/m^3), phi solved exactly."""
    axes = [torch.linspace(-1.0, 1.0, n, dtype=torch.float64) for n in grid]
    r2 = sum(a.square() for a in torch.meshgrid(*axes, indexing="ij"))
    scale = torch.tensor([1e-3, 5e-2], dtype=torch.float64)[:batch, None, None, None]
    rho = torch.exp(-r2 / 0.2)[None].repeat(batch, 1, 1, 1) * scale
    cell_size = torch.tensor(
        [[1e-4, 2e-4, 3e-5], [2e-4, 1e-4, 5e-5]], dtype=torch.float64
    )[:batch]
    gamma = torch.tensor([100.0, 250.0], dtype=torch.float64)[:batch]
    phi = solve_poisson_on_grid(rho, stretched_cell_size(cell_size, gamma))
    return rho, phi, cell_size, gamma


def test_model_settings_accept_input_channels():
    rho_hat = torch.rand(2, 8, 8, 8)
    extent = torch.tensor([[0.5, 1.0, 2.0]]).expand(2, -1)  # product = 1
    x = input_channels(rho_hat, extent)
    assert x.shape[1] == INPUT_CHANNELS

    assert MODEL["in_channels"] == INPUT_CHANNELS
    assert FNO(**MODEL)(x).shape == (2, 1, 8, 8, 8)


def test_state_dict_round_trip(tmp_path):
    """The .pth holds only the state dict; neuralop's `_metadata` rebuilds the model."""
    model = FNO(**MODEL).eval()
    path = tmp_path / "model.pth"
    torch.save(model.state_dict(), path)

    loaded = load_model(path, torch.device("cpu")).eval()
    x = torch.rand(1, INPUT_CHANNELS, 8, 8, 8)
    assert torch.equal(loaded(x), model(x))


def test_decode_prediction_exact_operator_gives_volts():
    """ExactOperator in hat units + decode_prediction == phi [V] from the solver."""
    rho, phi, cell_size, gamma = _physical_fields()
    rho_hat, phi_hat, scaling = nondimensionalize_fields(rho, phi, cell_size, gamma)

    out = ExactOperator()(input_channels(rho_hat, scaling.normalized_extent))
    phi_hat_dec, phi_dec = decode_prediction(out[:, 0], scaling, nondim=True)

    assert torch.allclose(phi_hat_dec, phi_hat, rtol=1e-6)
    assert torch.allclose(phi_dec, phi, rtol=1e-6)

    # nondim=False: the network learns phi in V directly, phi_hat is computed from it
    phi_hat_raw, phi_raw = decode_prediction(phi, scaling, nondim=False)
    assert torch.equal(phi_raw, phi)
    assert torch.allclose(phi_hat_raw, phi_hat, rtol=1e-12)


def test_predict_undoes_output_normalizer():
    """``predict`` returns the model output in the units of y, also when the data
    processor is (by default) in training mode."""
    rho, phi, cell_size, gamma = _physical_fields()
    y = phi[:, None].float()
    normalizer = UnitGaussianNormalizer(dim=[0, 2, 3, 4])
    normalizer.fit(y)
    processor = DefaultDataProcessor(out_normalizer=normalizer)
    assert processor.training

    class Standardized(torch.nn.Module):
        """Stand-in network that outputs exactly the standardized target."""

        def forward(self, x):
            return normalizer.transform(y)

    out = predict(Standardized(), {"x": y, "y": y}, processor)
    assert torch.allclose(out, y, rtol=1e-4)


def test_loss_accepts_trainer_keywords():
    """The neuralop Trainer calls the loss as ``loss(out, **sample)``."""
    rho_hat = torch.rand(2, 8, 8, 8)
    x = input_channels(rho_hat, torch.tensor([[0.5, 1.0, 2.0]]).expand(2, -1))
    pred, target = torch.rand(2, 1, 8, 8, 8), torch.rand(2, 1, 8, 8, 8)

    loss = SobolevLoss(gradient_weight=1.0, physics_weight=0.1)(pred, y=target, x=x)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert SobolevLoss(gradient_weight=0.0)(target, y=target, x=x) == 0.0
