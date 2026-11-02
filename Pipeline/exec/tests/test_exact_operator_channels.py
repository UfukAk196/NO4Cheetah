"""Self-check of the flexible ``ExactOperator`` input channels.

The out-of-band ``cell_size`` path must give exactly the same result as the
``normalized_extents`` path for the same geometry -- otherwise the reference used for
``"gamma"`` / ``"only rho"`` data would be silently wrong.
"""

import torch

from scno.models.exact import ExactOperator
from scno.utils.normalization import INPUT_CHANNELS, input_channels

GRID = (4, 4, 4)


def _geometry(batch=2):
    """A (B, 3) normalized extent with product 1, and the matching cell size in units of L."""
    extent = torch.tensor([[1.0, 2.0, 0.5], [0.5, 1.0, 2.0]])
    grid = torch.tensor(GRID, dtype=torch.float64)
    return extent, extent / grid  # cell size in units of L


def test_paths_agree():
    """4-channel input and out-of-band cell_size must produce identical phi_hat."""
    rho = torch.rand(2, *GRID, dtype=torch.float64)
    extent, cell = _geometry()

    x4 = input_channels(rho, extent, "normalized_extents")
    from_x = ExactOperator()(x4)

    from_cell = ExactOperator(cell_size=cell)(x4)
    assert torch.allclose(from_x, from_cell, atol=1e-12), (
        (from_x - from_cell).abs().max()
    )


def test_gamma_and_only_rho_variants():
    """The other variants work once the geometry is supplied out of band."""
    rho = torch.rand(2, *GRID, dtype=torch.float64)
    extent, cell = _geometry()
    gamma = torch.tensor([1.0, 100.0])
    reference = ExactOperator()(input_channels(rho, extent, "normalized_extents"))

    for variant, aspect in (("gamma", gamma), ("only rho", None)):
        x = input_channels(rho, aspect, variant)
        assert x.shape[1] == INPUT_CHANNELS[variant]
        out = ExactOperator(cell_size=cell)(x)
        assert torch.allclose(out, reference, atol=1e-12), variant


def test_missing_cell_size_raises():
    """Without the geometry the operator must fail loudly, not silently mis-slice."""
    rho = torch.rand(1, *GRID, dtype=torch.float64)
    x = input_channels(rho, torch.tensor([1.0]), "gamma")
    try:
        ExactOperator()(x)
    except ValueError as e:
        assert "cell_size" in str(e)
    else:
        raise AssertionError("2-channel input without cell_size must raise")


if __name__ == "__main__":
    test_paths_agree()
    test_gamma_and_only_rho_variants()
    test_missing_cell_size_raises()
    print("OK: both paths agree, gamma/only-rho work, missing cell_size raises")
