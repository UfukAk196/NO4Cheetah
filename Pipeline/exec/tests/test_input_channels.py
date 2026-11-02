"""Self-check of the three ``input_channels`` variants and the derived ``in_channels``.

The channel count of the model input must match ``MODEL["in_channels"]`` for every
``DATA["in_channel_type"]``; a mismatch is a shape error deep inside the FNO, so it is
checked here at the source.
"""

import torch

from scno.config import DATA, MODEL
from scno.utils.normalization import INPUT_CHANNELS, input_channels

GRID = (4, 5, 6)


def test_variants_and_channel_counts():
    rho = torch.rand(2, *GRID)
    extent = torch.tensor([[1.0, 2.0, 0.5], [0.5, 1.0, 2.0]])  # (B, 3), product = 1
    gamma = torch.tensor([1.0, 100.0])  # (B,)

    cases = {
        "normalized_extents": (extent, 4),
        "gamma": (gamma, 2),
        "only rho": (None, 1),
    }
    for name, (aspect, channels) in cases.items():
        x = input_channels(rho, aspect, name)
        assert x.shape == (2, channels, *GRID), (name, x.shape)
        assert channels == INPUT_CHANNELS[name]
        # Channel 0 is always the field itself.
        assert torch.equal(x[:, 0], rho)


def test_config_matches_variant():
    """MODEL["in_channels"] must equal the channel count of the configured variant."""
    assert MODEL["in_channels"] == INPUT_CHANNELS[DATA["in_channel_type"]]


def test_missing_aspect_raises():
    rho = torch.rand(1, *GRID)
    for name in ("normalized_extents", "gamma"):
        try:
            input_channels(rho, None, name)
        except ValueError:
            pass
        else:
            raise AssertionError(f"{name} without aspect must raise")


if __name__ == "__main__":
    test_variants_and_channel_counts()
    test_config_matches_variant()
    test_missing_aspect_raises()
    print(
        f"OK: 3 variants, MODEL['in_channels']={MODEL['in_channels']} "
        f"matches DATA['in_channel_type']={DATA['in_channel_type']!r}"
    )
