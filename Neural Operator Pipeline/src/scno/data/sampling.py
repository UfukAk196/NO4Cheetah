"""G1 - Parameter sampler for the Cheetah space-charge data generator.

Draws one set of beam parameters per sample from the bounds in ``DEFAULT_RANGES``
(or the ``data.ranges`` override in the config):

- ``sigma_x``  [m]  (horizontal, log-uniform)
- ``sigma_y``  [m]  (vertical,     log-uniform)
- ``sigma_tau``[m]  (longitudinal, log-uniform)
- ``energy``   [eV] (uniform)
- ``total_charge`` [C] (uniform)
- ``n_particles`` [int] (uniform)

The three spatial sigmas spread over orders of magnitude (µm..mm), hence log-uniform;
the other three quantities are drawn uniformly.

The function is deterministic: the same ``seed`` gives an identical list.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# --------------------------------------------------------------------------- #
# Default ranges -- the single source of the beam parameter bounds. A config can
# override them with a ``data.ranges`` block of the same shape.
# TODO: placeholder values; replace with the real bounds from the ML pipeline before
# generating the full dataset. All bounds are inclusive: values lie in [min, max].
# --------------------------------------------------------------------------- #
DEFAULT_RANGES: dict[str, dict[str, float]] = {
    # log-uniform
    "sigma_x": {"min": 5.0e-6, "max": 1.5e-3},  # [m] horizontal
    "sigma_y": {"min": 5.0e-6, "max": 1.5e-3},  # [m] vertical
    "sigma_tau": {"min": 1.0e-3, "max": 5.0e-3},  # [m] longitudinal
    # uniform
    "energy": {"min": 5.0e7, "max": 1.5e8},  # [eV]
    "total_charge": {"min": 1.0e-10, "max": 2.0e-9},  # [C]
    # uniform (integer)
    "n_particles": {"min": 1.0e4, "max": 1.0e6},  # [1] macroparticles per bunch
}

# Quantities drawn log-uniformly (see docstring).
LOG_UNIFORM_KEYS = ("sigma_x", "sigma_y", "sigma_tau")

# Quantities drawn uniformly.
UNIFORM_KEYS = ("energy", "total_charge", "n_particles")


def _all_ranges_present(ranges: dict[str, dict[str, float]]) -> list[str]:
    """Checks that all six parameters have a {"min": .., "max": ..}.

    Returns the list of missing parameters (empty = ok).
    """
    missing: list[str] = []
    for key in LOG_UNIFORM_KEYS + UNIFORM_KEYS:
        if key not in ranges:
            missing.append(key)
            continue
        for bound in ("min", "max"):
            if bound not in ranges[key]:
                missing.append(f"{key}.{bound}")
    return missing


def sample_beam_params(
    n: int, seed: int, ranges: dict[str, dict[str, float]] | None = None
) -> list[dict[str, Any]]:
    """Draws ``n`` independent sets of beam parameters.

    Deterministic: the same ``seed`` gives identical output.

    Parameters
    ----------
    n : int
        Number of samples (sets of beam parameters).
    seed : int
        Seed of the underlying random generator.
    ranges : dict[str, dict[str, float]], optional
        Bounds {"min": .., "max": ..} per parameter. Defaults to ``DEFAULT_RANGES``.

    Returns
    -------
    list[dict[str, Any]]
        ``n`` dicts with keys ``sigma_x, sigma_y, sigma_tau`` [m], ``energy`` [eV],
        ``total_charge`` [C], ``n_particles`` [int].
    """
    if n < 0:
        raise ValueError(f"n must be >= 0, got {n}")

    if ranges is None:
        ranges = DEFAULT_RANGES

    missing = _all_ranges_present(ranges)
    if missing:
        raise ValueError(
            "ranges must have a 'min'/'max' entry for every parameter; "
            f"missing for: {missing}"
        )

    rng = np.random.default_rng(seed)

    params_list: list[dict[str, Any]] = []
    for _ in range(n):
        sample: dict[str, Any] = {}

        # The uniformly drawn quantities come first, so that the order of the random
        # numbers stays stable if someone changes the set of log-uniform parameters.
        for key in UNIFORM_KEYS:
            # float() enforced: YAML configs sometimes deliver bounds as strings
            # (e.g. '5.0e8'), which would otherwise end up as a string comparison.
            lo, hi = float(ranges[key]["min"]), float(ranges[key]["max"])
            if lo > hi:
                raise ValueError(f"ranges[{key}]: min ({lo}) > max ({hi})")
            sample[key] = float(rng.uniform(lo, hi))

        # n_particles is an integer and must never be < 1.
        sample["n_particles"] = max(1, int(round(sample["n_particles"])))

        for key in LOG_UNIFORM_KEYS:
            lo, hi = float(ranges[key]["min"]), float(ranges[key]["max"])
            if lo <= 0 or hi <= 0:
                raise ValueError(f"ranges[{key}] must be strictly positive")
            if lo > hi:
                raise ValueError(f"ranges[{key}]: min ({lo}) > max ({hi})")
            # log-uniform: exp(U(ln(min), ln(max)))
            sample[key] = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))

        params_list.append(sample)

    return params_list
