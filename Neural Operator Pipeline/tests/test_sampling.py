"""G1 acceptance test for the parameter sampler.

Checks the acceptance criterion of the plan (G1):
- the same ``seed`` -> an identical list
- all values within the bounds of ``ranges``
Additionally: correct keys/types and the log-uniform / uniform distribution.
"""

from __future__ import annotations

from scno.data.sampling import (
    DEFAULT_RANGES,
    LOG_UNIFORM_KEYS,
    UNIFORM_KEYS,
    sample_beam_params,
)


def _example_ranges() -> dict[str, dict[str, float]]:
    """Small test bounds, independent of the TODO defaults."""
    return {
        "sigma_x": {"min": 1.0e-4, "max": 1.0e-3},
        "sigma_y": {"min": 1.0e-4, "max": 1.0e-3},
        "sigma_tau": {"min": 1.0e-3, "max": 8.0e-3},
        "energy": {"min": 1.0e8, "max": 1.0e9},
        "total_charge": {"min": 1.0e-10, "max": 1.0e-9},
        "n_particles": {"min": 1.0e4, "max": 1.0e5},
    }


def test_same_seed_identical_output():
    n = 20
    ranges = _example_ranges()

    a = sample_beam_params(n, seed=42, ranges=ranges)
    b = sample_beam_params(n, seed=42, ranges=ranges)

    assert a == b, "the same seed must give an identical list"


def test_different_seed_differs():
    n = 100
    ranges = _example_ranges()

    a = sample_beam_params(n, seed=1, ranges=ranges)
    b = sample_beam_params(n, seed=2, ranges=ranges)

    assert a != b, "different seeds should give different samples"


def test_all_values_in_ranges():
    n = 500
    ranges = _example_ranges()

    samples = sample_beam_params(n, seed=7, ranges=ranges)
    assert len(samples) == n

    for sample in samples:
        for key, bounds in ranges.items():
            lo, hi = bounds["min"], bounds["max"]
            value = sample[key]
            assert lo <= value <= hi, f"{key}={value} is not in [{lo}, {hi}]"


def test_keys_and_types():
    n = 5
    ranges = _example_ranges()
    samples = sample_beam_params(n, seed=3, ranges=ranges)

    expected_keys = set(LOG_UNIFORM_KEYS + UNIFORM_KEYS)
    for sample in samples:
        assert set(sample.keys()) == expected_keys
        # Spatial sigmas [m] are floats
        for key in LOG_UNIFORM_KEYS:
            assert isinstance(sample[key], float), f"{key} must be a float"
        # Energy [eV] and charge [C] are floats
        assert isinstance(sample["energy"], float)
        assert isinstance(sample["total_charge"], float)
        # The number of particles is an integer (but not a bool, even though bool is
        # an int)
        assert isinstance(sample["n_particles"], int)
        assert not isinstance(sample["n_particles"], bool)
        assert sample["n_particles"] >= 1


def test_default_ranges_valid():
    """The TODO defaults must themselves be consistent and usable."""
    samples = sample_beam_params(10, seed=0, ranges=DEFAULT_RANGES)
    assert len(samples) == 10
    for sample in samples:
        for key, bounds in DEFAULT_RANGES.items():
            assert bounds["min"] <= sample[key] <= bounds["max"]


def test_ranges_as_strings_from_yaml():
    """YAML configs sometimes deliver bounds as strings (e.g. '5.0e8').

    sample_beam_params must cope with that without running into string comparisons.
    """
    ranges = {
        "sigma_x": {"min": "5.0e-4", "max": "1.5e-3"},
        "sigma_y": {"min": "5.0e-4", "max": "1.5e-3"},
        "sigma_tau": {"min": "1.0e-3", "max": "5.0e-3"},
        "energy": {"min": "5.0e8", "max": "1.5e9"},
        "total_charge": {"min": "1.0e-10", "max": "2.0e-9"},
        "n_particles": {"min": "1.0e4", "max": "3.0e4"},
    }
    samples = sample_beam_params(50, seed=0, ranges=ranges)
    assert len(samples) == 50
    for sample in samples:
        for key, bounds in ranges.items():
            lo, hi = float(bounds["min"]), float(bounds["max"])
            assert lo <= sample[key] <= hi, (key, sample[key])
        assert isinstance(sample["n_particles"], int)


def test_log_uniform_spans_decades_within_bounds():
    """Log-uniformly drawn quantities lie strictly within [min, max]."""
    ranges = _example_ranges()
    samples = sample_beam_params(2000, seed=11, ranges=ranges)

    for key in LOG_UNIFORM_KEYS:
        lo, hi = ranges[key]["min"], ranges[key]["max"]
        gathered = [s[key] for s in samples]
        assert min(gathered) >= lo
        assert max(gathered) <= hi
        # Real spread: min/max should end up close to the bounds
        assert min(gathered) < lo * 1.5
        assert max(gathered) > hi / 1.5
