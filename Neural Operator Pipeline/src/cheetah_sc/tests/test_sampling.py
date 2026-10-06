"""G1-Abnahmetest fuer den Parameter-Sampler.

Prueft das Abnahmekriterium aus dem Plan (G1):
- gleicher ``seed`` -> identische Liste
- alle Werte in den Grenzen aus ```ranges```
Zusaetzlich: korrekte Keys/Typen und die log-uniform / uniform-Verteilung.
"""

from __future__ import annotations

from cheetah_sc.sampling import (
    DEFAULT_RANGES,
    LOG_UNIFORM_KEYS,
    UNIFORM_KEYS,
    sample_beam_params,
)


def _example_ranges() -> dict[str, dict[str, float]]:
    """Kleine, von den TODO-Defaults unabhaengige Testgrenzen."""
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

    assert a == b, "gleicher Seed muss identische Liste liefern"


def test_different_seed_differs():
    n = 100
    ranges = _example_ranges()

    a = sample_beam_params(n, seed=1, ranges=ranges)
    b = sample_beam_params(n, seed=2, ranges=ranges)

    assert a != b, "unterschiedliche Seeds sollten verschiedene Stichproben liefern"


def test_all_values_in_ranges():
    n = 500
    ranges = _example_ranges()

    samples = sample_beam_params(n, seed=7, ranges=ranges)
    assert len(samples) == n

    for sample in samples:
        for key, bounds in ranges.items():
            lo, hi = bounds["min"], bounds["max"]
            value = sample[key]
            assert lo <= value <= hi, (
                f"{key}={value} liegt nicht in [{lo}, {hi}]"
            )


def test_keys_and_types():
    n = 5
    ranges = _example_ranges()
    samples = sample_beam_params(n, seed=3, ranges=ranges)

    expected_keys = set(LOG_UNIFORM_KEYS + UNIFORM_KEYS)
    for sample in samples:
        assert set(sample.keys()) == expected_keys
        # raeumliche Sigmas [m] sind Fliesskomma
        for key in LOG_UNIFORM_KEYS:
            assert isinstance(sample[key], float), f"{key} muss float sein"
        # Energie [eV] und Ladung [C] sind Fliesskomma
        assert isinstance(sample["energy"], float)
        assert isinstance(sample["total_charge"], float)
        # Teilchenzahl ist ganzzahlig (aber kein bool, auch wenn bool int ist)
        assert isinstance(sample["n_particles"], int)
        assert not isinstance(sample["n_particles"], bool)
        assert sample["n_particles"] >= 1


def test_default_ranges_valid():
    """Die TODO-Defaults muessen selbst konsistent und aufrufbar sein."""
    samples = sample_beam_params(10, seed=0, ranges=DEFAULT_RANGES)
    assert len(samples) == 10
    for sample in samples:
        for key, bounds in DEFAULT_RANGES.items():
            assert bounds["min"] <= sample[key] <= bounds["max"]


def test_log_uniform_spans_decades_within_bounds():
    """Log-uniform gezogene Groessen liegen strikt innerhalb [min, max]."""
    ranges = _example_ranges()
    samples = sample_beam_params(2000, seed=11, ranges=ranges)

    for key in LOG_UNIFORM_KEYS:
        lo, hi = ranges[key]["min"], ranges[key]["max"]
        gathered = [s[key] for s in samples]
        assert min(gathered) >= lo
        assert max(gathered) <= hi
        # echte Streuung: min/max sollten nahe an den Grenzen landen
        assert min(gathered) < lo * 1.5
        assert max(gathered) > hi / 1.5
