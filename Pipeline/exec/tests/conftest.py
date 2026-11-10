import sys
from pathlib import Path

import pytest

# The run scripts are not a package; make `import evaluate_model` work.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Small beam parameter bounds for fast tests, independent of the defaults in
# ``scno.data.generate.DEFAULT_RANGES``.
SMALL_RANGES = {
    "sigma_x": {"min": 5.0e-4, "max": 1.5e-3},
    "sigma_y": {"min": 5.0e-4, "max": 1.5e-3},
    "sigma_tau": {"min": 1.0e-3, "max": 5.0e-3},
    "energy": {"min": 5.0e8, "max": 1.5e9},
    "total_charge": {"min": 1.0e-10, "max": 2.0e-9},
    "n_particles": {"min": 1.0e4, "max": 2.0e4},
}


@pytest.fixture
def small_ranges() -> dict:
    return {key: dict(bounds) for key, bounds in SMALL_RANGES.items()}
