import sys
from pathlib import Path

import pytest

# The run scripts are not a package; make `import evaluate_model` work.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

# Small beam parameter bounds for fast tests, independent of the TODO defaults in
# ``scno.data.sampling.DEFAULT_RANGES``.
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


@pytest.fixture(scope="module")
def h5_file(tmp_path_factory):
    """A small raw dataset file (6 samples, 16^3)."""
    from scno.data.generate import write_hdf5

    path = tmp_path_factory.mktemp("h5") / "small.h5"
    return write_hdf5(path, num_samples=6, grid_shape=16, seed=5, ranges=SMALL_RANGES)
