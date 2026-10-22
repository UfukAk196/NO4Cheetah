"""
Diagnosis: how good is the nondimensionalization actually?

If phi_hat still spreads over several orders of magnitude across the samples, the
network has to keep learning scales instead of physics -- and the relative error stays
stubbornly high, no matter how large the model is made. This script measures the
spread and compares the two reference lengths against each other.

A result of this kind belongs in the thesis, because it backs a design decision with
numbers instead of gut feeling.

Call: python scripts/inspect_scaling.py --data data/cheetah_sc_train_16.h5
"""

import argparse

import numpy as np

from scno.data.dataset import CheetahDataset


def spread(values: np.ndarray) -> float:
    """Spread in decades between the 5th and 95th percentile."""
    lo, hi = np.percentile(values, [5, 95])
    return float(np.log10(hi / max(lo, 1e-300)))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/cheetah_sc_train_16.h5")
    a = p.parse_args()

    # The file holds raw fields; phi_hat and the scaling are derived on loading.
    fields = CheetahDataset(a.data).read_fields(slice(None))
    phi = fields["phi_hat"].numpy()
    normalized_extent = fields["scaling"].normalized_extent.numpy()
    cell = fields["cell_size"].numpy()
    gamma = fields["gamma"].numpy()
    charge = fields["scaling"].total_charge.numpy()

    std_hat = phi.reshape(len(phi), -1).std(axis=1)

    # Cross-check: the same potentials with the transverse box edge Ex as reference
    # length instead of the geometric mean of all three edges.
    scaled_ex = std_hat * normalized_extent[:, 0]

    print(f"Dataset: {a.data}, {len(phi)} samples, grid {phi.shape[1:]}")
    print()
    print("Raw quantities (what would enter the network without nondimensionalization):")
    print(f"  total charge    : {spread(charge):5.2f} decades")
    print(f"  cell size dx    : {spread(cell[:, 0]):5.2f} decades")
    print(f"  gamma           : {gamma.min():.1f} .. {gamma.max():.1f}")
    print()
    print("After nondimensionalization, spread of std(phi_hat):")
    print(f"  reference L = (Ex*Ey*Ez)^(1/3) : {spread(std_hat):5.2f} decades   <- used")
    print(f"  reference L = Ex               : {spread(scaled_ex):5.2f} decades")
    print()
    print("Aspect ratios E_i / L (enter the network as log):")
    for i, name in enumerate("xyz"):
        col = normalized_extent[:, i]
        print(f"  E{name}/L: {col.min():9.3e} .. {col.max():9.3e}")
    print()
    print("Rule of thumb: if the spread of phi_hat stays below about one decade, the")
    print("scaling is fine. Clearly more means that a separate scalar prediction")
    print("(a second head for log|phi|) is worthwhile.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
