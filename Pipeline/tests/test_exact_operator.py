"""Control: the exact solver as a model must pass every evaluation with zero error.

If it does not, the bug is in the normalization, the axis order or the evaluation --
look there, not in the network.
"""

from __future__ import annotations

import torch

from scno.data.dataset import CheetahDataset
from scno.evaluate import evaluate, gradient_consistency, resolution_study
from scno.models.exact import ExactOperator


def test_exact_operator_has_zero_error(h5_file):
    report = evaluate(ExactOperator(), CheetahDataset(h5_file), batch_size=2)
    assert report["potential_rel_l2"]["max"] < 1e-4
    assert report["force_rel_l2"]["max"] < 1e-4


def test_exact_operator_gradients_match(h5_file):
    report = gradient_consistency(ExactOperator(), CheetahDataset(h5_file), num_samples=2)
    assert report["cosine_similarity_min"] > 0.999
    assert abs(report["magnitude_ratio_mean"] - 1.0) < 1e-3


def test_exact_operator_resolution_study(h5_file):
    report = resolution_study(ExactOperator(), h5_file, nondim=True, factors=(1, 2))
    assert report["downsample_2"]["grid"] == (8, 8, 8)
    # Coarsening averages phi_hat over blocks, which is not the exact solution on the
    # coarse grid -- the error is small but not zero.
    assert report["downsample_1"]["potential_rel_l2_mean"] < 1e-4
    assert torch.isfinite(torch.tensor(report["downsample_2"]["force_rel_l2_mean"]))
