# Changelog

## v0.1.0 [🚧 Work in Progress]

### 🚨 Breaking Changes

- Require the Cheetah fork branch `space-charge-grid-solve`, which splits the `SpaceChargeKick` Poisson solve into grid-level methods
- Replace `scno.cheetah_element` with `scno.accelerator.NeuralSpaceChargeKick`, a `SpaceChargeKick` subclass that takes a neuralop `BaseModel` and overrides only `_solve_poisson_on_grid`
- Train only with the neuralop `Trainer` (`python -m scno.train`); remove the own training loop, `make_loaders`, `scripts/smoke_train.py`, `scripts/load_model.py` and `scripts/smoke_test.py` in favour of `configs/smoke.yaml`
- Save and load models as neuralop checkpoints; `scno.evaluate` takes a training run (`--run`) instead of a `.pt` file
- Use neuralop's `FNO` argument names in the `model:` config and remove the own `FNO3d`
- Merge the dataset settings of `configs/generate.yaml` into the `data:` section of the training configs, whose keys are the arguments of `load_cheetah_sc`; remove `configs/generate.yaml` and `configs/sampling.yaml`
- Fit the Gaussian output normalizer only for `nondim=False` by default (`encode_output=None`), so nondimensionalized training has a single scaling layer
- Replace `build_loss` with `SobolevLoss`, built on neuralop's `LpLoss` and `FiniteDiff` and summed over the batch as the `Trainer` expects

### 🚀 Features

- Add tests for `NeuralSpaceChargeKick` against `SpaceChargeKick` with the exact operator (tracking and gradients) and for the `ExactOperator` control

### 🐛 Bug fixes

- Evaluate on the dataset's `x`/`y` through the data processor, so `nondim=False` models are no longer evaluated on `rho_hat`
- Undo the output normalizer through a single `predict` path that sets the data processor to evaluation mode
- Update `DEFAULT_RANGES` to the bounds of the configs, which had diverged

### 🐆 Other

- Add project guidelines for AI agents (`AGENTS.md`)
- Remove `utils/physics.py` in favour of Cheetah's `SpaceChargeKick` methods; `ExactOperator` and `scripts/verify_physics.py` use `_solve_poisson_on_grid`
- Remove the unused `dimensionalize_fields`, `density_from_hat`, `coordinate_channels`, `build_input`, `space_charge_field` and the `format` file attribute
- Move the tests to `tests/` with shared fixtures in `tests/conftest.py`
- Translate comments, docstrings and README to English
