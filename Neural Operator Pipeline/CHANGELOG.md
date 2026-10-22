# Changelog

## v0.1.0 [🚧 Work in Progress]

### 🚨 Breaking Changes

- Replace `scno.cheetah_element` with `scno.accelerator.NeuralSpaceChargeKick`, a subclass of the unchanged `SpaceChargeKick` that takes a neuralop `BaseModel` and overrides only `_solve_poisson_equation`
- Train only with the neuralop `Trainer` in `scripts/train_model.py` and evaluate with `scripts/evaluate_model.py`; remove the own training loop, `make_loaders`, `scripts/smoke_train.py`, `scripts/load_model.py`, `scripts/smoke_test.py` and the `main` functions of `scno.train` and `scno.evaluate`
- Remove all YAML configs (`configs/`) and the `scno.data.generate` CLI; all settings are the constants at the top of `scripts/train_model.py`, which the other scripts import, and missing data is generated on the first training run
- Write results only to `models/<MODEL_NAME>.pth` (neuralop `state_dict()` with its `_metadata`), `figures/` and `tables/` (history and evaluation as CSV) instead of `outputs/`
- Use neuralop's `FNO` argument names for the model settings and remove the own `FNO3d`
- Rename `notebooks/smoke_train.ipynb` to `notebooks/train_model.ipynb`
- Fit the Gaussian output normalizer only for `nondim=False` by default (`encode_output=None`), so nondimensionalized training has a single scaling layer
- Replace `build_loss` with `SobolevLoss`, built on neuralop's `LpLoss` and `FiniteDiff` and summed over the batch as the `Trainer` expects

### 🚀 Features

- Add tests for `NeuralSpaceChargeKick` against `SpaceChargeKick` with the exact operator (tracking and gradients) and for the `ExactOperator` control

### 🐛 Bug fixes

- Evaluate on the dataset's `x`/`y` through the data processor, so `nondim=False` models are no longer evaluated on `rho_hat`
- Undo the output normalizer through a single `predict` path that sets the data processor to evaluation mode
- Update `DEFAULT_RANGES` to the bounds of the former configs, which had diverged
- Stretch the cell size by gamma without an in-place write in `stretched_cell_size`, so gradients w.r.t. the beam energy flow through the lattice element

### 🐆 Other

- Add project guidelines for AI agents (`AGENTS.md`)
- Remove `utils/physics.py` in favour of Cheetah's `SpaceChargeKick` methods; `ExactOperator` and `scripts/verify_physics.py` use `models.exact.solve_poisson_on_grid` (Cheetah's integrated Green function with the padding and FFT mirrored, guarded by a test against `SpaceChargeKick`)
- Remove the unused `dimensionalize_fields`, `density_from_hat`, `coordinate_channels`, `build_input`, `space_charge_field`, `load_config`, the `format` file attribute and the `pyyaml` dependency
- `scripts/verify_physics.py` measures with the evaluation's `field_error`, `scripts/inspect_scaling.py` reads the training file of the `train_model.py` settings
- Move the tests to `tests/` with shared fixtures in `tests/conftest.py`
- Translate comments, docstrings and README to English
