# Plan: standardization without centering (RMS scaling) for rho and phi, plus one clear overview

## Context
After the lean-out (now on `main`), there is **only one** scaling step left in the code. The docs still describe the old nondimensionalization (`fields.py`, `sampling.py`, `normalization.py`, which no longer exist), and that makes the flow hard to follow. Goal: (1) replace centering with **standardization without centering**, z = x / RMS(x), for **rho and phi**, and (2) make the docs describe exactly this one step and nothing else.

## Overview: where scaling happens today (the full chain)
1. `generate.py` writes physical rho [C/m³] and phi [V] to `.h5`. **No scaling.**
2. `dataset.py:117-133` `CheetahDataset` fits `UnitGaussianNormalizer` on the **training file** (channel-wise means one scalar per field over all samples and grid points). Default today: only phi (`encode_output=True`, `encode_input=False`).
3. neuralop 2.0.0 `DefaultDataProcessor` (checked against the tag source):
   - `preprocess`: x is always encoded with `in_normalizer`. y is encoded **only in `training` mode**.
   - `postprocess`: the model output is decoded **only in eval mode**.
   - So training compares z-space with z-space, and eval/`Trainer` test losses compare physical phi with physical phi.
4. `train_model.py:146` saves the fitted processor (`torch.save`). `evaluate_model.py:49` loads it, and `evaluate.predict` (`evaluate.py:44`) runs eval-mode pre/postprocess, so the results are in V.

neuralop's `UnitGaussianNormalizer` **always centers**: `transform = (x - mean)/(std + eps)`, `inverse = x*(std + eps) + mean`, with `eps = 1e-7`. It has no flag for this. `fit()` sets the buffers `mean` and `std` (`torch.std`, Bessel-corrected) and the attribute `squared_mean = mean(x²)`.

## Changes

### 1. `src/scno/data/dataset.py`
- New small class next to `CheetahH5Dataset`:
  ```python
  class RMSNormalizer(UnitGaussianNormalizer):
      """Standardization without centering: z = x / rms, rms = sqrt(mean(x^2)) over the
      reduce dims. Fits like UnitGaussianNormalizer, then sets mean = 0, std = rms, so the
      inherited transform/inverse_transform (and the DataProcessor) work unchanged."""
      def fit(self, data_batch):
          super().fit(data_batch)
          self.mean = torch.zeros_like(self.mean)
          self.std = self.squared_mean.sqrt()
  ```
  Assigning to registered buffers updates them, so they're included when `.to(device)` is called and when the processor is `torch.save`d.
- `_fit_normalizer` uses `RMSNormalizer` instead of `UnitGaussianNormalizer`. The `reduce_dims` logic stays the same.
- `encode_input` defaults to `True` in `CheetahDataset` **and** `load_cheetah_sc`.
- Rewrite the comment block at lines 117-120 and the `encode_*`/`encoding` docstrings to describe z = x / RMS fitted on train, which fields it applies to, and when the DataProcessor encodes and decodes (points 2-3 above, briefly).

### 2. Small follow-ups
- `exec/train_model.py:40-41`: update the comment to "rho, phi → x / RMS over the training samples and grid".
- `exec/evaluate_model.py:56`: `load_cheetah_sc(**DATA, encode_input=False, encode_output=False)`. The processor comes from the saved file anyway, so this avoids refitting for nothing.
- `exec/tests/test_dataset.py::test_normalizer_shapes`: add `mean == 0` and `mean(transform(x_train)**2) ≈ 1` (rtol 1e-4) for in and out.

### 3. Remove stale docs (the overview problem)
- `src/scno/data/__init__.py`: the work packages G1-G5 now list only `generate.py` (write HDF5) and `dataset.py` (CheetahDataset, RMS scaling, `load_cheetah_sc`).
- `src/scno/utils/__init__.py`: drop "nondimensionalization".
- `README.md` (lines ~56, ~149, ~220-237): remove the `normalization.py` tree entry and the nondimensionalization section, and replace them with a short section "Scaling" containing the overview chain above.

### Not changed
`generate.py`, `evaluate.py`, `config.py`, `tune_model.py`. Old processors saved in `outputs/models` stay loadable (they're plain `UnitGaussianNormalizer`), but they are still centered. Only new runs use RMS.

### Known limitation (no code change)
A single global scale per field doesn't even out the spread **between samples**. rho and phi scale with charge and 1/σ³ across the drawn ranges, so they vary by orders of magnitude. If the loss stays dominated by a few large samples, the next step would be per-sample scaling, and that would be a separate task.

## Verification
1. `pytest exec/tests/test_dataset.py` passes, including the new RMS assertions.
2. A short check in the scratchpad: `ds = CheetahDataset(...)`, then `z = ds.data_processor.in_normalizer.transform(x_train)`. Expect `z.pow(2).mean() ≈ 1` and `inverse_transform(z) ≈ x_train`, and the same for phi.
3. Run `python exec/train_model.py` briefly (small `epochs`) and confirm the first epochs run, the eval losses are reported in V, and `final_model_data_processor.pt` loads in `evaluate_model.py`.
