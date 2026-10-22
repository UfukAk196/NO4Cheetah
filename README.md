# NO4Cheetah — Neural Operator for Space Charge in Cheetah

Master's thesis: a **differentiable neural operator (FNO)** that replaces the FFT
Poisson solver in Cheetah's
[`SpaceChargeKick`](https://github.com/desy-ml/cheetah), in such a way that not only
the potential is right but the **gradients** through the tracking also stay usable.

```
          particles ──► CIC deposition ──► ρ on the grid
                                              │
               ┌──────────────────────────────┴──────────────────────┐
               │  classical: IGF convolution via FFT (Hockney), O(N log N) │
               │  here:      neural operator  ρ̂ ↦ φ̂                       │  ◄── replaced
               └──────────────────────────────┬──────────────────────┘
                                              │
                    φ ──► central differences ──► E field ──► momentum kick
```

**Only the Poisson solve** is replaced. Deposition, gradient and interpolation stay
Cheetah's classical torch code: cheap, exact and already differentiable. In code this
is one overridden method: `NeuralSpaceChargeKick` subclasses `SpaceChargeKick` and
replaces only its grid-level solve `_solve_poisson_on_grid`.

> **Status:** work in progress. The pipeline from data generation through training to
> evaluation runs, and the element tracks in the lattice. Hyperparameters and beam
> parameter bounds are still open.

---

## Contents of the repo

| Folder | Content |
|---|---|
| [`Neural Operator Pipeline/`](Neural%20Operator%20Pipeline/) | **The code**: Python package `scno`, scripts, configs, notebooks, figures |
| [`Konzeptionierung/`](Konzeptionierung/) | Architecture and interface drafts, profiling of the SC kick, text drafts of the thesis |
| [`Projektplanung/`](Projektplanung/) | Work plan, scheduling |
| [`Quellen/`](Quellen/) | Literature |
| [`bsc-msc-template/`](bsc-msc-template/) | LaTeX template of the thesis |

The code depends on a Cheetah branch, `space-charge-grid-solve` in the fork
`UfukAk196/cheetah`, which splits the `SpaceChargeKick` Poisson solve into
`_grid_geometry`, `_array_rho` and the grid-level `_solve_poisson_on_grid`.

### The code in detail

```
Neural Operator Pipeline/
├── pyproject.toml            the only package list (package "scno")
├── environment.yml           conda, CPU
├── environment-gpu.yml       conda, CUDA (e.g. the Maxwell cluster)
├── requirements.txt          pip only (Colab, venv)
├── CHANGELOG.md              changes, in Cheetah's format
├── configs/
│   ├── default.yaml          data, model, loss, training (copy for experiments)
│   └── smoke.yaml            the same in miniature
├── src/scno/
│   ├── accelerator/
│   │   └── neural_space_charge_kick.py   NeuralSpaceChargeKick: the FNO in the lattice
│   ├── utils/
│   │   ├── normalization.py  nondimensionalization, please read first
│   │   ├── losses.py         Sobolev loss (field + gradient), Poisson residual
│   │   └── common.py         seed, device, config
│   ├── data/
│   │   ├── sampling.py       draw beam parameters (log-uniform / uniform)
│   │   ├── fields.py         (ρ, φ) directly from Cheetah's SpaceChargeKick
│   │   ├── generate.py       write HDF5 datasets (CLI)
│   │   └── dataset.py        CheetahDataset, load_cheetah_sc
│   ├── models/
│   │   └── exact.py          the exact solver as a "model" (control)
│   ├── train.py              training with the neuralop Trainer
│   └── evaluate.py           accuracy, resolution study, gradient fidelity, plots
├── tests/                    pytest
├── scripts/                  verify_physics, inspect_scaling
├── notebooks/                cheetah_test, neuralop_test, smoke_train
├── figures/                  generated plots
└── tables/                   profiling results
```

---

## Installation

All commands run in the folder **`Neural Operator Pipeline/`**. Python 3.10–3.12.

**Conda (recommended, local)**

```bash
cd "Neural Operator Pipeline"
conda env create -f environment.yml          # CPU
# conda env create -f environment-gpu.yml    # GPU; match cuda-version to `nvidia-smi`
conda activate no4cheetah                    # or no4cheetah-gpu
```

The environment installs `scno` in editable mode (`pip install -e ".[dev]"`). After
changes to `pyproject.toml`, simply run `pip install -e ".[dev]"` again. As long as
the Cheetah branch is not pushed yet, install the local clone on top:
`pip install -e "../../Cheetah repos/cheetah-accelerator"`.

**Google Colab / plain pip**

```bash
git clone https://github.com/UfukAk196/NO4Cheetah.git
cd "NO4Cheetah/Neural Operator Pipeline"
pip install -r requirements.txt              # torch comes with Colab
```

Main dependencies: `torch>=2.2`, `cheetah-accelerator` (fork branch above),
`neuraloperator==2.0.0`, `h5py`.

---

## Quick start

```bash
# 1. Is the reference solver right? (charged sphere, analytical)        ~30 s
python scripts/verify_physics.py

# 2. Unit tests, incl. the element against SpaceChargeKick
pytest

# 3. Small end-to-end run with the neuralop Trainer, then evaluation + plots per resolution
python -m scno.train --config configs/smoke.yaml
python -m scno.evaluate --run outputs/smoke --plot
```

The smoke run generates the data under `data/` if needed, writes checkpoints and loss
curves to `outputs/smoke/` and plots to `figures/`.

---

## Pipeline step by step

### 1. Generate data

```bash
python -m scno.data.generate --config configs/default.yaml
```

Writes one HDF5 file per resolution to `data/`, i.e. `data/cheetah_sc_train_{res}.h5`
and `data/cheetah_sc_test_{res}.h5` (folder set by `data.data_root` in the config). The
files hold the physical fields ρ [C/m³] and φ [V]; the nondimensionalization happens on
loading. Every sample has a unique seed, so train and test never overlap. Drawn are
`sigma_x`, `sigma_y`, `sigma_tau`, `energy`, `total_charge` and `n_particles` within
the bounds of `scno.data.sampling.DEFAULT_RANGES` (overridable with `data.ranges`).

> The bounds are still **example values** (TODO in `sampling.py`).

### 2. Train

```bash
python -m scno.train --config configs/default.yaml
```

Generates missing data and trains with the neuralop `Trainer`. Single values can be
overridden without touching the config:

```bash
python -m scno.train --config configs/default.yaml \
    --override training.epochs=300 model.hidden_channels=32 loss.gradient_weight=2.0
```

`output_dir` (default `outputs/run_default/`) receives the resolved `config.yaml`, the
neuralop checkpoints `best_model_*` (best h1 loss on the first test resolution) and
`model_*` (final), `history.json` and the loss curves. Load a model with
`FNO.from_checkpoint(output_dir, "best_model")`.

### 3. Evaluate

```bash
python -m scno.evaluate --run outputs/run_default                 # test file at the training resolution
python -m scno.evaluate --run outputs/run_default --data data/cheetah_sc_test_64.h5 --plot
```

Reports to `<run>/evaluation.json`:

- **Accuracy:** relative L2 error of φ̂ and of the field ∇φ̂
- **Resolution study:** zero-shot on a different grid than in training
- **Gradient fidelity:** cosine between the gradients through the model and through the exact solver

### 4. Into the lattice

```python
import torch
from neuralop.models import FNO
from scno.accelerator import NeuralSpaceChargeKick

operator = FNO.from_checkpoint("outputs/run_default", "best_model")
kick = NeuralSpaceChargeKick(operator=operator, effect_length=torch.tensor(0.3))
```

The element is a drop-in replacement for `SpaceChargeKick` (also on vectorized beams)
and takes models trained with `nondim: true`.

---

## The two design decisions

### Nondimensionalization

Charges spread over decades, cell sizes over two, γ over three. So what is known
analytically is removed analytically (details in
[`normalization.py`](Neural%20Operator%20Pipeline/src/scno/utils/normalization.py)):

```
L       = (Ex · Ey · Ez)^(1/3)        box edges of the γ-stretched box (rest frame)
rho_hat = rho · L³ / Q
phi_hat = phi · 4π ε0 · L / Q         ⇒   Δ phi_hat = −4π · rho_hat
```

The network gets 4 channels: `rho_hat` and the three constant aspect ratios
`log(E_i / L)`. One model thus covers all beam sizes, charges and energies.

**L depends on the box size, not on the cell size.** Otherwise the same physical
density maps to a different `phi_hat` depending on the resolution, and zero-shot
super-resolution collapses. Measured: 142 % error on 32³ (cell reference) versus 3 %
(box reference).

The nondimensionalization can be switched off for ablations (`data.nondim: false`):
the model then learns ρ [C/m³] → φ [V], and only in that case a Gaussian normalizer is
fitted on the output. How much `phi_hat` still spreads is measured by
`scripts/inspect_scaling.py`.

### The loss must know the gradient

The kick is the gradient of the potential. A good L2 on φ does not yet mean good
forces, because differentiation amplifies high frequencies. The default is therefore a
**Sobolev loss** (field + gradient, `loss.gradient_weight`). Optionally the Poisson
residual is added (`loss.physics_weight > 0`).

---

## First results

Prediction after training on 16³, evaluated zero-shot on 64³:

<p align="center">
  <img src="Neural%20Operator%20Pipeline/figures/phi_hat_gt_vs_pred_after_training_res64.png" width="80%">
</p>

Early calibration run from the skeleton phase (96 samples, 16³, 60 epochs, the former
own FNO-3D, CPU):

| Quantity | Value |
|---|---|
| Reference solver vs. analytical sphere (64³) | 0.5 % |
| Potential, rel. L2 | 1.3 % |
| Forces, rel. L2 | 2.7 % |
| Potential, zero-shot 32³ | 3.0 % |
| Forces, zero-shot 32³ | 23 % |
| Gradient fidelity (cosine to the exact solver) | **0.06** |

The last row is the core of the thesis. A model that hits the potential to 1 % can
still have a practically useless Jacobian. Cross-check: `models/exact.py` reaches an
error of 8·10⁻⁸ and a cosine of 1.000000 in the same evaluation, so the measurement
setup is right.

---

## Open points

1. **Raise gradient fidelity:** Sobolev training with a higher gradient weight,
   learning the field directly (3 output channels), Jacobian matching, Green structure
   in the architecture
2. **Remaining scale spread** of `phi_hat` (γ stretching), e.g. with a separate head
   for `log|phi_hat|`
3. **Real beam parameter bounds** and more data (order of 5k–50k samples), plus more
   realistic distributions from Cheetah runs
4. **Ablation against a CNN/U-Net** with the same parameter count, especially when
   changing the resolution
5. **Measure runtime honestly:** forward *and* backward pass, CPU vs. GPU, large grids
6. **Long-range test in the lattice** with a trained operator (`NeuralSpaceChargeKick`)
7. **Physics residual** (`loss.physics_weight`) with little vs. much data

---

## Working rules

- First `verify_physics.py`, then train. If the teacher computes wrongly, the student
  reliably learns nonsense.
- First run on tiny data (`configs/smoke.yaml`), then scale.
- If an evaluation looks odd, first send `ExactOperator` through the same evaluation.
  If that does not give an error ≈ 0, the network is not the problem.
- Copy the config for experiments instead of editing it, so that numbers and settings
  stay together.

---

## Literature

- Li et al., *Fourier Neural Operator for Parametric PDEs* (ICLR 2021)
- Kovachki et al., *Neural Operator: Learning Maps Between Function Spaces* (JMLR 2023)
- Azizzadenesheli et al., *Neural Operators for Accelerating Scientific Simulations and Design* (Nat. Rev. Phys. 2024)
- Qiang et al., PRST-AB 9, 044204 (2006): IGF method, implemented in Cheetah's `SpaceChargeKick`
- Kaiser et al., *Cheetah*: [github.com/desy-ml/cheetah](https://github.com/desy-ml/cheetah)
- [neuraloperator.github.io](https://neuraloperator.github.io), in particular the Darcy flow example as an elliptic neighbour of this problem

---

## License

[GPL-3.0](LICENSE)
