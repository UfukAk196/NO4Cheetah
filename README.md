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
is one overridden method: `NeuralSpaceChargeKick` subclasses Cheetah's unchanged
`SpaceChargeKick` and replaces only its `_solve_poisson_equation`.

> **Status:** work in progress. The pipeline from data generation through training to
> evaluation runs, and the element tracks in the lattice. Hyperparameters and beam
> parameter bounds are still open.

---

## Contents of the repo

| Folder | Content |
|---|---|
| [`Neural Operator Pipeline/`](Neural%20Operator%20Pipeline/) | **The code**: Python package `scno`, scripts, notebooks, models, figures, tables |
| [`Konzeptionierung/`](Konzeptionierung/) | Architecture and interface drafts, profiling of the SC kick, text drafts of the thesis |
| [`Projektplanung/`](Projektplanung/) | Work plan, scheduling |
| [`Quellen/`](Quellen/) | Literature |
| [`bsc-msc-template/`](bsc-msc-template/) | LaTeX template of the thesis |

Cheetah itself stays unchanged (installed from the fork `UfukAk196/cheetah`, identical
to upstream), so the element can later move into Cheetah as it is.

### The code in detail

```
Neural Operator Pipeline/
├── pyproject.toml            the only package list (package "scno")
├── environment.yml           conda, CPU
├── environment-gpu.yml       conda, CUDA (e.g. the Maxwell cluster)
├── requirements.txt          pip only (Colab, venv)
├── CHANGELOG.md              changes, in Cheetah's format
├── src/scno/
│   ├── accelerator/
│   │   └── neural_space_charge_kick.py   NeuralSpaceChargeKick: the FNO in the lattice
│   ├── utils/
│   │   ├── normalization.py  nondimensionalization, please read first
│   │   ├── losses.py         Sobolev loss (field + gradient), Poisson residual
│   │   └── common.py         seed, device, CSV tables
│   ├── data/
│   │   ├── sampling.py       draw beam parameters (log-uniform / uniform)
│   │   ├── fields.py         (ρ, φ) directly from Cheetah's SpaceChargeKick
│   │   ├── generate.py       write HDF5 datasets
│   │   └── dataset.py        CheetahDataset, load_cheetah_sc
│   ├── models/
│   │   └── exact.py          the exact solver as a "model" (control)
│   ├── train.py              HistoryTrainer (neuralop Trainer with loss history)
│   └── evaluate.py           accuracy, resolution study, gradient fidelity, plots
├── scripts/
│   ├── train_model.py        ALL settings + training run
│   ├── evaluate_model.py     evaluation run of the saved model
│   ├── verify_physics.py     Cheetah's solver vs. charged sphere
│   └── inspect_scaling.py    spread of phi_hat in the training data
├── tests/                    pytest
├── notebooks/                cheetah_test, neuralop_test, train_model (Colab launcher)
├── data/                     generated HDF5 datasets (git-ignored)
├── models/                   trained model state dicts (.pth)
├── figures/                  plots (loss curves, predictions)
└── tables/                   CSV tables (history, evaluation) and profiling results
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
changes to `pyproject.toml`, simply run `pip install -e ".[dev]"` again. To work
against a local Cheetah clone instead, install it on top:
`pip install -e "../../Cheetah repos/cheetah-accelerator"`.

**Google Colab / plain pip**

```bash
git clone https://github.com/UfukAk196/NO4Cheetah.git
cd "NO4Cheetah/Neural Operator Pipeline"
pip install -r requirements.txt              # torch comes with Colab
```

Main dependencies: `torch>=2.2`, `cheetah-accelerator` (unchanged, from the fork),
`neuraloperator==2.0.0`, `h5py`.

---

## Quick start

```bash
# 1. Is the reference solver right? (charged sphere, analytical)        ~30 s
python scripts/verify_physics.py

# 2. Unit tests, incl. the element against SpaceChargeKick
pytest

# 3. Train, then evaluate
python scripts/train_model.py
python scripts/evaluate_model.py
```

---

## Pipeline step by step

All settings (data, model, loss, training) are the constants at the top of
[`scripts/train_model.py`](Neural%20Operator%20Pipeline/scripts/train_model.py);
`evaluate_model.py` and `inspect_scaling.py` import them. For an experiment, change them
there and give the run its own `MODEL_NAME`, so that numbers and settings stay together.

### 1. Data

Generated automatically on the first training run if files are missing. One HDF5 file per
resolution in `data/`, i.e. `data/cheetah_sc_train_{res}.h5` and
`data/cheetah_sc_test_{res}.h5`. The files hold the physical fields ρ [C/m³] and φ [V];
the nondimensionalization happens on loading. Every sample has a unique seed, so train
and test never overlap. Drawn are `sigma_x`, `sigma_y`, `sigma_tau`, `energy`,
`total_charge` and `n_particles` within the bounds of
`scno.data.sampling.DEFAULT_RANGES` (overridable with `DATA["ranges"]`).

> The bounds are still **example values** (TODO in `sampling.py`).

### 2. Train

```bash
python scripts/train_model.py
```

Trains with the neuralop `Trainer` and writes only:

- `models/<MODEL_NAME>.pth`: state dict of the final model. neuralop's `state_dict()`
  carries the init kwargs as `_metadata`, so this one file is enough to rebuild the model.
- `figures/loss_curves_<MODEL_NAME>.png`
- `tables/<MODEL_NAME>_history.csv`: losses per epoch (eval columns every `EVAL_INTERVAL` epochs)

### 3. Evaluate

```bash
python scripts/evaluate_model.py
```

Writes `tables/<MODEL_NAME>_evaluation.csv` and the prediction plots per test resolution
to `figures/`:

- **Accuracy:** relative L2 error of φ̂ and of the field ∇φ̂
- **Resolution study:** zero-shot on a different grid than in training
- **Gradient fidelity:** cosine between the gradients through the model and through the exact solver

### 4. Into the lattice

```python
import torch
from evaluate_model import load_model   # run from scripts/, or copy the 4 lines
from scno.accelerator import NeuralSpaceChargeKick

operator = load_model("models/fno_cheetah_space_charge.pth", torch.device("cpu"))
kick = NeuralSpaceChargeKick(operator=operator, effect_length=torch.tensor(0.3))
```

The element is a drop-in replacement for `SpaceChargeKick` (also on vectorized beams)
and takes models trained with `DATA["nondim"] = True`.

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

The nondimensionalization can be switched off for ablations (`DATA["nondim"] = False`):
the model then learns ρ [C/m³] → φ [V], and only in that case a Gaussian normalizer is
fitted on the output. How much `phi_hat` still spreads is measured by
`scripts/inspect_scaling.py`.

### The loss must know the gradient

The kick is the gradient of the potential. A good L2 on φ does not yet mean good
forces, because differentiation amplifies high frequencies. The default is therefore a
**Sobolev loss** (field + gradient, `LOSS["gradient_weight"]`). Optionally the Poisson
residual is added (`LOSS["physics_weight"] > 0`).

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
7. **Physics residual** (`LOSS["physics_weight"]`) with little vs. much data

---

## Working rules

- First `verify_physics.py`, then train. If the teacher computes wrongly, the student
  reliably learns nonsense.
- First run on tiny data (few samples and epochs in `train_model.py`), then scale.
- If an evaluation looks odd, first send `ExactOperator` through the same evaluation.
  If that does not give an error ≈ 0, the network is not the problem.
- Give every experiment its own `MODEL_NAME` in `train_model.py`, so that numbers and
  settings stay together.

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
