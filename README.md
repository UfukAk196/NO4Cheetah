# NO4Cheetah — Neural Operator for Space Charge in Cheetah

Master's thesis: a **differentiable neural operator** that is meant to replace the FFT
Poisson solver in Cheetah's
[`SpaceChargeKick`](https://github.com/desy-ml/cheetah), in such a way that not only
the potential is right but the **gradients** through the tracking also stay usable.

```
          particles ──► CIC deposition ──► ρ on the grid
                                              │
               ┌──────────────────────────────┴─────────────────────────────┐
               │  classical: IGF convolution via FFT (Hockney), O(N log N)  │
               │  here:      neural operator  ρ ↦ φ                        │  ◄── replaced
               └──────────────────────────────┬─────────────────────────────┘
                                              │
                    φ ──► central differences ──► E field ──► momentum kick
```

**Only the Poisson solve** is to be replaced. Deposition, gradient and interpolation
stay Cheetah's classical torch code: cheap, exact and already differentiable. The
eventual aim is one overridden method — a `SpaceChargeKick` subclass that replaces only
`_solve_poisson_equation` by the operator — so the element could later move into Cheetah
unchanged.

> **Status:** work in progress. This repo currently holds the **pipeline** that trains
> and evaluates that operator end to end: data generation from Cheetah, a
> [`neuraloperator`](https://neuraloperator.github.io) **TFNO**, and the evaluation of
> potential and force. The focus right now is the operator itself — the data
> representation (log + z-score, beam parameters as input channels) and the
> hyperparameters. The drop-in lattice element is **not** part of the package.

---

## Contents of the repo

| Folder | Content |
|---|---|
| [`Pipeline/`](Pipeline/) | **The code**: Python package `scno`, run scripts, Slurm scripts, tutorials, models, figures, tables |
| [`Konzeptionierung/`](Konzeptionierung/) | Architecture and interface drafts, profiling of the SC kick, text drafts of the thesis |
| [`Projektplanung/`](Projektplanung/) | Work plan, scheduling |
| [`Quellen/`](Quellen/) | Literature |
| [`bsc-msc-template/`](bsc-msc-template/) | LaTeX template of the thesis |

Cheetah itself stays unchanged (installed from the fork `UfukAk196/cheetah`, identical
to upstream), so the operator can later move into Cheetah as it is.

### The code in detail

```
Pipeline/
├── pyproject.toml            the only package list (package "scno")
├── environment.yml           conda environment "no4cheetah"
├── requirements.txt          pip only (Colab, venv): installs scno from pyproject ("-e .")
├── CHANGELOG.md              changes, in Cheetah's format
├── src/scno/
│   ├── config.py             paths and the global seed (hyperparameters live in the run scripts)
│   ├── data/
│   │   ├── generate.py       draw beam parameters and write HDF5 (ρ, φ) directly from
│   │   │                     Cheetah's SpaceChargeKick; DEFAULT_RANGES
│   │   └── dataset.py        CheetahDataset / load_cheetah_sc: log + z-score, beam
│   │                         parameters as constant input channels
│   ├── evaluate.py           rel. L2 of φ and of the force, plus plots (projections,
│   │                         central cuts, loss curves)
│   └── utils/
│       └── common.py         run IDs (<date>_<NN>), EPS, print helper
├── exec/
│   ├── train_model.py        training run (TFNO + neuralop Trainer); hyperparameters at the top
│   ├── evaluate_model.py     evaluation run of the saved model
│   ├── tune_model.py         distributed Optuna search (create / generate / work / report)
│   ├── tests/                pytest (dataset, generate, shapes)
│   └── tutorials/            cheetah_test, neuralop_test
├── scripts/                  Slurm scripts (train-script.sh, tune-script.sh) and their .out/.err logs
├── data/                     generated HDF5 datasets (git-ignored)
└── outputs/                  everything a run writes
    ├── models/               final_model_* (neuralop checkpoint), final_model_data_processor.pt,
    │                         checkpoints/ (Trainer's best "32_h1" checkpoint)
    ├── figures/<RUN>/        plots of training and evaluation (loss curves, predictions)
    ├── tables/<RUN>/         CSV tables (history, evaluation, samples)
    └── studies/
        ├── optuna/<RUN>/     Optuna journal, trials table and plots of a study
        └── wandb/            Weights & Biases runs (when Trainer logging is enabled)
```

---

## Installation

All commands run in the folder **`Pipeline/`**. Python 3.10–3.12.

**Conda (local)**

```bash
cd Pipeline
conda env create -f environment.yml          # env "no4cheetah"
conda activate no4cheetah
```

The environment installs `scno` in editable mode (`pip install -e ".[dev]"`). After
changes to `pyproject.toml`, simply run `conda env create -f environment.yml"` again.

**Google Colab / plain pip**

```bash
git clone https://github.com/UfukAk196/NO4Cheetah.git
cd NO4Cheetah/Pipeline
pip install -r requirements.txt              # installs scno ("-e ."); torch comes with Colab
```

Main dependencies: `cheetah-accelerator` (unchanged, from the fork),
`neuraloperator==2.0.0`, `torch` (pulled in by neuraloperator), `h5py`, `optuna`,
`wandb`. requirements.txt gets it from [`pyproject.toml`](Pipeline/pyproject.toml).


**Maxwell Cluster**
```bash
git clone https://github.com/UfukAk196/NO4Cheetah.git
cd NO4Cheetah/Pipeline
conda init
conda env create -f environment.yml          # env "no4cheetah"
conda activate no4cheetah
```

Same as above.

---

## Quick start

```bash
# 1. Unit tests (data generator, dataset, shapes)
pytest

# 2. Train, then evaluate (evaluate runs automatically at the end of training)
python exec/train_model.py
python exec/evaluate_model.py
```

---

## Pipeline step by step

Paths and the global seed are in [`src/scno/config.py`](Pipeline/src/scno/config.py);
the **hyperparameters** (model, optimizer, losses, epochs) are the constants at the top
of the run scripts [`exec/train_model.py`](Pipeline/exec/train_model.py) and
[`exec/tune_model.py`](Pipeline/exec/tune_model.py). Change them there for an experiment;
every run gets its own ID, so numbers and figures stay together.

### 1. Data

Generated automatically on the first training run if files are missing. One HDF5 file per
resolution in `data/`, i.e. `data/cheetah_sc_train_{res}.h5` and
`data/cheetah_sc_test_{res}.h5`. The files hold the physical fields ρ [C/m³] and φ [V]
exactly as they come out of Cheetah, plus a `params` group (one row per sample) that
makes every beam reproducible. Every sample uses its own seed, so train and test never
overlap. Drawn are `sigma_x`, `sigma_y`, `sigma_tau`, `energy`, `total_charge` and
`n_particles` within the bounds of `scno.data.generate.DEFAULT_RANGES` (overridable with
`DATA["ranges"]`).

> The bounds are still **example values** (TODO in `generate.py`); the log-uniform
> sampling of the sigmas is currently commented out (all parameters uniform).

### 2. Train

```bash
python exec/train_model.py      # on Maxwell: sbatch scripts/train-script.sh
```

Trains a `neuralop` **TFNO** with the neuralop `Trainer`. The model input is
`x = [ρ, *DATA["input_params"]]` — the density plus one constant channel per beam
parameter (`sigma_x, sigma_y, sigma_tau, gamma` by default); the target is φ. The run
writes:

- `outputs/models/final_model_*`: the best "32_h1" epoch as a neuralop checkpoint
  (state dict + init kwargs), rebuilt with `TFNO.from_checkpoint`
- `outputs/models/final_model_data_processor.pt`: the fitted normalizer, so evaluation
  uses exactly the training normalization
- `outputs/models/checkpoints/`: the Trainer's running best checkpoint
- `outputs/figures/<RUN>/loss_curves.png`
- `outputs/tables/<RUN>/history.csv`: losses per logged epoch (every `eval_interval`)

`<RUN>` is the run ID `<YYYY-MM-DD>_<NN>`. `train_model.py` calls `evaluate_model.py`
itself at the end, so the evaluation of the fresh model lands in the same run folders.

### Tune hyperparameters (optional, Maxwell)

```bash
bash scripts/tune-script.sh 4             # on a Maxwell login node: study + data job + 4 workers
python exec/tune_model.py report <RUN>    # trials table + Optuna plots of the study
```

`exec/tune_model.py` minimizes the H1 test loss at the training resolution with
TPE + Hyperband pruning; the workers share the study via a journal file
(`outputs/studies/optuna/<RUN>/<DATE>_optuna_journal_<NN>.log`). Its CLI is
`create` (new study), `generate` (generate the data once, before the workers),
`work <RUN>` (one worker) and `report [RUN]` (table + plots). Copy the best parameters
into `exec/train_model.py`.

### 3. Evaluate

```bash
python exec/evaluate_model.py [RUN]   # default: the latest run
```

Loads `outputs/models/final_model_*` and the training `data_processor`, then writes per
test resolution to `outputs/tables/<RUN>/` and `outputs/figures/<RUN>/`:

- **Accuracy:** relative L2 error of φ (`potential_rel_l2`) and of the force ∇φ
  (`force_rel_l2`), mean and max over the test samples (`evaluation.csv`, `samples.csv`)
- **Resolution study:** evaluated zero-shot on a grid (64³) different from training (32³)
- **Plots:** phase space of the first test beams, φ ground truth vs. prediction
  (projections and central cuts) on a shared scale

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
