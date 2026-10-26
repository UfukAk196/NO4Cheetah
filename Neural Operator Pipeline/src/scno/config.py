"""All settings of the pipeline runs: paths, data, model, loss and training.

The single place to change an experiment. ``scripts/train_model.py``,
``evaluate_model.py``, ``tune_model.py``, ``inspect_scaling.py`` and the tests import
these constants, so that training, evaluation, tuning and diagnosis see exactly the same
setup.

Every run gets an ID ``<date>_<NN>`` (``scno.utils.common.new_run_id``); its outputs go
only to ``models/<MODEL_NAME>_<RUN>.pth``, ``figures/<RUN>/`` and ``tables/<RUN>/``.
"""

from pathlib import Path

# ponytail: the pipeline folder, found from this file -- correct for the editable
# install (`pip install -e .`) the project uses; a regular install would point into
# site-packages (then make it an environment variable).
ROOT = Path(__file__).resolve().parents[2]
MODEL_NAME = "fno_cheetah_space_charge"
MODELS_DIR = ROOT / "models"
FIGURES_DIR = ROOT / "figures"
TABLES_DIR = ROOT / "tables"
RUN_DIRS = (MODELS_DIR, FIGURES_DIR, TABLES_DIR)  # where run IDs are looked up

SEED = 0
DEVICE = "auto"  # auto | cpu | cuda | mps

# Keyword arguments of scno.data.dataset.load_cheetah_sc.
DATA = {
    "data_root": ROOT / "data",
    "seed": 0,
    "nondim": True,  # True: rho_hat -> phi_hat, False: rho [C/m^3] -> phi [V] (ablation)
    # "encode_output": unset = not nondim (Gaussian normalizer only for the raw ablation)
    "n_train": 16,  # TODO: target 5k-50k samples
    "n_tests": [4, 4, 4],
    "train_resolution": 16,
    "test_resolutions": [16, 32, 64],
    "batch_size": 4,
    "test_batch_sizes": [4, 4, 4],
    "num_workers": 0,  # keep at 0 on Windows
    # "ranges": override of scno.data.sampling.DEFAULT_RANGES, same shape
}

# Keyword arguments of neuralop.models.FNO.
MODEL = {
    "n_modes": (8, 8, 8),  # kept Fourier modes per axis
    "in_channels": 4,  # rho_hat + 3 aspect ratios log(E_i/L)
    "out_channels": 1,  # phi_hat
    "hidden_channels": 16,
    "n_layers": 3,
    "domain_padding": 0.25,  # padding against the open boundaries
    "factorization": None,  # e.g. "tucker" for fewer weights
    "rank": 1.0,
}

# Keyword arguments of scno.utils.losses.SobolevLoss.
LOSS = {
    "gradient_weight": 1.0,  # weight of the gradient error -- the kick is a gradient
    "physics_weight": 0.0,  # > 0 adds the Poisson residual (PINN part)
}

TRAINING = {
    "epochs": 1000,
    "learning_rate": 1.0e-4,
    "weight_decay": 1.0e-4,
}

EVAL_INTERVAL = 100  # epochs between evaluations on the test resolutions
WANDB = False  # log to Weights & Biases (project "space-charge-neural-operator")

# Optuna search of scripts/tune_model.py (search space: tune_model.suggest).
N_TRIALS = 100  # finished (complete + pruned) trials over all workers
TUNE_EPOCHS = 200  # epochs of a trial that is not pruned
TUNE_EVAL_INTERVAL = 10  # epochs between evaluations = pruning checks


def model_path(run: str) -> Path:
    """``models/<MODEL_NAME>_<RUN>.pth``: the state dict of run ``run``."""
    return MODELS_DIR / f"{MODEL_NAME}_{run}.pth"
