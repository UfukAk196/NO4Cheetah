"""All settings of the pipeline runs: paths, data, model, loss and training.

The single place to change an experiment. ``exec/train_model.py``,
``evaluate_model.py``, ``tune_model.py``, ``inspect_scaling.py`` and the tests import
these constants, so that training, evaluation, tuning and diagnosis see exactly the same
setup.

Every run gets an ID ``<date>_<NN>`` (``scno.utils.common.new_run_id``); its outputs go
only to ``outputs/``: ``models/<MODEL_NAME>_<RUN>.pth``, ``figures/<RUN>/`` and
``tables/<RUN>/`` (training, evaluation) or ``studies/optuna/<RUN>/`` (Optuna study);
Weights & Biases logs go to ``outputs/studies/wandb/``.
"""

from pathlib import Path

from scno.utils.normalization import INPUT_CHANNELS

# ponytail: the pipeline folder, found from this file -- correct for the editable
# install (`pip install -e .`) the project uses; a regular install would point into
# site-packages (then make it an environment variable).
ROOT = Path(__file__).resolve().parents[2]
MODEL_NAME = "fno_cheetah_space_charge"
OUTPUTS_DIR = ROOT / "outputs"  # everything a run writes
MODELS_DIR = OUTPUTS_DIR / "models"
FIGURES_DIR = OUTPUTS_DIR / "figures"
TABLES_DIR = OUTPUTS_DIR / "tables"
STUDIES_DIR = (
    OUTPUTS_DIR / "studies"
)  # wandb.init(dir=STUDIES_DIR) writes to studies/wandb/
OPTUNA_DIR = STUDIES_DIR / "optuna"
RUN_DIRS = (
    MODELS_DIR,
    FIGURES_DIR,
    TABLES_DIR,
    OPTUNA_DIR,
)  # where run IDs are looked up

SEED = 0
DEVICE = "auto"  # auto | cpu | cuda | mps

# Keyword arguments of scno.data.dataset.load_cheetah_sc.
DATA = {
    "data_root": ROOT / "data",
    "seed": 0,
    "nondim": False,  # True: rho_hat -> phi_hat, False: rho [C/m^3] -> phi [V] (ablation)
    # "encode_output": unset = not nondim (Gaussian normalizer only for the raw ablation)
    "n_train": 16,  # TODO: target 5k-50k samples
    "n_tests": [4, 4],
    "train_resolution": 32,
    "test_resolutions": [32, 64],
    "batch_size": 4,
    "test_batch_sizes": [4, 4],
    "encode_input": True,
    "in_channel_type": "gamma",  # "gamma", "normalized_extents" or "only rho" (the input channels of the model)
    "num_workers": 0,  # keep at 0 on Windows
    # "ranges": override of scno.data.sampling.DEFAULT_RANGES, same shape
}

# ================= MODEL, LOSS, and TRAINING hyperparameters according to optuna study from 2026-10-28   =================

# Keyword arguments of neuralop.models.TFNO.
MODEL = {
    "n_modes": (5, 5, 5,),  # kept Fourier modes per axis; less according to optuna research from 2026-10-28.    TODO: try more according to best-practices (e.g 16)
    # Derived from DATA["in_channel_type"] below, so the model always matches the data.
    "in_channels": INPUT_CHANNELS[DATA["in_channel_type"]],
    "out_channels": 1,  # phi_hat                                                                               TODO: try unnormalized/ -scaled
    "hidden_channels": 32,
    "n_layers": 3,
    "domain_padding": 0.125,  # padding against the open boundaries
    # "factorization": "Tucker", # default value Tucker for TFNO
    # "rank": 0.1,   # default value 0.1
}

# Keyword arguments of scno.utils.losses.SobolevLoss.
LOSS = {  # TODO: einfachen MSE/ MAE loss probieren, wobei Sobolev bereits sinn macht, da Problem in f-domain
    "gradient_weight": 0.24,  # weight of the gradient error -- the kick is a gradient
    "physics_weight": 1.0,  # > 0 adds the Poisson residual (PINN part)
}

TRAINING = {
    "epochs": 10000,
    "learning_rate": 2.0e-4,
    "weight_decay": 4.0e-6,
    "scheduler": False,
}

EVAL_INTERVAL = 100  # epochs between evaluations on the test resolutions
WANDB = False  # log to Weights & Biases (project "space-charge-neural-operator")

# Optuna search of exec/tune_model.py (search space: tune_model.suggest).
N_TRIALS = 100  # finished (complete + pruned) trials over all workers
TUNE_EPOCHS = 200  # epochs of a trial that is not pruned
TUNE_EVAL_INTERVAL = 10  # epochs between evaluations = pruning checks


def model_path(run: str) -> Path:
    """``models/<MODEL_NAME>_<RUN>.pth``: the state dict of run ``run``."""
    return MODELS_DIR / f"{MODEL_NAME}_{run}.pth"
