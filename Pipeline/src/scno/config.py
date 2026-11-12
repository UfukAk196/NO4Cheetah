"""Paths of the pipeline and the global seed.

Hyperparameters and data settings are not here but at the top of the run scripts
(``exec/train_model.py``, ``exec/tune_model.py``), where they are used.

Every run gets an ID ``<date>_<NN>`` (``scno.utils.common.new_run_id``); its outputs go
only to ``outputs/``: ``figures/<RUN>/`` and ``tables/<RUN>/`` (training, evaluation) or
``studies/optuna/<RUN>/`` (Optuna study). The model is saved flat in ``models/`` and
overwritten by every training run.
"""

from pathlib import Path

# ponytail: the pipeline folder, found from this file -- correct for the editable
# install (`pip install -e .`) the project uses; a regular install would point into
# site-packages (then make it an environment variable).
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"  # the HDF5 files of generate.py
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

# Global Seed
SEED = 0

# Data settings
DATA = {
    "n_train": 10000,
    "batch_size": 64,
    "n_tests": [1250, 1250],
    "train_resolution": 32,
    "test_resolutions": [32, 64],  # tuning only needs the training resolution
    "test_batch_sizes": [32, 32],
    "seed": SEED,
    # Beam parameters added to rho as constant input channels (``scno.data.dataset.FEATURES``);
    # () = rho only. Ablation sets, e.g.: ("gamma",), ("log_sigma_x", "log_sigma_y"),
    # ("log_sigma_x", "log_sigma_y", "log_sigma_tau"),
    # ("log_sigma_x", "log_sigma_y", "log_sigma_tau", "gamma")
    "input_params": ("log_sigma_x", "log_sigma_y", "log_sigma_tau", "gamma"),
}