"""
Trains the TFNO on the Cheetah space-charge dataset with the neuralop ``Trainer``.

All hyperparameters are set below, top to bottom: data, model, optimizer, scheduler,
losses, epochs. Missing data files are generated on the first run.

Every run gets an ID ``<date>_<NN>`` and writes to ``outputs/``:
    models/checkpoints/                    Trainer checkpoint of the best "32_h1" epoch
    models/final_model_*                   that best model (state dict + init kwargs)
    models/final_model_data_processor.pt   the fitted normalizer, for evaluate_model.py
    figures/<RUN>/loss_curves.png          loss curves (plot_losses)
    tables/<RUN>/history.csv               losses at the logged epochs
    + the evaluation of evaluate_model.py (trained vs. zero-shot resolution) in the
      same run folders

Call: python exec/train_model.py
"""

import sys
from collections import defaultdict

import pandas as pd
import torch
from neuralop import H1Loss, LpLoss, Trainer
from neuralop.models import TFNO
from neuralop.training import AdamW
from neuralop.utils import count_model_params

from evaluate_model import main as evaluate_run
from scno.config import FIGURES_DIR, MODELS_DIR, RUN_DIRS, TABLES_DIR, SEED, DATA
from scno.data.dataset import load_cheetah_sc
from scno.evaluate import plot_losses
from scno.utils.common import new_run_id, print_data_settings

run = new_run_id(*RUN_DIRS)
print(f"Run: {run}")
print_data_settings(DATA)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Data; input x = [rho, *DATA["input_params"]]. Normalization: every input channel and phi
# -> (field - mean) / std over the training samples and the grid (encode_input=True,
# encode_output=True, encoding="channel-wise" are the defaults of load_cheetah_sc).
train_loader, test_loaders, data_processor = load_cheetah_sc(
    n_train=DATA["n_train"],
    batch_size=DATA["batch_size"],
    n_tests=DATA["n_tests"],
    test_resolutions=DATA["test_resolutions"],
    test_batch_sizes=DATA["test_batch_sizes"],
    input_params=DATA["input_params"],
    seed=SEED,
    ranges=None,
)
data_processor = data_processor.to(device)

torch.manual_seed(SEED)

model = TFNO(
    n_modes=(8, 8, 8),
    in_channels=1 + len(DATA["input_params"]),  # rho + one channel per beam parameter
    out_channels=1,
    hidden_channels=23,
    n_layers=3,
    domain_padding=0.5,
)
model = model.to(device)

# Count and display the number of parameters
n_params = count_model_params(model)
print(f"\nOur model has {n_params} parameters.")
sys.stdout.flush()

optimizer = AdamW(model.parameters(), lr=3.23e-3, weight_decay=8.21e-4)

b_scheduler = True
if b_scheduler:
    # Option A: (Scheduler in-use)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5)
else:
    # Option B: (Scheduler deactivated)
    scheduler = torch.optim.lr_scheduler.ConstantLR(
        optimizer, factor=1.0, total_iters=0)

l2loss = LpLoss(d=3, p=2)  # L2 loss for function values
h1loss = H1Loss(d=3, periodic_in_x=False, periodic_in_y=False, periodic_in_z=False)  # H1 loss includes gradient information

train_loss = h1loss
eval_losses = {"h1": h1loss, "l2": l2loss}

print("\n### MODEL ###\n", model)
print("\n### OPTIMIZER ###\n", optimizer)
print("\n### SCHEDULER ###\n", scheduler)
print("\n### LOSSES ###")
print(f"\n * Train: {train_loss}")
print(f"\n * Test: {eval_losses}")
sys.stdout.flush()

epochs = 5000  # TODO: goal: 10000; better 2000, since overfitting starts after 2500 (for 1000:250 datas); more epochs possible with more data
eval_intv_div = 10 if epochs < 1000 else 100 # log every x epochs

trainer = Trainer(
    model=model,
    n_epochs=epochs,
    device=device,
    data_processor=data_processor,
    wandb_log=False,  # Enable Weights & Biases logging for this tutorial
    eval_interval=epochs//eval_intv_div,
    use_distributed=False,  # Single GPU/CPU training
    verbose=True,  # Print training progress
)

# Loss history: the Trainer only prints its metrics, so its own log_training / log_eval
# are called as usual and their arguments are recorded on the way. Both run every
# eval_interval epochs.
history = {"train": defaultdict(list), "eval": defaultdict(list)}
log_training, log_eval = trainer.log_training, trainer.log_eval

def record_training(**kwargs):
    log_training(**kwargs)
    history["train"]["epoch"].append(kwargs["epoch"])
    for key in ("avg_loss", "train_err", "lr"):
        history["train"][key].append(float(kwargs[key]))


def record_eval(epoch, eval_metrics):
    log_eval(epoch=epoch, eval_metrics=eval_metrics)
    history["eval"]["epoch"].append(epoch)
    for key, value in eval_metrics.items():
        history["eval"][key].append(float(value))


trainer.log_training, trainer.log_eval = record_training, record_eval

trainer.train(
    train_loader=train_loader,
    test_loaders=test_loaders,
    optimizer=optimizer,
    scheduler=scheduler,
    regularizer=False,
    training_loss=train_loss,
    eval_losses=eval_losses,
    save_best= "32_h1",
    save_dir = MODELS_DIR / "checkpoints",
)

model = TFNO.from_checkpoint(MODELS_DIR / "checkpoints", "best_model", map_location=device).to(device)
model.save_checkpoint(MODELS_DIR, "final_model")
torch.save(data_processor, MODELS_DIR / "final_model_data_processor.pt")

plot_losses(history, save_dir=FIGURES_DIR / run)
(TABLES_DIR / run).mkdir(parents=True, exist_ok=True)
pd.DataFrame(history["train"]).merge(pd.DataFrame(history["eval"]), on="epoch", how="outer").to_csv(
    TABLES_DIR / run / "history.csv", index=False
)