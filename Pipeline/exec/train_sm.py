import torch
import matplotlib.pyplot as plt
import sys

from neuralop.models import TFNO
from neuralop import Trainer
from neuralop.training import AdamW
from neuralop import LpLoss, H1Loss

from scno.data.dataset_sm import load_cheetah_sc
from scno.config import FIGURES_DIR, MODELS_DIR, SEED
from scno.evaluate import predict

device = "cuda" if torch.cuda.is_available() else "cpu"

train_loader, test_loaders, data_processor = load_cheetah_sc(
    n_train=1000,
    batch_size=64,
    n_tests=[100, 50],
    test_resolutions=[32, 64],
    test_batch_sizes=[32, 32],
    seed=SEED,
    ranges=None,
)
data_processor = data_processor.to(device)

model = TFNO(
    n_modes=(5, 5, 5),
    in_channels=1,
    out_channels=1,
    hidden_channels=32,
    n_layers=3,
    domain_padding=0.125,
)
model = model.to(device)

optimizer = AdamW(model.parameters(), lr=2.0e-4, weight_decay=4.0e-6)

b_scheduler = False
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

epochs = 100  # TODO: goal: 10000

trainer = Trainer(
    model=model,
    n_epochs=epochs,
    device=device,
    data_processor=data_processor,
    wandb_log=False,  # Enable Weights & Biases logging for this tutorial
    eval_interval=epochs//10,  # Evaluate every 10 epochs
    use_distributed=False,  # Single GPU/CPU training
    verbose=True,  # Print training progress
)

trainer.train(
    train_loader=train_loader,
    test_loaders=test_loaders,
    optimizer=optimizer,
    scheduler=scheduler,
    regularizer=False,
    training_loss=train_loss,
    eval_losses=eval_losses,
    save_best= "32_h1",
    save_dir = MODELS_DIR,
)

# Plot the best checkpoint (save_best), not the last-epoch model.
model = TFNO.from_checkpoint(MODELS_DIR, "best_model", map_location=device).to(device)
model.eval()

test_samples = test_loaders[32].dataset

fig = plt.figure(figsize=(7, 7))
for index in range(3):
    data = test_samples[index]

    # Input rho and ground-truth output phi, both in physical units
    x = data["x"]
    y = data["y"].to("cpu")
    # Model prediction, decoded to physical units by predict (data processor in eval mode)
    with torch.no_grad():
        out = predict(model, torch.utils.data.default_collate([data]), data_processor)[0].to("cpu")
    x = x.to("cpu")

    # Plot input
    ax = fig.add_subplot(3, 3, index * 3 + 1)
    ax.imshow(x[0, :, :, x.shape[-1]//2], cmap="gray")
    if index == 0:
        ax.set_title("Input rho")
    plt.xticks([], [])
    plt.yticks([], [])

    # Plot ground-truth output
    ax = fig.add_subplot(3, 3, index * 3 + 2)
    ax.imshow(y[0, :, :, y.shape[-1]//2].squeeze())
    if index == 0:
        ax.set_title("Ground-truth output phi")
    plt.xticks([], [])
    plt.yticks([], [])

    # Plot model prediction
    ax = fig.add_subplot(3, 3, index * 3 + 3)
    ax.imshow(out[0, :, :, out.shape[-1]//2].squeeze().detach().numpy())
    if index == 0:
        ax.set_title("Model prediction phi")
    plt.xticks([], [])
    plt.yticks([], [])

fig.suptitle("TFNO predictions on 32x32 Cheetah Space Charge Kick", y=0.98)
plt.tight_layout()
fig.show()

save_dir = FIGURES_DIR
save_dir.mkdir(parents=True, exist_ok=True)
path = save_dir / "predictions_32x32.png"
fig.savefig(path, dpi=300)

test_samples = test_loaders[64].dataset

fig = plt.figure(figsize=(7, 7))
for index in range(3):
    data = test_samples[index]

    # Input rho and ground-truth output phi at higher-resolution, physical units
    x = data["x"]
    y = data["y"].to("cpu")
    # Model prediction at higher-resolution, decoded to physical units
    with torch.no_grad():
        out = predict(model, torch.utils.data.default_collate([data]), data_processor)[0].to("cpu")
    x = x.to("cpu")

    # Plot input at higher-resolution
    ax = fig.add_subplot(3, 3, index * 3 + 1)
    ax.imshow(x[0, :, :, x.shape[-1]//2], cmap="gray")
    if index == 0:
        ax.set_title("Input rho at 64x64")
    plt.xticks([], [])
    plt.yticks([], [])

    # Plot ground-truth output at higher-resolution
    ax = fig.add_subplot(3, 3, index * 3 + 2)
    ax.imshow(y[0, :, :, y.shape[-1]//2].squeeze())
    if index == 0:
        ax.set_title("Ground-truth output phi")
    plt.xticks([], [])
    plt.yticks([], [])

    # Plot model prediction at higher-resolution
    ax = fig.add_subplot(3, 3, index * 3 + 3)
    ax.imshow(out[0, :, :, out.shape[-1]//2].squeeze().detach().numpy())
    if index == 0:
        ax.set_title("TFNO prediction phi")
    plt.xticks([], [])
    plt.yticks([], [])

fig.suptitle("Zero-shot super-resolution: 32x32 → 64x64", y=0.98)
plt.tight_layout()
fig.show()

path = save_dir / "predictions_64x64.png"
fig.savefig(path, dpi=300)