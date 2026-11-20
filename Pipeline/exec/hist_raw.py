import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch

from scno.config import SEED, DATA
from scno.data.dataset import load_cheetah_sc
from scno.utils.common import print_data_settings

#plot data settings
print_data_settings(DATA)
# set device agnostic code
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

# load input features (expect rho) from test_dataset with res=32
feats = test_loaders[DATA["test_resolutions"][0]].dataset.features    # tensor of shape (1250, 4)
# columns follow DATA["input_params"]: log_sigma_x, log_sigma_y, log_sigma_tau, gamma
df = pd.DataFrame(feats.numpy(), columns=DATA["input_params"])

# get panda dataframes from for input features
# np.exp on sigmas, since they are log
df_sigma_x = df["sigma_x"]
df_sigma_y = df["sigma_y"]
df_sigma_tau = df["sigma_tau"]
# get gamma directly
df_gamma = df["gamma"]


# machine epsilon
eps = np.finfo(np.float32).eps   # ≈ 1.19e-7

# prepare rho for log operations
rho = test_loaders[DATA["test_resolutions"][0]].dataset.data["rho"][:]    # tensor of shape (1250, 32, 32, 32)
rho_plus_eps = rho.flatten() + eps                    # shifts every value by eps
# get panda dataframe for rho
df_rho =  pd.DataFrame(np.log(rho_plus_eps), columns=["rho"])

# prepare phi for log operations
phi = test_loaders[DATA["test_resolutions"][0]].dataset.data["phi"][:]
phi_plus_eps = phi.flatten() + eps
# get panda dataframe for phi
df_phi = pd.DataFrame(np.log(phi_plus_eps), columns=["phi"])


# mean, median, std of sigma_x
mean_sigma_x = df_sigma_x.mean()
median_sigma_x = df_sigma_x.median()
std_sigma_x = df_sigma_x.std()

# plot histogram of sigma_x with errorbars
plt.hist(df_sigma_x, bins=30, alpha=0.6)
plt.errorbar(mean_sigma_x, 0, xerr=std_sigma_x, fmt='o', color='red', label='error')
plt.axvline(median_sigma_x, color='green', linestyle='dashed', linewidth=1, label='Median')

plt.title("sigma_x distribution")
plt.xlabel("value")
#plt.xscale("log")
plt.ylabel("frequency")
plt.show()


# mean, median, std of sigma_y
mean_sigma_y = df_sigma_y.mean()
median_sigma_y = df_sigma_y.median()
std_sigma_y = df_sigma_y.std()

# plot histogram of sigma_y with errorbars
plt.hist(df_sigma_y, bins=30, alpha=0.6)
plt.errorbar(mean_sigma_y, 0, xerr=std_sigma_y, fmt='o', color='red', label='error')
plt.axvline(median_sigma_y, color='green', linestyle='dashed', linewidth=1, label='Median')

plt.title("sigma_y distribution")
plt.xlabel("value")
#plt.xscale("log")
plt.ylabel("frequency")
plt.show()


# mean, median, std of sigma_tau
mean_sigma_tau = df_sigma_tau.mean()
median_sigma_tau = df_sigma_tau.median()
std_sigma_tau = df_sigma_tau.std()

# plot histogram of sigma_tau with errorbars
plt.hist(df_sigma_tau, bins=30, alpha=0.6)
plt.errorbar(mean_sigma_tau, 0, xerr=std_sigma_tau, fmt='o', color='red', label='error')
plt.axvline(median_sigma_tau, color='green', linestyle='dashed', linewidth=1, label='Median')

plt.title("sigma_tau distribution")
plt.xlabel("value")
#plt.xscale("log")
plt.ylabel("frequency")
plt.show()


# mean, median, std of gamma
mean_gamma = df_gamma.mean()
median_gamma = df_gamma.median()
std_gamma = df_gamma.std()

# plot histogram of gamma with errorbars
plt.hist(df_gamma, bins=30, alpha=0.6)
plt.errorbar(mean_gamma, 0, xerr=std_gamma, fmt='o', color='red', label='error')
plt.axvline(median_gamma, color='green', linestyle='dashed', linewidth=1, label='Median')

plt.title("gamma distribution")
plt.xlabel("value")
plt.ylabel("frequency")
plt.show()


# mean, median, std of rho
mean_rho = df_rho["rho"].mean()
median_rho = df_rho["rho"].median()
std_rho = df_rho["rho"].std()

# plot histogram of rho with errorbars
plt.hist(df_rho["rho"], bins=200, alpha=0.6)
plt.errorbar(mean_rho, 0, xerr=std_rho, fmt='o', color='red', label='error')
plt.axvline(median_rho, color='green', linestyle='dashed', linewidth=1, label='Median')

plt.title("rho distribution")
plt.xlabel("value")
#plt.xscale("log")
plt.ylabel("frequency")
plt.show()


# mean, median, std of phi
mean_phi = df_phi["phi"].mean()
median_phi = df_phi["phi"].median()
std_phi = df_phi["phi"].std()

# plot histogram of phi with errorbars
plt.hist(df_phi["phi"], bins=200, alpha=0.6)
plt.errorbar(mean_phi, 0, xerr=std_phi, fmt='o', color='red', label='error')
plt.axvline(median_phi, color='green', linestyle='dashed', linewidth=1, label='Median')

plt.title("phi distribution")
plt.xlabel("value")
#plt.xscale("log")
plt.ylabel("frequency")
plt.show()