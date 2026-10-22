"""G4+G5 - Dataset and DataLoader for the Cheetah space-charge dataset.

Reads the HDF5 files from ``generate.py``:

- ``CheetahDataset(path, nondim=True)``: one .h5 file, access per sample. The file
  holds physical fields (rho [C/m^3], phi [V]); nondimensionalization happens here on
  loading. Provides ``x`` (4 channels: rho_hat or rho + log(E_i/L), see
  ``input_channels``) and ``y`` (1 channel: phi_hat or phi) for the neuralop
  ``Trainer``, and on request both variants plus scaling (``rho``, ``phi``,
  ``rho_hat``, ``phi_hat``, ``total_charge``, ``reference_length``,
  ``normalized_extent``, ``cell_size``, ``gamma``, ``kind``) for ``evaluate.py``.
  ``downsample=f`` coarsens by a factor f per axis (resolution study).
- ``load_cheetah_sc(...)``: same signature and return value as neuralop's
  ``load_darcy_flow_small``, plus ``train_resolution``, ``generate``, ``num_workers``
  and ``seed``. Returns ``(train_loader, test_loaders, data_processor)``, with
  ``shuffle=True`` in the training DataLoader (missing in the Darcy template). The
  normalizers are fitted on the training data as in neuralop's ``PTDataset``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import h5py
import numpy as np
import torch
from neuralop.data.transforms.data_processors import DefaultDataProcessor
from neuralop.data.transforms.normalizers import UnitGaussianNormalizer
from torch.utils.data import DataLoader, Dataset, Subset

from scno.utils.normalization import input_channels, nondimensionalize_fields

from .generate import dataset_path, generate_cheetah_sc

# Example data root, analogous to ``example_data_root`` in neuralop's darcy.py.
example_data_root = Path("data")


class CheetahDataset(Dataset):
    """Field pairs (rho_hat -> phi_hat, or raw rho -> phi) from an HDF5 file.

    The file is opened only on first access and kept once per process, so that
    DataLoader workers each have their own handle.

    Parameters
    ----------
    path : str | Path
        .h5 file from ``write_hdf5``.
    downsample : int
        Coarsening per axis. Density and potential are averaged over f^3 blocks (the
        total charge is conserved), the cell size grows by f; box, L and
        normalized_extent stay the same.
    return_meta : bool
        False -> samples contain only ``x``/``y`` (the neuralop ``Trainer`` passes every
        key on to model and loss).
    nondim : bool
        True -> ``x``/``y`` nondimensionalized (rho_hat, phi_hat).
        False -> physical (rho in C/m^3, phi in V), for the ablation.
        The channels log(E_i/L) are the same in both cases.
    """

    def __init__(
        self,
        path: Union[str, Path],
        downsample: int = 1,
        return_meta: bool = True,
        nondim: bool = True,
    ):
        self.path = Path(path)
        self.downsample = int(downsample)
        self.return_meta = return_meta
        self.nondim = nondim
        self._file = None

        with h5py.File(self.path, "r") as f:
            if "rho" not in f:
                raise ValueError(
                    f"{self.path}: old file format (rho_hat/phi_hat). The files now hold "
                    "physical fields -- delete the file and generate it anew "
                    "(load_cheetah_sc(generate=True) creates missing files)."
                )
            self._len = f["rho"].shape[0]
            self.grid_shape = tuple(int(n) for n in f["rho"].shape[1:])

        if self.downsample < 1 or any(n % self.downsample for n in self.grid_shape):
            raise ValueError(
                f"downsample={downsample} does not divide the grid {self.grid_shape}"
            )

    def _h5(self) -> h5py.File:
        if self._file is None:
            self._file = h5py.File(self.path, "r")
        return self._file

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_file"] = None  # h5py handles cannot be pickled (workers)
        return state

    def __len__(self) -> int:
        return self._len

    def _coarsen(self, field: torch.Tensor) -> torch.Tensor:
        """(..., nx, ny, nz) -> averaged over f^3 blocks."""
        if self.downsample == 1:
            return field
        grid = field.shape[-3:]
        pooled = torch.nn.functional.avg_pool3d(
            field.reshape(-1, 1, *grid), self.downsample
        )
        return pooled.reshape(*field.shape[:-3], *pooled.shape[-3:])

    def read_fields(self, index) -> dict:
        """Raw fields + scaling for ``index`` (int or slice), with batch dimension."""
        f = self._h5()
        rho = self._coarsen(
            torch.from_numpy(f["rho"][index].reshape(-1, *self.grid_shape))
        )
        phi = self._coarsen(
            torch.from_numpy(f["phi"][index].reshape(-1, *self.grid_shape))
        )
        cell_size = (
            torch.from_numpy(f["cell_size"][index]).reshape(-1, 3) * self.downsample
        )
        gamma = torch.from_numpy(np.atleast_1d(f["gamma"][index]))
        rho_hat, phi_hat, scaling = nondimensionalize_fields(
            rho.double(), phi.double(), cell_size, gamma.double()
        )
        return {
            "rho": rho,
            "phi": phi,
            "rho_hat": rho_hat.float(),
            "phi_hat": phi_hat.float(),
            "scaling": scaling,
            "cell_size": cell_size,
            "gamma": gamma,
            "kind": torch.from_numpy(np.atleast_1d(f["kind"][index])),
        }

    def _xy(self, fields: dict) -> tuple[torch.Tensor, torch.Tensor]:
        if self.nondim:
            rho, phi = fields["rho_hat"], fields["phi_hat"]
        else:
            rho, phi = fields["rho"], fields["phi"]
        x = input_channels(rho, fields["scaling"].normalized_extent)  # (B, 4, ...)
        return x, phi[:, None]  # (B, 1, ...)

    def __getitem__(self, index: int) -> dict:
        fields = self.read_fields(int(index))
        x, y = self._xy(fields)
        item = {"x": x[0], "y": y[0]}

        if self.return_meta:
            scaling = fields["scaling"]
            item.update(
                rho=fields["rho"][0],
                phi=fields["phi"][0],
                rho_hat=fields["rho_hat"][0],
                phi_hat=fields["phi_hat"][0],
                total_charge=scaling.total_charge[0],
                reference_length=scaling.reference_length[0],
                normalized_extent=scaling.normalized_extent[0],
                cell_size=fields["cell_size"][0],
                gamma=fields["gamma"][0],
                kind=fields["kind"][0],
            )
        return item

    def read_xy(self, n: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Model inputs/outputs of the first ``n`` samples at once (for fitting).

        x (n, 4, nx, ny, nz), y (n, 1, nx, ny, nz) -- as in ``__getitem__``.
        """
        return self._xy(self.read_fields(slice(n)))


def missing_files(
    root_dir: Union[str, Path], train_resolution: int, test_resolutions: list[int]
) -> list[str]:
    """Names of missing files (one training file + one per test resolution).

    Note: the generator writes exactly ONE training file
    (``cheetah_sc_train_{train_resolution}.h5``) and one test file per test resolution
    (``cheetah_sc_test_{res}.h5``). No ``train_{test_res}.h5`` is created.
    """
    paths = [dataset_path(root_dir, "train", train_resolution)]
    paths += [dataset_path(root_dir, "test", res) for res in test_resolutions]
    return [p.name for p in paths if not p.exists()]


def _first(dataset: CheetahDataset, n: int) -> Subset:
    if n > len(dataset):
        raise ValueError(
            f"{dataset.path.name} has only {len(dataset)} samples, {n} requested"
        )
    return Subset(dataset, range(n))


def _fit_normalizer(
    data: torch.Tensor, encoding: str, channel_dim: int
) -> UnitGaussianNormalizer:
    """As in neuralop's ``PTDataset``: channel-wise or pixel-wise."""
    if encoding == "channel-wise":
        reduce_dims = list(range(data.ndim))
        reduce_dims.pop(channel_dim)  # keep the mean per channel
    elif encoding == "pixel-wise":
        reduce_dims = [0]
    else:
        raise ValueError(f"unknown encoding: {encoding}")
    normalizer = UnitGaussianNormalizer(dim=reduce_dims)
    normalizer.fit(data)
    return normalizer


def load_cheetah_sc(
    n_train,
    n_tests,
    batch_size,
    test_batch_sizes,
    nondim=True,
    data_root=example_data_root,
    train_resolution=32,
    test_resolutions=[32, 64],
    encode_input=False,
    encode_output=None,
    encoding="channel-wise",
    channel_dim=1,
    generate=True,
    seed=0,
    ranges=None,
    num_workers=1,
):
    """Loads (or generates) the Cheetah space-charge dataset.

    Returns three objects like ``load_darcy_flow_small``:
        (train_loader, test_loaders, data_processor)

    ``scripts/train_model.py`` passes its ``DATA`` settings as keyword arguments.

    Parameters
    ----------
    n_train : int
        Number of training samples (the first n_train of the file).
    n_tests : List[int]
        Number of test samples per test resolution.
    batch_size : int
        Batch size of the training data.
    test_batch_sizes : List[int]
        Batch sizes of the test data.
    nondim : bool, optional
        True (default): the model sees rho_hat -> phi_hat. False: rho [C/m^3] ->
        phi [V] (ablation). Acts on loading, not on generating -- the same files
        serve both variants.
    data_root : str | Path
        Root of the .h5 files.
    train_resolution : int, optional
        Resolution of the training data (default 32).
    test_resolutions : List[int], optional
        Resolutions of the test data (default [32, 64]).
    encode_input : bool, optional
        Fit a Gaussian normalizer on ``x`` (default False).
    encode_output : bool | None, optional
        Fit a Gaussian normalizer on ``y``. ``None`` (default) means ``not nondim``:
        with nondim=True the physical scaling already makes phi_hat O(1), and a
        second scaling layer would only subtract a constant mean and distort the
        relative training loss. The normalizer is kept for the raw ablation.
    encoding, channel_dim:
        as in ``load_darcy_flow_small``.
    generate : bool, optional
        Whether missing files are generated (default True). Existing files are not
        regenerated.
    seed : int, optional
        Base seed for the generation.
    ranges : dict, optional
        Beam parameter bounds for the generation.
    num_workers : int, optional
        Workers of the training/test DataLoaders (default 1).

    Returns
    -------
    train_loader : torch.utils.data.DataLoader
        Training DataLoader with ``shuffle=True``; samples ``{"x", "y"}``.
    test_loaders : dict[int, torch.utils.data.DataLoader]
        One DataLoader per resolution in ``test_resolutions`` (shuffle=False).
    data_processor : neuralop.data.transforms.DefaultDataProcessor
        With the normalizers fitted on the training data (if any).
    """
    if len(n_tests) != len(test_resolutions):
        raise ValueError(
            f"n_tests ({len(n_tests)}) must match "
            f"test_resolutions ({len(test_resolutions)})"
        )
    if len(test_batch_sizes) != len(test_resolutions):
        raise ValueError(
            f"test_batch_sizes ({len(test_batch_sizes)}) must match "
            f"test_resolutions ({len(test_resolutions)})"
        )
    if encode_output is None:
        encode_output = not nondim

    data_root = Path(data_root)
    if generate and missing_files(data_root, train_resolution, test_resolutions):
        generate_cheetah_sc(
            root=data_root,
            n_train=n_train,
            n_tests=n_tests,
            train_resolution=train_resolution,
            test_resolutions=test_resolutions,
            seed=seed,
            ranges=ranges,
        )

    train_ds = CheetahDataset(
        dataset_path(data_root, "train", train_resolution),
        return_meta=False,
        nondim=nondim,
    )
    train_db = _first(train_ds, n_train)

    in_normalizer = out_normalizer = None
    if encode_input or encode_output:
        x_train, y_train = train_ds.read_xy(n_train)  # (n, 4, ...), (n, 1, ...)
    if encode_input:
        in_normalizer = _fit_normalizer(x_train, encoding, channel_dim)
    if encode_output:
        out_normalizer = _fit_normalizer(y_train, encoding, channel_dim)
    data_processor = DefaultDataProcessor(
        in_normalizer=in_normalizer, out_normalizer=out_normalizer
    )

    train_loader = DataLoader(
        train_db,
        batch_size=batch_size,
        shuffle=True,  # missing in the Darcy template
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=False,
    )

    test_loaders = {}
    for res, n_test, test_bsize in zip(test_resolutions, n_tests, test_batch_sizes):
        test_ds = CheetahDataset(
            dataset_path(data_root, "test", res), return_meta=False, nondim=nondim
        )
        test_loaders[res] = DataLoader(
            _first(test_ds, n_test),
            batch_size=test_bsize,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=False,
        )

    return train_loader, test_loaders, data_processor
