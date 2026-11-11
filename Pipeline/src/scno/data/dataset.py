from pathlib import Path
from typing import List, Sequence, Union

import h5py
import numpy as np
import torch
from scipy.constants import c, e, m_e
from torch.utils.data import DataLoader
from neuralop.data.datasets.hdf5_dataset import H5pyDataset
from neuralop.data.transforms.data_processors import DefaultDataProcessor
from neuralop.data.transforms.normalizers import UnitGaussianNormalizer

from scno.config import DATA_DIR

from .generate import dataset_path, generate_cheetah_sc

ELECTRON_MASS_EV = m_e * c**2 / e  # as cheetah's ``Species("electron").mass_eV``

# Beam parameters that can be added to rho as constant input channels (``input_params``),
# each computed from the ``params`` group of a file: (N,) arrays -> (N,). Besides rho,
# ``SpaceChargeKick._solve_poisson_equation`` depends only on the cell size (sigmas) and
# gamma (energy). Sigmas are log-uniform in the data, hence their log.
FEATURES = {
    "log_sigma_x": lambda p: np.log(p["sigma_x"]),
    "log_sigma_y": lambda p: np.log(p["sigma_y"]),
    "log_sigma_tau": lambda p: np.log(p["sigma_tau"]),
    "gamma": lambda p: p["energy"] / ELECTRON_MASS_EV,  # as ``beam.relativistic_gamma``
}


def compute_features(
    path: Union[Path, str], input_params: Sequence[str], n_samples: int = None
) -> np.ndarray | None:
    """The ``input_params`` of the first ``n_samples`` samples of a file, (N, k) float32,
    in raw (not normalized) units; ``None`` if ``input_params`` is empty."""
    if not input_params:
        return None
    unknown = set(input_params) - set(FEATURES)
    if unknown:
        raise ValueError(f"unknown input_params {sorted(unknown)}, known: {list(FEATURES)}")
    with h5py.File(str(path), "r") as f:
        params = {k: f["params"][k][:n_samples] for k in f["params"]}
    return np.stack([FEATURES[k](params) for k in input_params], axis=-1).astype("float32")


class CheetahH5Dataset(H5pyDataset):
    """One .h5 file of ``generate.py``: sample ``{"x": [rho, *features], "y": phi}``, of
    shapes (1 + k, res, res, res) and (1, res, res, res), i.e. with the channel dim the
    model expects. Each of the k beam parameters in ``features`` (one row per sample,
    see ``compute_features``) becomes a constant channel.

    neuralop's ``H5pyDataset`` is written for 2D data: its ``resolution`` argument only
    picks a subsampling step for the first two spatial axes of the 1024^2 Navier-Stokes
    files. Here the grids are loaded as stored, so ``__init__`` and ``__getitem__`` are
    replaced; the lazy h5 file handle (``data``) and ``__len__`` are inherited.
    """

    def __init__(
        self,
        data_path: Union[Path, str],
        n_samples: int = None,
        features: np.ndarray | None = None,
    ):
        self.data_path = data_path
        self._data = None
        if n_samples is None:
            with h5py.File(str(data_path), "r") as f:
                n_samples = f["rho"].shape[0]
        self.n_samples = n_samples
        self.features = None if features is None else torch.from_numpy(features)

    def __getitem__(self, idx: int) -> dict:
        assert idx < self.n_samples, f"sample {idx} of a dataset with {self.n_samples} samples"
        rho = torch.tensor(self.data["rho"][idx], dtype=torch.float32).unsqueeze(0)
        phi = torch.tensor(self.data["phi"][idx], dtype=torch.float32).unsqueeze(0)
        if self.features is None:
            return {"x": rho, "y": phi}
        # one constant (res, res, res) channel per beam parameter, behind rho
        channels = self.features[idx][:, None, None, None].expand(-1, *rho.shape[1:])
        return {"x": torch.cat([rho, channels]), "y": phi}


class CheetahDataset:
    """
    CheetahDataset stores data from the Cheetah simulation: input is the 3D charge
    density rho plus optionally beam parameters as constant channels (``input_params``),
    output the 3D potential phi, all on the same (res, res, res) grid, in physical units
    as written by ``generate.py``.

    Missing .h5 files are generated first (``generate_cheetah_sc`` only writes files that
    are absent or hold too few samples, so on existing data this is a no-op).

    Parameters
    ----------
    root_dir : Union[Path, str]
        folder of the .h5 files
    n_train : int
        number of train instances
    n_tests : List[int]
        number of test instances per test dataset
    train_resolution : int
        resolution of data for training set
    test_resolutions : List[int], optional
        resolution of data for testing sets, by default [32, 64]
    input_params : Sequence[str], optional
        names in ``FEATURES`` added as input channels behind rho, by default () (rho only)
    encode_input : bool, optional
        whether to normalize all input channels in the DataProcessor, by default True
    encode_output : bool, optional
        whether to normalize the output phi in the DataProcessor, by default True
    encoding : str, optional
        "channel-wise": one mean/std per channel over all samples and grid points,
        "pixel-wise": one mean/std per grid point over all samples;
        default "channel-wise"
    channel_dim : int, optional
        dimension of saved tensors to index data channels, by default 1
    seed : int, optional
        seed of the data generation, by default 0
    ranges : dict, optional
        beam parameter bounds of the data generation, by default ``DEFAULT_RANGES``

    Attributes
    ----------
    train_db: torch.utils.data.Dataset of training examples
    test_dbs: dict ``{res: Dataset}`` of test examples
    data_processor: neuralop.data.transforms.DefaultDataProcessor with the fitted normalizers
    """

    def __init__(
        self,
        root_dir: Union[Path, str],
        n_train: int,
        n_tests: List[int],
        train_resolution: int,
        test_resolutions: List[int] = [32, 64],
        input_params: Sequence[str] = (),
        encode_input: bool = True,
        encode_output: bool = True,
        encoding="channel-wise",
        channel_dim=1,
        seed=0,
        ranges=None,
    ):
        root_dir = Path(root_dir)
        generate_cheetah_sc(
            root=root_dir,
            n_train=n_train,
            n_tests=n_tests,
            train_resolution=train_resolution,
            test_resolutions=test_resolutions,
            seed=seed,
            ranges=ranges,
        )

        train_path = dataset_path(root_dir, "train", train_resolution)
        train_features = compute_features(train_path, input_params, n_train)
        self._train_db = CheetahH5Dataset(train_path, n_samples=n_train, features=train_features)
        self._test_dbs = {}
        for res, n_test in zip(test_resolutions, n_tests):
            path = dataset_path(root_dir, "test", res)
            self._test_dbs[res] = CheetahH5Dataset(
                path, n_samples=n_test, features=compute_features(path, input_params, n_test)
            )

        # Normalization (neuralop UnitGaussianNormalizer): z = (field - mean) / std, with
        # mean/std fitted on the training data only. The DataProcessor applies it to x
        # (encode_input) before the model and, in training, to y (encode_output); in eval
        # mode it decodes the model output back to phi [V].
        in_normalizer = out_normalizer = None
        if encode_input or encode_output:
            with h5py.File(train_path, "r") as f:
                x_train = torch.from_numpy(f["rho"][:n_train]).unsqueeze(channel_dim)
                y_train = torch.from_numpy(f["phi"][:n_train]).unsqueeze(channel_dim)
        if encode_input:
            in_normalizer = self._fit_normalizer(x_train, encoding, channel_dim)
            if train_features is not None:
                in_normalizer = self._append_features(
                    in_normalizer, train_features, encoding, channel_dim
                )
        if encode_output:
            out_normalizer = self._fit_normalizer(y_train, encoding, channel_dim)

        self._data_processor = DefaultDataProcessor(
            in_normalizer=in_normalizer, out_normalizer=out_normalizer
        )

    @staticmethod
    def _reduce_dims(ndim: int, encoding: str, channel_dim: int) -> list[int]:
        """As in neuralop's ``PTDataset``: the dims mean/std are reduced over."""
        if encoding == "channel-wise":
            reduce_dims = list(range(ndim))
            reduce_dims.pop(channel_dim)  # keep the mean per channel
        elif encoding == "pixel-wise":
            reduce_dims = [0]
        else:
            raise ValueError(f"unknown encoding: {encoding}")
        return reduce_dims

    @classmethod
    def _fit_normalizer(
        cls, data: torch.Tensor, encoding: str, channel_dim: int
    ) -> UnitGaussianNormalizer:
        normalizer = UnitGaussianNormalizer(dim=cls._reduce_dims(data.ndim, encoding, channel_dim))
        normalizer.fit(data)
        return normalizer

    @classmethod
    def _append_features(
        cls,
        rho_normalizer: UnitGaussianNormalizer,
        features: np.ndarray,
        encoding: str,
        channel_dim: int,
    ) -> UnitGaussianNormalizer:
        """Normalizer of all input channels [rho, *features]: rho's mean/std plus one
        mean/std per feature over the training samples. A constant channel has the same
        mean/std over samples x grid, so this equals fitting on the stacked inputs, which
        would load the training set (1 + k) times."""
        shape = [1] * rho_normalizer.mean.ndim
        shape[0], shape[channel_dim] = features.shape
        feature_normalizer = cls._fit_normalizer(
            torch.from_numpy(features).reshape(shape), encoding, channel_dim
        )
        spatial = list(rho_normalizer.mean.shape)
        spatial[channel_dim] = features.shape[1]  # per grid point for "pixel-wise"
        mean, std = (
            torch.cat(
                [getattr(rho_normalizer, s), getattr(feature_normalizer, s).expand(spatial)],
                dim=channel_dim,
            )
            for s in ("mean", "std")
        )
        return UnitGaussianNormalizer(mean=mean, std=std, dim=rho_normalizer.dim)

    @property
    def train_db(self):
        return self._train_db

    @property
    def test_dbs(self):
        return self._test_dbs

    @property
    def data_processor(self):
        return self._data_processor


def load_cheetah_sc(
    n_train,
    n_tests,
    batch_size,
    test_batch_sizes,
    data_root=DATA_DIR,
    train_resolution=32,
    test_resolutions=[32, 64],
    input_params=(),
    encode_input=True,
    encode_output=True,
    encoding="channel-wise",
    channel_dim=1,
    num_workers=0,
    seed=0,
    ranges=None,
):
    """``(train_loader, test_loaders, data_processor)`` as neuralop's ``load_*`` functions;
    the arguments are those of ``CheetahDataset`` plus the batch sizes."""
    dataset = CheetahDataset(
        root_dir=data_root,
        n_train=n_train,
        n_tests=n_tests,
        train_resolution=train_resolution,
        test_resolutions=test_resolutions,
        input_params=input_params,
        encode_input=encode_input,
        encode_output=encode_output,
        encoding=encoding,
        channel_dim=channel_dim,
        seed=seed,
        ranges=ranges,
    )

    train_loader = DataLoader(
        dataset.train_db,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=False,
    )

    test_loaders = {}
    for res, test_bsize in zip(test_resolutions, test_batch_sizes):
        test_loaders[res] = DataLoader(
            dataset.test_dbs[res],
            batch_size=test_bsize,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=False,
        )

    return train_loader, test_loaders, dataset.data_processor
