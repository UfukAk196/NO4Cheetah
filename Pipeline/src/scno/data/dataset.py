from pathlib import Path
from typing import List, Union

import h5py
import torch
from torch.utils.data import DataLoader
from neuralop.data.datasets.hdf5_dataset import H5pyDataset
from neuralop.data.transforms.data_processors import DefaultDataProcessor
from neuralop.data.transforms.normalizers import UnitGaussianNormalizer

from scno.config import DATA_DIR

from .generate import dataset_path, generate_cheetah_sc


class CheetahH5Dataset(H5pyDataset):
    """One .h5 file of ``generate.py``: sample ``{"x": rho, "y": phi}``, each of shape
    (1, res, res, res), i.e. with the channel dim the model expects.

    neuralop's ``H5pyDataset`` is written for 2D data: its ``resolution`` argument only
    picks a subsampling step for the first two spatial axes of the 1024^2 Navier-Stokes
    files. Here the grids are loaded as stored, so ``__init__`` and ``__getitem__`` are
    replaced; the lazy h5 file handle (``data``) and ``__len__`` are inherited.
    """

    def __init__(self, data_path: Union[Path, str], n_samples: int = None):
        self.data_path = data_path
        self._data = None
        if n_samples is None:
            with h5py.File(str(data_path), "r") as f:
                n_samples = f["x"].shape[0]
        self.n_samples = n_samples

    def __getitem__(self, idx: int) -> dict:
        assert idx < self.n_samples, f"sample {idx} of a dataset with {self.n_samples} samples"
        x = torch.tensor(self.data["x"][idx], dtype=torch.float32).unsqueeze(0)
        y = torch.tensor(self.data["y"][idx], dtype=torch.float32).unsqueeze(0)
        return {"x": x, "y": y}


class CheetahDataset:
    """
    CheetahDataset stores data from the Cheetah simulation: input is the 3D charge
    density rho, output the 3D potential phi, both on the same (res, res, res) grid with
    one channel, in physical units as written by ``generate.py``.

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
    encode_input : bool, optional
        whether to normalize the input rho in the DataProcessor, by default False
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
        encode_input: bool = False,
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

        self._train_db = CheetahH5Dataset(
            dataset_path(root_dir, "train", train_resolution), n_samples=n_train
        )
        self._test_dbs = {
            res: CheetahH5Dataset(dataset_path(root_dir, "test", res), n_samples=n_test)
            for res, n_test in zip(test_resolutions, n_tests)
        }

        # Normalization (neuralop UnitGaussianNormalizer): z = (field - mean) / std, with
        # mean/std fitted on the training data only. The DataProcessor applies it to x
        # (encode_input) before the model and, in training, to y (encode_output); in eval
        # mode it decodes the model output back to phi [V].
        in_normalizer = out_normalizer = None
        if encode_input or encode_output:
            with h5py.File(dataset_path(root_dir, "train", train_resolution), "r") as f:
                x_train = torch.from_numpy(f["x"][:n_train]).unsqueeze(channel_dim)
                y_train = torch.from_numpy(f["y"][:n_train]).unsqueeze(channel_dim)
        if encode_input:
            in_normalizer = self._fit_normalizer(x_train, encoding, channel_dim)
        if encode_output:
            out_normalizer = self._fit_normalizer(y_train, encoding, channel_dim)

        self._data_processor = DefaultDataProcessor(
            in_normalizer=in_normalizer, out_normalizer=out_normalizer
        )

    @staticmethod
    def _fit_normalizer(
        data: torch.Tensor, encoding: str, channel_dim: int
    ) -> UnitGaussianNormalizer:
        """As in neuralop's ``PTDataset``: the dims mean/std are reduced over."""
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
    encode_input=False,
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
