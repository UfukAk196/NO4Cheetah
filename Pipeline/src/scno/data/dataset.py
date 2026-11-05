import os
from pathlib import Path
from typing import Union, List

import h5py
import torch
from torch.utils.data import DataLoader
from neuralop.data.datasets.hdf5_dataset import H5pyDataset
from neuralop.data.transforms.data_processors import DefaultDataProcessor
from neuralop.data.transforms.normalizers import UnitGaussianNormalizer

from .generate_sm import dataset_path, generate_cheetah_sc


def missing_files(
    root_dir: Union[str, Path], train_resolution: int, test_resolutions: List[int]
) -> List[str]:
    """Names of missing .h5 files (one training file + one per test resolution)."""
    paths = [dataset_path(root_dir, "train", train_resolution)]
    paths += [dataset_path(root_dir, "test", res) for res in test_resolutions]
    return [p.name for p in paths if not p.exists()]


def _add_channel(t: torch.Tensor) -> torch.Tensor:
    """Add the channel dim that ``H5pyDataset`` does not add.

    Module-level (not a lambda) so it can be pickled for ``num_workers > 0`` on Windows.
    """
    return t.unsqueeze(0)


class CheetahDataset:
    """
    CheetahDataset stores data from the Cheetah simulation. Input and output are both
    3d fields of charge density rho and potential field phi respectively with
    three channels of data, each for one component x, y, z (in case of rho) and tau (in case of phi).


    Parameters
    ----------
    root_dir : Union[Path, str]
        root at which to download data files
    n_train : int
        number of train instances
    n_tests : List[int]
        number of test instances per test dataset
    batch_size : int
        batch size of training set
    test_batch_sizes : List[int]
        batch size of test sets
    train_resolution : int
        resolution of data for training set
    test_resolutions : List[int], optional
        resolution of data for testing sets, by default [16,32]
    encode_input : bool, optional
        whether to normalize inputs in provided DataProcessor,
        by default False
    encode_output : bool, optional
        whether to normalize outputs in provided DataProcessor,
        by default True
    encoding : str, optional
        parameter for input/output normalization. Whether
        to normalize by channel ("channel-wise") or
        by pixel ("pixel-wise"), default "channel-wise"
    channel_dim : int, optional
        dimension of saved tensors to index data channels, by default 1
    subsampling_rate : int or List[int], optional
        rate at which to subsample each input dimension, by default None
    download : bool, optional
        whether to download data if not present, by default True

    Attributes
    ----------
    train_db: torch.utils.data.Dataset of training examples
    test_db:  ""                       of test examples
    data_processor: neuralop.data.transforms.DataProcessor to process data examples
        optional, default is None
    """

    def __init__(
        self,
        root_dir: Union[Path, str],
        n_train: int,
        n_tests: List[int],
        batch_size: int,
        test_batch_sizes: List[int],
        train_resolution: int,
        test_resolutions: int = [32, 64],
        encode_input: bool = False,
        encode_output: bool = True,
        encoding="channel-wise",
        channel_dim=1,
        subsampling_rate=None,
        generate: bool = True,
        seed=0,
        ranges=None,
    ):
        """Initialize the CheetahDataset.

        See class docstring for detailed parameter descriptions.
        """
        # convert root dir to Path
        if isinstance(root_dir, str):
            root_dir = Path(root_dir)
        if not root_dir.exists():
            root_dir.mkdir(parents=True)

        if subsampling_rate is not None:
            raise NotImplementedError(
                "H5pyDataset subsamples only 2 axes, which is wrong for 3D data"
            )

        # generate data if requested and missing
        if generate:
            generate_cheetah_sc(
                root=root_dir,
                n_train=n_train,
                n_tests=n_tests,
                train_resolution=train_resolution,
                test_resolutions=test_resolutions,
                seed=seed,
                ranges=ranges,
            )

        # H5pyDataset's resolution map is built for the 1024^2 Navier-Stokes data;
        # 1024 -> subsample_step 1, i.e. no subsampling, which is what we want.
        self._train_db = H5pyDataset(
            dataset_path(root_dir, "train", train_resolution),
            resolution=1024,
            transform_x=_add_channel,
            transform_y=_add_channel,
            n_samples=n_train,
        )

        self._test_dbs = {}
        for res, n_test in zip(test_resolutions, n_tests):
            self._test_dbs[res] = H5pyDataset(
                dataset_path(root_dir, "test", res),
                resolution=1024,
                transform_x=_add_channel,
                transform_y=_add_channel,
                n_samples=n_test,
            )

        # Fit the optional encoders on the training data (as in PTDataset).
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

    @property
    def train_db(self):
        return self._train_db

    @property
    def test_dbs(self):
        return self._test_dbs

    @property
    def data_processor(self):
        return self._data_processor


example_data_root = Path("data")

# load cheetah sc data for backwards compatibility
def load_cheetah_sc(
    n_train,
    n_tests,
    batch_size,
    test_batch_sizes,
    data_root=example_data_root,
    train_resolution=32,
    test_resolutions=[32, 64],
    encode_input=False,
    encode_output=True,
    encoding="channel-wise",
    channel_dim=1,
    subsampling_rate=None,
    num_workers=0,
    seed=0,
    ranges=None,
):

    if missing_files(data_root, train_resolution, test_resolutions):
        generate = True
    else:
        generate = False
        
    dataset = CheetahDataset(
        root_dir=data_root,
        n_train=n_train,
        n_tests=n_tests,
        batch_size=batch_size,
        test_batch_sizes=test_batch_sizes,
        train_resolution=train_resolution,
        test_resolutions=test_resolutions,
        encode_input=encode_input,
        encode_output=encode_output,
        encoding=encoding,
        channel_dim=channel_dim,
        subsampling_rate=subsampling_rate,
        generate=generate,
        seed=seed,
        ranges=ranges,
    )

    # return dataloaders for backwards compat
    train_loader = DataLoader(
        dataset.train_db,
        batch_size=batch_size,
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