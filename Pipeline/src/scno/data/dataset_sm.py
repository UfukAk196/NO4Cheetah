import os
from pathlib import Path
from typing import Union, List

from torch.utils.data import DataLoader
from neuralop.data.datasets.pt_dataset import PTDataset

from .generate_sm import dataset_path, generate_cheetah_sc


def missing_files(
    root_dir: Union[str, Path], train_resolution: int, test_resolutions: List[int]
) -> List[str]:
    """Names of missing .pt files (one training file + one per test resolution)."""
    paths = [dataset_path(root_dir, "train", train_resolution)]
    paths += [dataset_path(root_dir, "test", res) for res in test_resolutions]
    return [p.name for p in paths if not p.exists()]


class CheetahDataset(PTDataset):
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

        # List of resolutions needed for dataset object
        resolutions = set(test_resolutions + [train_resolution])

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

        # once generated/if files already exist, init PTDataset
        super().__init__(
            root_dir=root_dir,
            n_train=n_train,
            n_tests=n_tests,
            dataset_name="cheetah_sc",
            batch_size=batch_size,
            test_batch_sizes=test_batch_sizes,
            train_resolution=train_resolution,
            test_resolutions=test_resolutions,
            encode_input=encode_input,
            encode_output=encode_output,
            encoding=encoding,
            channel_dim=channel_dim,
            input_subsampling_rate=subsampling_rate,
            output_subsampling_rate=subsampling_rate,
        )


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