"""G4+G5 - Dataset und DataLoader fuer den Cheetah-Space-Charge-Datensatz.

Liest die HDF5-Dateien aus ``generate.py``:

- ``CheetahDataset(path)``: eine .h5-Datei, Zugriff pro Probe. Liefert
  ``x`` (4 Kanaele: rho_hat + log(E_i/L), s. ``input_channels``) und
  ``y`` (1 Kanal: phi_hat) fuer den neuralop-``Trainer`` und auf
  Wunsch die Skalierung (``rho_hat``, ``phi_hat``, ``total_charge``,
  ``reference_length``, ``normalized_extent``, ``cell_size``, ``gamma``, ``kind``) fuer
  ``train.py``/``evaluate.py``. ``downsample=f`` vergroebert um Faktor f
  pro Achse (Aufloesungsstudie).
- ``make_loaders(path, ...)``: Train/Val-Split einer Datei fuer ``train.py``.
- ``load_cheetah_sc(...)``: gleiche Signatur und Rueckgabe wie
  ``load_darcy_flow_small`` plus ``train_resolution``, ``generate``,
  ``num_workers`` und ``seed``. Gibt ``(train_loader, test_loaders,
  data_processor)`` zurueck, mit ``shuffle=True`` im Train-DataLoader (fehlt
  in der Darcy-Vorlage). Die Normalisierer werden wie in neuralops
  ``PTDataset`` an den Trainingsdaten gefittet.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Union

import h5py
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from neuralop.data.transforms.data_processors import DefaultDataProcessor
from neuralop.data.transforms.normalizers import UnitGaussianNormalizer

from .generate import dataset_path, generate_cheetah_sc

# Beispiel-Datenwurzel, analog ``example_data_root`` in darcy.py.
# TODO: an die tatsaechliche Projektstruktur anpassen.
example_data_root = Path("data")

META_KEYS = ("total_charge", "reference_length", "normalized_extent", "cell_size", "gamma", "kind")


def input_channels(rho_hat: torch.Tensor, normalized_extent: torch.Tensor) -> torch.Tensor:
    """Modelleingabe x: rho_hat + drei konstante Kanaele log(E_i/L).

    E_i sind die (gamma-gestreckten) Boxkanten, nicht das E-Feld. Ohne sie
    sieht das Netz nur Gitterindizes und kann gleiche rho_hat bei
    verschiedenen Achsenverhaeltnissen nicht unterscheiden.

    rho_hat (..., nx, ny, nz), normalized_extent (..., 3) -> x (..., 4, nx, ny, nz)
    """
    grid = rho_hat.shape[-3:]
    log_aspect = normalized_extent.log().to(rho_hat.dtype)[..., :, None, None, None]
    return torch.cat(
        [rho_hat.unsqueeze(-4), log_aspect.expand(*log_aspect.shape[:-3], *grid)], dim=-4
    )


class CheetahDataset(Dataset):
    """Feld-Paare (rho_hat -> phi_hat) aus einer HDF5-Datei.

    Die Datei wird erst beim ersten Zugriff geoeffnet und je Prozess einmal
    gehalten, damit DataLoader-Worker jeweils ihren eigenen Handle haben.

    Parameters
    ----------
    path : str | Path
        .h5-Datei aus ``write_hdf5``.
    downsample : int
        Vergroeberung pro Achse. Dichte und Potential werden ueber
        f^3-Bloecke gemittelt (Mittelwert von rho_hat bleibt 1), die
        Zellgroesse waechst um f; Box, L und normalized_extent bleiben gleich.
    return_meta : bool
        False -> Proben enthalten nur ``x``/``y`` (neuralop-``Trainer`` reicht
        alle Keys an Modell und Loss weiter).
    """

    def __init__(self, path: Union[str, Path], downsample: int = 1, return_meta: bool = True):
        self.path = Path(path)
        self.downsample = int(downsample)
        self.return_meta = return_meta
        self._file = None

        with h5py.File(self.path, "r") as f:
            self._len = f["rho_hat"].shape[0]
            self.grid_shape = tuple(int(n) for n in f["rho_hat"].shape[1:])

        if self.downsample < 1 or any(n % self.downsample for n in self.grid_shape):
            raise ValueError(
                f"downsample={downsample} teilt das Gitter {self.grid_shape} nicht"
            )

    def _h5(self) -> h5py.File:
        if self._file is None:
            self._file = h5py.File(self.path, "r")
        return self._file

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_file"] = None  # h5py-Handles lassen sich nicht pickeln (Worker)
        return state

    def __len__(self) -> int:
        return self._len

    def _coarsen(self, field: torch.Tensor) -> torch.Tensor:
        if self.downsample == 1:
            return field
        return torch.nn.functional.avg_pool3d(field[None, None], self.downsample)[0, 0]

    def __getitem__(self, index: int) -> dict:
        f = self._h5()
        rho = self._coarsen(torch.from_numpy(f["rho_hat"][index]))
        phi = self._coarsen(torch.from_numpy(f["phi_hat"][index]))
        normalized_extent = torch.from_numpy(f["normalized_extent"][index])
        item = {"x": input_channels(rho, normalized_extent), "y": phi[None]}  # (4, ...), (1, ...)

        if self.return_meta:
            item["rho_hat"] = rho
            item["phi_hat"] = phi
            for key in META_KEYS:
                item[key] = torch.as_tensor(f[key][index])
            item["cell_size"] = item["cell_size"] * self.downsample
        return item

    def read_all(self, key: str, n: int | None = None) -> torch.Tensor:
        """Die ersten ``n`` Eintraege eines Feldes am Stueck (zum Fitten)."""
        return torch.from_numpy(self._h5()[key][slice(n)])

    def read_inputs(self, n: int | None = None) -> torch.Tensor:
        """Modelleingaben x der ersten ``n`` Proben: (n, 4, nx, ny, nz)."""
        return input_channels(self.read_all("rho_hat", n), self.read_all("normalized_extent", n))


def make_loaders(
    path: Union[str, Path],
    batch_size: int,
    val_fraction: float = 0.15,
    num_workers: int = 0,
    seed: int = 0,
) -> tuple[DataLoader, DataLoader]:
    """Train/Val-Loader aus einer .h5-Datei (zufaelliger, reproduzierbarer Split)."""
    dataset = CheetahDataset(path)
    n_val = max(1, int(round(len(dataset) * val_fraction)))
    if n_val >= len(dataset):
        raise ValueError(f"{path}: zu wenige Proben ({len(dataset)}) fuer einen Val-Split")

    order = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(seed))
    val_set = Subset(dataset, order[:n_val].tolist())
    train_set = Subset(dataset, order[n_val:].tolist())

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return train_loader, val_loader


def missing_files(
    root_dir: Union[str, Path], train_resolution: int, test_resolutions: list[int]
) -> list[str]:
    """Namen fehlender Dateien (eine Trainingsdatei + je Test-Aufloesung).

    Hinweis: Der Generator schreibt genau EINE Trainingsdatei
    (``cheetah_sc_train_{train_resolution}.h5``) und pro Test-Aufloesung
    eine Testdatei (``cheetah_sc_test_{res}.h5``). Es wird also KEIN
    ``train_{test_res}.h5`` erzeugt.
    """
    paths = [dataset_path(root_dir, "train", train_resolution)]
    paths += [dataset_path(root_dir, "test", res) for res in test_resolutions]
    return [p.name for p in paths if not p.exists()]


def _first(dataset: CheetahDataset, n: int) -> Subset:
    if n > len(dataset):
        raise ValueError(f"{dataset.path.name} hat nur {len(dataset)} Proben, verlangt {n}")
    return Subset(dataset, range(n))


def _fit_normalizer(data: torch.Tensor, encoding: str, channel_dim: int) -> UnitGaussianNormalizer:
    """Wie in neuralops ``PTDataset``: channel-wise oder pixel-wise."""
    if encoding == "channel-wise":
        reduce_dims = list(range(data.ndim))
        reduce_dims.pop(channel_dim)  # Mittelwert je Kanal behalten
    elif encoding == "pixel-wise":
        reduce_dims = [0]
    else:
        raise ValueError(f"unbekanntes encoding: {encoding}")
    normalizer = UnitGaussianNormalizer(dim=reduce_dims)
    normalizer.fit(data)
    return normalizer


def load_cheetah_sc(
    n_train,
    n_tests,
    batch_size,
    test_batch_sizes,
    nondim,
    data_root=example_data_root,
    train_resolution=32,
    test_resolutions=[32, 64],
    encode_input=False,
    encode_output=True,
    encoding="channel-wise",
    channel_dim=1,
    generate=True,
    seed=0,
    ranges=None,
    num_workers=1,
):
    """Lade (oder generiere) den Cheetah-Space-Charge-Datensatz.

    Gibt drei Objekte wie ``load_darcy_flow_small`` zurueck:
        (train_loader, test_loaders, data_processor)

    Parameters
    ----------
    n_train : int
        Anzahl Trainingsbeispiele (die ersten n_train der Datei).
    n_tests : List[int]
        Anzahl Testbeispiele je Test-Aufloesung.
    batch_size : int
        Batchgroesse der Trainingsdaten.
    test_batch_sizes : List[int]
        Batchgroessen der Testdaten.
    data_root : str | Path
        Wurzel der .h5-Dateien.
    train_resolution : int, optional
        Aufloesung der Trainingsdaten (Default 32).
    test_resolutions : List[int], optional
        Aufloesungen der Testdaten (Default [32, 64]).
    encode_input, encode_output, encoding, channel_dim:
        wie ``load_darcy_flow_small``.
    generate : bool, optional
        Ob fehlende Dateien generiert werden (Default True). Vorhandene
        Dateien werden nicht neu erzeugt.
    seed : int, optional
        Basis-Seed fuer die Generierung.
    ranges : dict, optional
        Strahlparameter-Grenzen fuer die Generierung.
    num_workers : int, optional
        Worker des Train-/Test-DataLoader (Default 1).

    Returns
    -------
    train_loader : torch.utils.data.DataLoader
        Trainings-DataLoader mit ``shuffle=True``; Proben ``{"x", "y"}``.
    test_loaders : dict[int, torch.utils.data.DataLoader]
        Ein DataLoader je Aufloesung in ``test_resolutions`` (shuffle=False).
    data_processor : neuralop.data.transforms.DefaultDataProcessor
        Mit an den Trainingsdaten gefitteten Normalisierern.
    """
    if len(n_tests) != len(test_resolutions):
        raise ValueError(
            f"n_tests ({len(n_tests)}) must match test_resolutions ({len(test_resolutions)})"
        )
    if len(test_batch_sizes) != len(test_resolutions):
        raise ValueError(
            f"test_batch_sizes ({len(test_batch_sizes)}) must match test_resolutions ({len(test_resolutions)})"
        )

    data_root = Path(data_root)
    if generate and missing_files(data_root, train_resolution, test_resolutions):
        generate_cheetah_sc(
            root=data_root,
            n_train=n_train,
            n_tests=n_tests,
            train_resolution=train_resolution,
            test_resolutions=test_resolutions,
            seed=seed,
            nondim=nondim,
            ranges=ranges,
        )

    train_ds = CheetahDataset(dataset_path(data_root, "train", train_resolution), return_meta=False)
    train_db = _first(train_ds, n_train)

    in_normalizer = out_normalizer = None
    if encode_input:
        x_train = train_ds.read_inputs(n_train)
        in_normalizer = _fit_normalizer(x_train, encoding, channel_dim)
    if encode_output:
        y_train = train_ds.read_all("phi_hat", n_train).unsqueeze(channel_dim)
        out_normalizer = _fit_normalizer(y_train, encoding, channel_dim)
    data_processor = DefaultDataProcessor(
        in_normalizer=in_normalizer, out_normalizer=out_normalizer
    )

    train_loader = DataLoader(
        train_db,
        batch_size=batch_size,
        shuffle=True,  # fehlt in der Darcy-Vorlage (Plan "Fallstricke")
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=False,
    )

    test_loaders = {}
    for res, n_test, test_bsize in zip(test_resolutions, n_tests, test_batch_sizes):
        test_ds = CheetahDataset(dataset_path(data_root, "test", res), return_meta=False)
        test_loaders[res] = DataLoader(
            _first(test_ds, n_test),
            batch_size=test_bsize,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=False,
        )

    return train_loader, test_loaders, data_processor
