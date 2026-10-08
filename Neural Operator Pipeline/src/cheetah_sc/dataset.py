"""G4+G5 - Dataset und DataLoader fuer den Cheetah-Space-Charge-Datensatz.

Nach dem Muster von ``neuralop/data/datasets/darcy.py``:

- ``CheetahDataset(PTDataset)``: wie ``DarcyDataset``, aber ``generate=True``
  ruft ``generate_cheetah_sc`` fuer fehlende Dateien statt eines
  Zenodo-Downloads. ``dataset_name="cheetah_sc"``,
  ``channels_squeezed=False`` (x hat mehrere Kanaele), ``channel_dim=1``.
- ``load_cheetah_sc(...)``: gleiche Signatur wie ``load_darcy_flow_small``
  plus ``train_resolution``, ``generate``, ``num_workers`` und ``seed``.
  Gibt ``(train_loader, test_loaders, dataset.data_processor)`` zurueck,
  mit ``shuffle=True`` im Train-DataLoader (fehlt in der Darcy-Vorlage).

PTDataset, DefaultDataProcessor und DataLoader werden unveraendert aus
neuralop bzw. PyTorch uebernommen.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Union

from torch.utils.data import DataLoader

from neuralop.data.datasets.pt_dataset import PTDataset

from .generate import DATASET_NAME, generate_cheetah_sc
from .sampling import DEFAULT_RANGES

# Beispiel-Datenwurzel, analog ``example_data_root`` in darcy.py.
# TODO: an die tatsaechliche Projektstruktur anpassen.
example_data_root = Path("data/cheetah_sc")


class CheetahDataset(PTDataset):
    """CheetahDataset speichert Feld-Paare (rho_hat -> phi_hat).

    Eingabe (x) ist die entdimensionalisierte Ladungsdichte via
    ``make_sample``: Kanal 0 = rho_hat, Kanaele 1..3 = E_hat.
    Ausgabe (y) ist das entdimensionalisierte Potential phi_hat.

    Im Gegensatz zu ``DarcyDataset``:
    - ``channels_squeezed=False``: x/y haben explizite Kanal-Dimension.
    - ``generate=True`` (Standard) erzeugt fehlende .pt-Dateien selbst statt
      sie von Zenodo zu laden.

    Parameters
    ----------
    root_dir : Union[Path, str]
        Wurzel, unter der die .pt-Dateien liegen/generiert werden.
    n_train : int
        Anzahl Trainingsbeispiele.
    n_tests : List[int]
        Anzahl Testbeispiele je Test-Aufloesung.
    batch_size : int
        Batchgroesse des Trainingssatzes (wird nur gespeichert).
    test_batch_sizes : List[int]
        Batchgroessen der Testsatzes.
    train_resolution : int
        Aufloesung der Trainingsdaten.
    test_resolutions : List[int]
        Aufloesungen der Testdaten.
    encode_input, encode_output, encoding, channel_dim:
        wie ``PTDataset`` / ``DarcyDataset``.
    subsampling_rate : int or List[int], optional
        Subsampling-Rate; sollte `None` bleiben (Aufloesungen werden immer
        echt erzeugt, s. Plan "Fallstricke").
    generate : bool, optional
        Ob fehlende .pt-Dateien per ``generate_cheetah_sc`` erzeugt werden.
    seed : int, optional
        Basis-Seed fuer die Generierung (reproduzierbar).
    ranges : dict, optional
        Strahlparameter-Grenzen fuer die Generierung (G1). Default: TODO-Werte.
    """

    def __init__(
        self,
        root_dir: Union[Path, str],
        n_train: int,
        n_tests: List[int],
        batch_size: int,
        test_batch_sizes: List[int],
        train_resolution: int,
        test_resolutions: List[int] = [32, 64],
        encode_input: bool = False,
        encode_output: bool = True,
        encoding="channel-wise",
        channel_dim=1,
        subsampling_rate=None,
        generate: bool = True,
        seed: int = 0,
        ranges: dict | None = None,
    ):
        if isinstance(root_dir, str):
            root_dir = Path(root_dir)

        if ranges is None:
            ranges = DEFAULT_RANGES

        # Fehlende .pt-Dateien selbst erzeugen (analog zum Download in Darcy).
        if generate:
            missing = self._missing_files(
                root_dir, train_resolution, test_resolutions
            )
            if missing:
                generate_cheetah_sc(
                    root=root_dir,
                    n_train=n_train,
                    n_tests=n_tests,
                    train_resolution=train_resolution,
                    test_resolutions=test_resolutions,
                    seed=seed,
                    ranges=ranges,
                )

        if len(n_tests) != len(test_resolutions):
            raise ValueError(
                f"n_tests ({len(n_tests)}) must match test_resolutions ({len(test_resolutions)})"
            )
        if len(test_batch_sizes) != len(test_resolutions):
            raise ValueError(
                f"test_batch_sizes ({len(test_batch_sizes)}) must match test_resolutions ({len(test_resolutions)})"
            )

        super().__init__(
            root_dir=root_dir,
            dataset_name=DATASET_NAME,
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
            # x/y haben explizite Kanal-Dimension -> Kanal nicht unsqueezen.
            channels_squeezed=False,
            input_subsampling_rate=subsampling_rate,
            output_subsampling_rate=subsampling_rate,
        )

    @staticmethod
    def _missing_files(
        root_dir: Path, train_resolution: int, test_resolutions: list[int]
    ) -> list[str]:
        """Namen fehlender Dateien (eine Trainingsdatei + je Test-Aufloesung).

        Hinweis: Der Generator schreibt genau EINE Trainingsdatei
        (``cheetah_sc_train_{train_resolution}.pt``) und pro Test-Aufloesung
        eine Testdatei (``cheetah_sc_test_{res}.pt``). Es wird also KEIN
        ``train_{test_res}.pt`` erzeugt.
        """
        missing: list[str] = []
        train_file = root_dir / f"{DATASET_NAME}_train_{train_resolution}.pt"
        if not train_file.exists():
            missing.append(train_file.name)
        for res in test_resolutions:
            test_file = root_dir / f"{DATASET_NAME}_test_{res}.pt"
            if not test_file.exists():
                missing.append(test_file.name)
        return missing


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
        Anzahl Trainingsbeispiele.
    n_tests : List[int]
        Anzahl Testbeispiele je Test-Aufloesung.
    batch_size : int
        Batchgroesse der Trainingsdaten.
    test_batch_sizes : List[int]
        Batchgroessen der Testdaten.
    data_root : str | Path
        Wurzel der .pt-Dateien.
    train_resolution : int, optional
        Aufloesung der Trainingsdaten (Default 32).
    test_resolutions : List[int], optional
        Aufloesungen der Testdaten (Default [32, 64]).
    encode_input, encode_output, encoding, channel_dim:
        wie ``load_darcy_flow_small``.
    generate : bool, optional
        Ob fehlende Dateien generiert werden (Default True).
    seed : int, optional
        Basis-Seed fuer die Generierung.
    ranges : dict, optional
        Strahlparameter-Grenzen fuer die Generierung.
    num_workers : int, optional
        Worker des Train-/Test-DataLoader (Default 1).

    Returns
    -------
    train_loader : torch.utils.data.DataLoader
        Trainings-DataLoader mit ``shuffle=True``.
    test_loaders : dict[int, torch.utils.data.DataLoader]
        Ein DataLoader je Aufloesung in ``test_resolutions`` (shuffle=False).
    data_processor : neuralop.data.transforms.DefaultDataProcessor
        Mit an den Trainingsdaten gefitteten Normalisierern.
    """
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
        generate=generate,
        seed=seed,
        ranges=ranges,
    )

    train_loader = DataLoader(
        dataset.train_db,
        batch_size=batch_size,
        shuffle=True,  # fehlt in der Darcy-Vorlage (Plan "Fallstricke")
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
