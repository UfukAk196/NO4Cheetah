"""G3 - Datei-Writer: erzeugt die ``.pt``-Dateien im PTDataset-Format.

Schreibt pro benoetigter Aufloesung eine Datei
    {root}/cheetah_sc_train_{res}.pt   und   {root}/cheetah_sc_test_{res}.pt
mit Inhalt ``torch.save({"x": x, "y": y, "meta": meta})``.

Formen:
    x (N, 4, res, res, res) float32 : rho_hat + 3x E_hat (dimensionslos)
    y (N, 1, res, res, res) float32 : phi_hat            (dimensionslos)
    meta : dict mit Strahlparametern je Probe + Header.

Seeds: Jede der insgesamt ``n_train + sum(n_tests)`` Proben bekommt einen
eindeutigen, fortlaufenden Seed ``seed + g`` (g = globaler Index). Dadurch
ueberschneiden sich Train- UND Test-Proben nie (auch nicht zwischen
verschiedenen Test-Aufloesungen); jede Probe ist reproduzierbar.
"""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

import cheetah

from .sampling import DEFAULT_RANGES, sample_beam_params
from .fields import DEFAULT_GRID_EXTENT, make_sample

DATASET_NAME = "cheetah_sc"


def _load_yaml_config(path: str | Path) -> dict:
    """Liest die CLI-Config (YAML). Nur ``yaml.safe_load``, keine Zwangs-Schema."""
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _probe_seed(base_seed: int, global_index: int) -> int:
    """Eindeutiger, reproduzierbarer Seed fuer Probe ``global_index``."""
    return base_seed + global_index


def _meta(
    base_seed: int,
    res: int,
    params_list: list[dict[str, Any]],
    global_indices: list[int],
) -> dict[str, Any]:
    """Sammelt Header- und Proben-Informationen fuer eine .pt-Datei."""
    return {
        "seed": base_seed,
        "res": res,
        "grid_extent": DEFAULT_GRID_EXTENT,
        "cheetah_version": cheetah.__version__,
        "date": date.today().isoformat(),
        "global_indices": global_indices,
        "params": params_list,
    }


def generate_cheetah_sc(
    root: str | Path,
    n_train: int,
    n_tests: list[int],
    train_resolution: int,
    test_resolutions: list[int],
    seed: int,
    ranges: dict[str, dict[str, float]] | None = None,
) -> list[Path]:
    """Erzeugt den Cheetah-Space-Charge-Datensatz als .pt-Dateien.

    Parameters
    ----------
    root : str | Path
        Zielverzeichnis fuer die .pt-Dateien (wird angelegt, falls noetig).
    n_train : int
        Anzahl Training-Proben.
    n_tests : list[int]
        Anzahl Test-Proben JE Test-Aufloesung (Laenge = len(test_resolutions)).
    train_resolution : int
        Aufloesung der Trainingsdaten.
    test_resolutions : list[int]
        Aufloesungen der Testdaten.
    seed : int
        Basis-Seed; alle Proben-Seeds werden daraus abgeleitet.
    ranges : dict, optional
        Strahlparameter-Grenzen (G1). Fehlt die Angabe, werden die
        TODO-Defaults von ``sampling.DEFAULT_RANGES`` verwendet.

    Returns
    -------
    list[Path]
        Pfade der geschriebenen .pt-Dateien.
    """
    if ranges is None:
        ranges = DEFAULT_RANGES

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    total = n_train + sum(n_tests)

    # Strahlparameter: eine deterministische Liste fuer alle Proben.
    all_params = sample_beam_params(total, seed=seed, ranges=ranges)

    written: list[Path] = []

    def _build_and_save(res: int, count: int, global_offset: int, kind: str) -> None:
        """Erzeugt ``count`` Proben ab ``global_offset`` und speichert sie.

        kind : "train" | "test" -> Dateiname.
        """
        xs: list[torch.Tensor] = []
        ys: list[torch.Tensor] = []
        meta_params: list[dict[str, Any]] = []
        global_indices: list[int] = []

        for i in tqdm(range(count), desc=f"{kind} @ {res}"):
            g = global_offset + i
            params = dict(all_params[g])
            params["seed"] = _probe_seed(seed, g)

            x, y = make_sample(params, res)
            if not (torch.isfinite(x).all() and torch.isfinite(y).all()):
                raise ValueError(
                    f"NaN/Inf in Probe {g} ({kind}_{res}.pt, params={params})"
                )

            xs.append(x)
            ys.append(y)
            meta_params.append(params)
            global_indices.append(g)

        # x: (count, 4, res, res, res), y: (count, 1, res, res, res)
        x_tensor = torch.stack(xs, dim=0)
        y_tensor = torch.stack(ys, dim=0)

        meta = _meta(seed, res, meta_params, global_indices)
        filename = root / f"{DATASET_NAME}_{kind}_{res}.pt"
        torch.save({"x": x_tensor, "y": y_tensor, "meta": meta}, filename)
        written.append(filename)

    # Trainingsdaten (eine Aufloesung).
    _build_and_save(train_resolution, n_train, 0, "train")

    # Testdaten (mehrere Aufloesungen), jeweils unabhaengige Proben via
    # fortlaufendem globalem Index -> keine Ueberschneidung untereinander.
    offset = n_train
    for res, n_test in zip(test_resolutions, n_tests):
        _build_and_save(res, n_test, offset, "test")
        offset += n_test

    return written


def main(argv: list[str] | None = None) -> None:
    """CLI: python -m cheetah_sc.generate --config configs/generate.yaml"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/generate.yaml",
        help="Pfad zur YAML-Config mit Generierungseinstellungen und ranges.",
    )
    args = parser.parse_args(argv)

    cfg = _load_yaml_config(args.config)
    ranges = cfg.get("ranges", DEFAULT_RANGES)
    written = generate_cheetah_sc(
        root=cfg["root"],
        n_train=cfg["n_train"],
        n_tests=cfg["n_tests"],
        train_resolution=cfg["train_resolution"],
        test_resolutions=cfg["test_resolutions"],
        seed=cfg["seed"],
        ranges=ranges,
    )
    print(f"Geschriebene Dateien ({len(written)}):")
    for path in written:
        print(f"  {path}")


if __name__ == "__main__":
    main()
