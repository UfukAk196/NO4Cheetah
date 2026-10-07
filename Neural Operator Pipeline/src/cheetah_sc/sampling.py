"""G1 - Parameter-Sampler fuer den Cheetah-Space-Charge-Datengenerator.

Zieht pro Probe einen Satz Strahlparameter aus den in ``config.yaml``
hinterlegten Grenzen:

- ``sigma_x``  [m]  (horizontal, log-uniform)
- ``sigma_y``  [m]  (vertikal,    log-uniform)
- ``sigma_tau``[m]  (longitudinal,log-uniform)
- ``energy``   [eV] (uniform)
- ``total_charge`` [C] (uniform)
- ``n_particles`` [int] (uniform)

Die drei raeumlichen Sigmas streuen ueber Groessenordnungen (µm..mm),
deshalb log-uniform; die uebrigen drei Grössen werden uniform gezogen.

Die Funktion ist deterministisch: gleicher ``seed`` -> identische Liste.
"""

from __future__ import annotations

from typing import Any

import numpy as np


# --------------------------------------------------------------------------- #
# Default-Bereiche (Platzhalter, TODO: durch echte Werte aus der ML-Pipeline
# ersetzen; die Config in ``configs/sampling.yaml`` hat dieselben Zahlen und
# ist der vorgesehene Einstiegsort).
# --------------------------------------------------------------------------- #
DEFAULT_RANGES: dict[str, dict[str, float]] = {
    # log-uniform: sigma_x in [min, max] Metern
    "sigma_x": {"min": 5.0e-4, "max": 2.0e-3},          # 0.5 .. 2 mm
    # log-uniform: sigma_y in [min, max] Metern
    "sigma_y": {"min": 5.0e-4, "max": 2.0e-3},          # 0.5 .. 2 mm
    # log-uniform: sigma_tau in [min, max] Metern
    "sigma_tau": {"min": 1.0e-3, "max": 8.0e-3},        # 1 .. 8 mm
    # uniform: Energie in [min, max] eV
    "energy": {"min": 5.0e8, "max": 2.0e9},             # 0.5 .. 2 GeV
    # uniform: Gesamtladung in [min, max] Coulomb
    "total_charge": {"min": 1.0e-10, "max": 5.0e-9},    # 0.1 .. 5 nC
    # uniform (ganzzahlig): Teilchenzahl pro Makropartikel-Bunch
    "n_particles": {"min": 5.0e4, "max": 5.0e5},
}

# Groessen, die log-uniform gezogen werden (siehe Docstring).
LOG_UNIFORM_KEYS = ("sigma_x", "sigma_y", "sigma_tau")

# Groessen, die uniform gezogen werden.
UNIFORM_KEYS = ("energy", "total_charge", "n_particles")


def _all_ranges_present(ranges: dict[str, dict[str, float]]) -> list[str]:
    """Prueft, ob alle sechs Parameter ein {"min": .., "max": ..} haben.

    Gibt die Liste der fehlenden Parameter zurueck (leer = ok).
    """
    missing: list[str] = []
    for key in LOG_UNIFORM_KEYS + UNIFORM_KEYS:
        if key not in ranges:
            missing.append(key)
            continue
        for bound in ("min", "max"):
            if bound not in ranges[key]:
                missing.append(f"{key}.{bound}")
    return missing


def sample_beam_params(
    n: int, seed: int, ranges: dict[str, dict[str, float]] | None = None
) -> list[dict[str, Any]]:
    """Ziehe ``n`` unabhaengige Strahlparameter-Saetze.

    Deterministisch: gleicher ``seed`` -> identische Ausgabe.

    Parameters
    ----------
    n : int
        Anzahl der Proben (Strahlparameter-Saetze).
    seed : int
        Seed fuer den zugrunde liegenden Zufallsgenerator.
    ranges : dict[str, dict[str, float]], optional
        Grenzen {"min": .., "max": ..} je Parameter. Fehlt die Angabe,
        werden ``DEFAULT_RANGES`` verwendet (TODO-Werte).

    Returns
    -------
    list[dict[str, Any]]
        ``n`` Dicts mit Keys ``sigma_x, sigma_y, sigma_tau`` [m],
        ``energy`` [eV], ``total_charge`` [C], ``n_particles`` [int].
    """
    if n < 0:
        raise ValueError(f"n muss >= 0 sein, war {n}")

    if ranges is None:
        ranges = DEFAULT_RANGES

    missing = _all_ranges_present(ranges)
    if missing:
        raise ValueError(
            "ranges muss fuer alle Parameter eine 'min'/'max' Angabe haben; "
            f"fehlt fuer: {missing}"
        )

    rng = np.random.default_rng(seed)

    params_list: list[dict[str, Any]] = []
    for _ in range(n):
        sample: dict[str, Any] = {}

        # Uebrige (uniform gezogene) Groessen zuerst, damit die Reihenfolge
        # der Zufallszahlen stabil bleibt, falls jemand die Menge der
        # log-uniformen Parameter aendert.
        for key in UNIFORM_KEYS:
            # float() erzwungen: YAML-Configs liefern Grenzen teils als Strings
            # (z.B. '5.0e8'), die sonst als str-Vergleich enden wuerden.
            lo, hi = float(ranges[key]["min"]), float(ranges[key]["max"])
            if lo > hi:
                raise ValueError(f"ranges[{key}]: min ({lo}) > max ({hi})")
            sample[key] = float(rng.uniform(lo, hi))

        # n_particles ist ganzzahlig.
        sample["n_particles"] = int(round(sample["n_particles"]))
        # Band-Ende konsistent halten (n_particles darf nie < 1 sein).
        sample["n_particles"] = max(1, sample["n_particles"])

        for key in LOG_UNIFORM_KEYS:
            lo, hi = float(ranges[key]["min"]), float(ranges[key]["max"])
            if lo <= 0 or hi <= 0:
                raise ValueError(f"ranges[{key}] muss strikt positiv sein")
            if lo > hi:
                raise ValueError(f"ranges[{key}]: min ({lo}) > max ({hi})")
            # log-uniform: exp(U(ln(min), ln(max)))
            sample[key] = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))

        params_list.append(sample)

    return params_list
