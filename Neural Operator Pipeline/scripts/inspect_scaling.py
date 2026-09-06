"""
Diagnose: wie gut ist die Entdimensionalisierung eigentlich?

Wenn phi_hat ueber die Proben hinweg noch mehrere Groessenordnungen streut,
muss das Netz weiter Skalen lernen statt Physik -- und der relative Fehler
bleibt hartnaeckig hoch, egal wie gross man das Modell macht. Dieses Skript
misst die Streuung und vergleicht die beiden Bezugslaengen gegeneinander.

Ein Ergebnis dieser Art gehoert in die Thesis, weil es eine Design-
Entscheidung mit Zahlen belegt statt mit Bauchgefuehl.

Aufruf: python scripts/inspect_scaling.py --data data/train.h5
"""

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def spread(values: np.ndarray) -> float:
    """Streuung in Dekaden zwischen 5. und 95. Perzentil."""
    lo, hi = np.percentile(values, [5, 95])
    return float(np.log10(hi / max(lo, 1e-300)))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/train.h5")
    a = p.parse_args()

    with h5py.File(a.data, "r") as f:
        phi = f["phi_hat"][:]
        aspect = f["aspect"][:]
        cell = f["cell_size"][:]
        gamma = f["gamma"][:]
        charge = f["total_charge"][:]
        length = f["reference_length"][:]

    std_hat = phi.reshape(len(phi), -1).std(axis=1)

    # Gegenprobe: dieselben Potentiale mit der transversalen Boxkante Ex als
    # Bezugslaenge statt des geometrischen Mittels aller drei Kanten.
    scaled_ex = std_hat * aspect[:, 0]

    print(f"Datensatz: {a.data}, {len(phi)} Proben, Gitter {phi.shape[1:]}")
    print()
    print("Rohe Groessen (das, was ohne Entdimensionalisierung ins Netz ginge):")
    print(f"  Gesamtladung    : {spread(charge):5.2f} Dekaden")
    print(f"  Zellgroesse dx  : {spread(cell[:, 0]):5.2f} Dekaden")
    print(f"  gamma           : {gamma.min():.1f} .. {gamma.max():.1f}")
    print()
    print("Nach Entdimensionalisierung, Streuung von std(phi_hat):")
    print(f"  Bezug L = (Ex*Ey*Ez)^(1/3) : {spread(std_hat):5.2f} Dekaden   <- verwendet")
    print(f"  Bezug L = Ex               : {spread(scaled_ex):5.2f} Dekaden")
    print()
    print("Achsenverhaeltnisse E_i / L (gehen als log ins Netz):")
    for i, name in enumerate("xyz"):
        col = aspect[:, i]
        print(f"  E{name}/L: {col.min():9.3e} .. {col.max():9.3e}")
    print()
    print("Faustregel: bleibt die Streuung von phi_hat unter etwa einer Dekade,")
    print("ist die Skalierung in Ordnung. Deutlich mehr heisst, dass eine")
    print("separate Skalarvorhersage (zweiter Kopf fuer log|phi|) lohnt.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
