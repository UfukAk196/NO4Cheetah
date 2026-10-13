"""cheetah_sc - Datengenerator fuer einen Cheetah-Space-Charge-Datensatz.

Ersetzt den Zenodo-Download eines `neuralop`-Datensatzes durch einen
eigenen Generator, der 3D-Trainingsdaten (rho_hat -> phi_hat) direkt aus
`cheetah.accelerator.SpaceChargeKick` erzeugt.

Arbeitspakete (Plan "Datengenerator fuer Cheetah-Space-Charge"):
    G1  sampling.py   - Strahlparameter aus config.yaml ziehen
    G2  fields.py     - Feld-Paar (rho_hat, phi_hat) aus Cheetah
    G3  generate.py   - .pt-Dateien schreiben (PTDataset-Format)
    G4  dataset.py    - CheetahDataset(PTDataset)
    G5  dataset.py    - load_cheetah_sc(...)
"""
