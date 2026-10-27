"""data - Data generator for a Cheetah space-charge dataset.

Replaces the Zenodo download of a ``neuralop`` dataset with a generator of its own,
which produces 3D training data (rho -> phi) directly from
``cheetah.accelerator.SpaceChargeKick``.

Work packages (plan "Datengenerator fuer Cheetah-Space-Charge"):
    G1  sampling.py   - draw beam parameters
    G2  fields.py     - field pair (rho, phi) from Cheetah
    G3  generate.py   - write HDF5 files (write_hdf5)
    G4  dataset.py    - CheetahDataset (HDF5, nondimensionalization on loading)
    G5  dataset.py    - load_cheetah_sc(...)
"""
