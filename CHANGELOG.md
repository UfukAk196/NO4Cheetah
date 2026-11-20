# Changelog

## v0.2.0 [🚧 Work in Progress]

### 🚨 Breaking Changes

- Change the data representation: log-transform ρ and φ (``log(field + eps)``) and z-score them channel-wise (neuralop ``UnitGaussianNormalizer``, fitted on the training data), append the beam parameters ``sigma_x``, ``sigma_y``, ``sigma_tau`` and ``gamma`` to ρ as constant input channels, and decode the model output back to φ [V] through ``LogExpDataProcessor``; this replaces the earlier analytic nondimensionalization (``rho_hat``/``phi_hat``) and its aspect-ratio channels
- Train a neuralop ``TFNO`` (input ``1 + len(input_params)`` channels, output φ) instead of the own ``FNO3d``/``FNO``
- Save the trained model as a neuralop checkpoint (``outputs/models/final_model_*`` via ``save_checkpoint``/``from_checkpoint``) together with the fitted data processor (``final_model_data_processor.pt``), instead of a bare state dict ``<name>_<run>.pth``; the ``Trainer`` keeps the best ``"32_h1"`` epoch under ``outputs/models/checkpoints/``
- Reorganize ``Pipeline/``: run scripts in ``exec/`` (``train_model.py``, ``evaluate_model.py``, ``tune_model.py``) with ``tests/`` and ``tutorials/``, the Slurm scripts and their ``.out``/``.err`` logs in ``scripts/``, and all results under ``outputs/`` (``models/``, ``figures/<RUN>/``, ``tables/<RUN>/``, ``studies/optuna/<RUN>/``, ``studies/wandb/``)
- Remove the YAML configs and the ``scno.data.generate`` CLI; data, model and training settings are the constants at the top of the run scripts (``scno.config`` keeps only paths and the global seed), and missing data is generated on the first training run
- Give every run a run ID ``<YYYY-MM-DD>_<NN>`` for its ``figures/``, ``tables/`` and ``studies/`` folders; the model file in ``models/`` is overwritten by every training run (no ``MODEL_NAME``)
- Use neuralop's ``H1Loss`` as the training loss (field plus gradient, open boundaries) and report ``H1`` and ``L2`` in evaluation

### 🚀 Features

- Generate the HDF5 datasets directly from Cheetah's ``SpaceChargeKick`` (``generate.py`` draws beam parameters within ``DEFAULT_RANGES`` and writes ρ and φ plus a reproducible ``params`` group), loaded by ``CheetahDataset``/``load_cheetah_sc`` on top of neuralop's ``H5pyDataset`` and ``DefaultDataProcessor``
- Add a distributed Optuna hyperparameter search (``exec/tune_model.py`` with the ``create``/``generate``/``work``/``report`` commands, TPE + Hyperband pruning on the H1 test loss, journal-file storage) and the Slurm scripts ``train-script.sh`` and ``tune-script.sh`` for Maxwell
- Record the neuralop ``Trainer``'s per-epoch metrics into ``history.csv`` and plot the training and evaluation loss curves (``plot_losses``)
- Evaluate potential and force as relative L2, including zero-shot super-resolution on a larger grid, and plot φ ground truth vs. prediction as projections and central cuts through Cheetah's ``ParticleBeam.plot_distribution``

### 🐛 Bug fixes

- Evaluate through the data processor in ``eval`` mode along a single ``predict`` path, so the model output is decoded back to φ [V] and models are no longer compared on the normalized field
- Append the beam-parameter channels to the input normalizer from their per-sample statistics, so a constant channel is standardized without loading the training set once per channel

### 🐆 Other

- Add project guidelines for AI agents (``AGENTS.md``)
- Translate comments, docstrings and the README to English
