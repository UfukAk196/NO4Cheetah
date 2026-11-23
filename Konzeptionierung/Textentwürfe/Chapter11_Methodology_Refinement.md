# Chapter 11 "Next Notes" → Methodology — refinement guide

Reference for turning the Ch. 11 notes (paragraphs in `2026-11-19_NO-Surrogat Spezifikation und Methodik.docx`) into
the Methodology chapter about the (T)FNO. For each note you get:

- **Rewrite** — thesis-ready English, all information preserved, corrections marked **[CORR]** with reason + source.
- **Where to look** — the docx section and the code file(s)/lines to verify or expand.

All code facts below were read from the committed `main` (package `scno`, `Pipeline/`) and the installed
`neuralop==2.0.0` in env `no4cheetah`. Verified numbers:

| Claim | Committed code | Note says | Verdict |
|---|---|---|---|
| TFNO rank (Tucker) | `0.1` (neuralop default) | 10 % | ✓ correct |
| Model params | **16 033** (committed) / **68 761** (Optuna-best) | ~100 000 | ✗ state the real number |
| domain_padding | `0.25` | 12.5 % → 1.4× | ✗ value + math wrong |
| training samples | `n_train = 1000` | 1e6 | ✗ (1e6 = particles/beam) |
| kicks to make data | once, `effect_length=0`, no drift | 3× with drifts | ✗ |
| epochs | `2000`, no early stopping | 10000 + early stopping | ✗ |
| scheduler | off (`ConstantLR`) | plateau after 5000 | ✗ |
| eval losses | `h1` **and** `l2` | h1 | see comment 11 |
| Optuna params | 6 (n_modes, hidden_channels, n_layers, domain_padding, lr, weight_decay) | same 6 | ✓ for current study |
| Cheetah default grid | `grid_shape=(32,32,32)` | 32³ | ✓ correct |

---

## §255 — TFNO / Tucker factorization

> *"Model is a TFNO … Tucker Factorization helps to make the model parameters sparse … Currently Tucker factor is on its default 10 percent … prevent overfitting and effects like identity collapse …"*

**Rewrite (thesis):**
> The surrogate is a *Tensorized Fourier Neural Operator* (TFNO): an FNO whose spectral weight tensor of each
> Fourier layer is stored in Tucker-factorized form. The factorization is kept at the library default rank of
> 0.1, so the factorized spectral weights hold roughly 10 % of the parameters of the corresponding dense FNO
> [neuralop docs]. In our configuration this is confirmed directly: the dense FNO would have 682 921 spectral/channel
> parameters, the rank-0.1 TFNO 68 761 (10.1 %). Besides the memory saving, the reduced number of free parameters
> acts as a regularizer: low-rank weight sharing between neighbouring Fourier modes damps overfitting, which in our
> experiments also removed an *identity-collapse* failure mode (the model reproducing its input ρ instead of φ)
> observed with the dense parametrization.

**[CORR]** — the "10 %" is correct, but phrase it as *"~10 % of the parameters of a dense FNO"*, not "10 % of parameters are actually used."
Source: `neuralop==2.0.0`, `neuralop/models/fno.py` — `class TFNO` docstring: *"A TFNO with rank 0.1 has roughly 10% of the parameters of a dense FNO,"* and `TFNO.__init__` sets `kwargs.setdefault("rank", 0.1)`, `factorization="Tucker"`. The 10.1 % figure is my own `count_model_params` check of your Optuna-best config vs. its dense equivalent.

**Note:** *"identity collapse"* is your own empirical term, not a literature term — present it as an observation from your test runs, not a cited effect (addresses **comment 1 — "Link source"**: there is no external source for identity collapse; the citable claims are (i) rank-0.1 ≈ 10 % params → neuralop docs, and (ii) factorization as regularization → Kossaifi et al., *Multi-Grid Tensorized FNO*, arXiv:2310.00120, and your own spec §8 argument).

**Where to look**
- docx §7 (model), §8 "Tucker-Faktorisierung" (§228–229) — your own derivation of the parameter count and the low-rank/regularization argument; reuse it, it is stronger than §255.
- Code: `exec/train_model.py` → `model = TFNO(...)` (rank not passed → default 0.1). `src/scno/config.py`.
- To report the real count: `count_model_params(model)` is already printed in `train_model.py` ("Our model has … parameters"). Put *that* number in the thesis.

---

## §256 — domain_padding vs. Hockney

> *"domain padding … to substitute for Hockney Method … which doubled grid size … Domain padding … with 12,5 % only enlarges the grid … making the 3D-grid … roughly 1,4 times bigger."*

**Rewrite (thesis):**
> The classical solver uses the Hockney method: to turn the periodic FFT convolution into the non-periodic
> (open-boundary) Green's-function convolution it needs, ρ and the integrated Green's function are zero-padded to
> **twice** the grid size in every dimension and cropped back afterwards. Doubling all three axes multiplies the number
> of grid points by 2³ = 8, which is the main cost of the field solve. The FNO instead counters the FFT's artificial
> periodicity with its own `domain_padding`: the input is zero-padded by a fixed fraction *p* **per side** before the
> Fourier layers and unpadded afterwards. With the symmetric padding of `neuralop`, a padding fraction *p* enlarges each
> axis by a factor (1 + 2*p*), i.e. the padded grid has (1 + 2*p*)³ as many points. For the trained model
> (*p* = 0.25) this is 48³ instead of 32³, a factor of (1.5)³ ≈ 3.4; for the Optuna-selected *p* = 0.125 it is 40³,
> a factor (1.25)³ ≈ 1.95. Either way the padded grid is far smaller than the 8× of the Hockney doubling.

**[CORR 1]** — value: the trained model uses `domain_padding=0.25`, **not** 12.5 %. 12.5 % (0.125) is the *Optuna-best* value you have not yet copied into `train_model.py`. Decide which model the chapter documents and use that number.
Source: `exec/train_model.py` → `domain_padding=0.25`; Optuna-best in the comment block there (`domain padding = 0.125`) and docx §7 (§216).

**[CORR 2]** — math: a fraction *p* does **not** enlarge the grid "by this very amount." `neuralop` pads **both** sides, so each axis grows by (1 + 2*p*), and the 3-D grid by (1 + 2*p*)³. So 12.5 % → (1.25)³ ≈ **1.95×** (≈2×), not 1.4×. The 1.4× you wrote is what one-sided padding (1 + *p*) would give: (1.125)³ ≈ 1.42. For the committed 25 %: (1.5)³ ≈ **3.4×**.
Source: `neuralop==2.0.0`, `neuralop/layers/padding.py`, `DomainPadding.pad`: `padding = [round(p * r) for p, r in zip(...)]` then `padding = [i for p in padding for i in (p, p)]` (printed as `"symmetric"`), i.e. `round(p·r)` on *each* side.

**Where to look**
- docx §2 "Der klassische Poisson-Solve" (§45–63) and §7 "Padding" (§217) — your Hockney explanation lives there; §256's "2³ = 8 times longer" is right and matches §63.
- Code: `exec/train_model.py` (`domain_padding`); `neuralop/layers/padding.py` for the semantics; Cheetah `SpaceChargeKick._array_rho` / `_integrated_green_function` for the doubling (docx §45 table, blocks 5–6).

---

## §257 — parameter count and data size

> *"Model has roughly 100.000 Parameters and was trained on 1e+6 data samples."*

**Rewrite (thesis):**
> The trained model has **[N]** parameters and was trained on **1000** samples at resolution 32³ (125 + 125 test
> samples at 32³ and 64³). Each sample is one beam of 10⁶ macroparticles; the 10⁶ is the particle count per beam, not
> the number of training samples.

**[CORR 1]** — "1e6 data samples" is wrong: `n_train = 1000`. The 10⁶ is `n_particles` (macroparticles per beam), fixed to 1e6.
Source: `src/scno/config.py` → `DATA["n_train"] = 1000`, `n_tests = [125, 125]`; `src/scno/data/generate.py` → `DEFAULT_RANGES["n_particles"] = {"min": 1e6, "max": 1e6}`.

**[CORR 2]** — "~100 000 parameters": the committed config (`n_modes=(5,5,5)`, `hidden_channels=16`, `n_layers=3`, rank 0.1) has **16 033** parameters; the Optuna-best config (`n_modes=13`, `hidden_channels=12`, `n_layers=2`) has **68 761**. Neither is 100k — insert the actual printed count for the model you report.
Source: my `count_model_params` runs on both configs (`exec/train_model.py` prints the same number at run time).

**Where to look**
- docx §5 "Aktuelle Mengen und Parameter-Bereiche" (§100–112) for the counts; §7 for the model config.
- Code: `src/scno/config.py` (`DATA`), `exec/train_model.py` (model + printed param count).

---

## §258 — data generation (beams, kick, extraction)

> *"generated by Cheetah … via from_parameters … gauss distributed … Then Space_charge_Kick was applied on them 3 times with drift-functions inbetween. Within each space charge kick phi and rho were extracted …"*

**Rewrite (thesis):**
> Each training pair (ρ, φ) is produced with Cheetah's own `SpaceChargeKick`. A beam is created with
> `ParticleBeam.from_parameters`, which samples a Gaussian particle cloud from the beam parameters. For that beam the
> generator reproduces the deposition and field-solve blocks of the kick: it deposits the charge density ρ with
> `_array_rho` and solves the potential φ with `_solve_poisson_equation`, storing the (ρ, φ) pair on the 32³ grid in
> physical units. The kick is evaluated once, with zero effect length — no drifts and no momentum update are applied,
> because only the solver's input ρ and output φ are needed as supervised training data.

**[CORR]** — "applied 3 times with drift-functions inbetween" is not what the generator does. In `generate.py::_simulate` the kick is a single zero-length evaluation (`SpaceChargeKick(effect_length=torch.tensor(0.0), ...)`): it calls `_array_rho` once and `_solve_poisson_equation` once, returns ρ and φ, and never tracks through drifts. (The "3 kicks + drifts" pattern is the *cold-beam expansion test* in `tests/test_space_charge_kick.py`, your docx §3 example — a different thing.) → answers **comment 4 ("Or only space charge kick applied once?")**: once, and no drifts.
Source: `src/scno/data/generate.py` → `_write_split` (`sck = SpaceChargeKick(effect_length=0.0, ...)`), `_simulate` (`rho = sck._array_rho(...)[...]`, `phi = sck._solve_poisson_equation(...)`).

**Where to look**
- docx §5 "Datengenerierung" (§96–99) — accurate; align §258 with it.
- Code: `src/scno/data/generate.py` (`_build_beam`, `_simulate`, `_write_split`). For "Gaussian": Cheetah `ParticleBeam.from_parameters`.

---

## §259 — parameter sampling (ranges, distribution)

> *"beams are generated via samples with uniformly distributed parameters inbetween specific ranges according to table (SAMPLE PARAMETERS WITH RANGES)."*

**Rewrite (thesis):**
> The beam parameters of each sample are drawn independently and uniformly from fixed ranges (Table **[ranges]**). The
> sigmas can optionally be drawn log-uniformly; that path is currently disabled, so all parameters — including the
> sigmas — are uniform.

**OK as written** (matches current code) — just insert the ranges table from docx §5 (§113–141). → answers **comment 8 ("Can I write that so?")**: yes, "uniformly distributed" is correct for the committed generator.
Source: `src/scno/data/generate.py` → `_draw_params` draws every key in `_UNIFORM_KEYS` (which includes the sigmas) with `rng.uniform`; `_LOG_UNIFORM_KEYS` and the log block are commented out.

**Caveat to fix elsewhere:** the code *docstrings* still say "sigmas log-uniform" (module docstring, `FEATURES` comment in `dataset.py`). They are stale; don't cite them. The table caption is **comment 14's "SOURCE"** target too — it is `DEFAULT_RANGES` in `generate.py`.

**Where to look**
- docx §5 table (§113–141) → the "SAMPLE PARAMETERS WITH RANGES" table; values = `DEFAULT_RANGES`.
- Code: `src/scno/data/generate.py` → `DEFAULT_RANGES`, `_draw_params`, `_UNIFORM_KEYS`.

---

## §260 — inputs of the model (why sigmas + gamma)

> *"Model takes rho together with sigma_x,_y,_tau, as well as gamma … Because phi is not only dependent on rho … but also on the cell sizes dx, dy, dz = dtau*gamma, as can be seen in equation (IGF). Then Phi is the output."*

**Rewrite (thesis):**
> The surrogate maps ρ to φ, but φ does not depend on ρ alone: the integrated Green's function depends on the cell
> sizes (Δx, Δy, γ·Δτ) (Eq. IGF). With a grid fixed to ±(extent·σ) per axis, the cell sizes are set by the beam
> sigmas and by γ. The model therefore receives, besides ρ, the four scalars σ_x, σ_y, σ_τ and γ as additional inputs,
> and outputs φ.

**OK** — physically correct and matches the code's `input_params`. One precision to add: γ enters only through the τ-axis (Δz̃ = γ·Δτ), which is why it appears together with the sigmas.
Source: `src/scno/config.py` → `DATA["input_params"] = ("sigma_x","sigma_y","sigma_tau","gamma")`; `src/scno/data/dataset.py` → `FEATURES`; docx §4 "Wovon φ tatsächlich abhängt" (§74–77) and §5 "Input-Kanäle" (§165–185).

→ relates to **comment 5** (docx anchor "beam" → "Or kick parameters?"): these four are *beam* parameters, but they enter the kick precisely as the solver's cell-size/γ dependence, so "beam parameters that determine the kick's grid" is the accurate phrasing.

**Where to look**
- docx §4 (§74–77, the φ = (1/4πε₀)·ρ∗IGF(dx,dy,γ·dτ) box) and §6 §194 — the scaling/anisotropy argument.
- Code: `dataset.py::FEATURES`, `compute_features`, `CheetahH5Dataset.__getitem__`.

---

## §261 — tensor shapes and channel expansion

> *"data is fed … as (B x 5 x res x res x res) … For that the scalars sigma_x,_y,_tau and gamma are upscaled to (1 x res x res x res). Phi comes out as (B x 1 x res x res x res)."*

**Rewrite (thesis):**
> Each input is a tensor of shape (B, 5, res, res, res): channel 0 is ρ on the res³ grid (res = 32 for training, 32 or
> 64 for testing), channels 1–4 are the scalars σ_x, σ_y, σ_τ, γ, each broadcast to a constant res³ channel. The output
> is φ of shape (B, 1, res, res, res).

**OK** — exact. (docx §5 channel table §167–184 and the shape table §143–163 confirm (64,5,32,32,32) for a batch.) → **comment 6 ("Maybe leave out")** is attached to the §6-version of this sentence (§261 duplicates docx §5/§6). You can safely drop the duplicate: keep the channel-expansion explanation once, in the data section, and reference it from the model section.
Source: `src/scno/data/dataset.py` → `CheetahH5Dataset.__getitem__` (`channels = self.features[idx][:,None,None,None].expand(...)`, `torch.cat([rho, channels])`).

**Where to look**
- docx §5 "Form der Daten" (§143–163) and "Input-Kanäle" (§165–185).
- Code: `dataset.py::CheetahH5Dataset.__getitem__`.

---

## §262 — normalization pipeline

> *"Before the data goes into the model, it is normalized … applying log on rho and phi only. Then all data channels (log-rho, log-phi, sigma_x, sigma_y, sigma_tau, gamma) undergo z-score."*

**Rewrite (thesis):**
> The data are normalized in two stages. First a log transform is applied to ρ and φ only, log(field + ε). Then every
> channel — log ρ, the four parameter channels, and the target log φ — is standardized channel-wise with a
> `UnitGaussianNormalizer` whose mean and standard deviation are fitted on the training set only, z = (field − μ)/σ. At
> evaluation time the processor inverts the z-score and the log, so φ is compared in volts, not in normalized space.

**OK** — matches code. Small precision: the parameter channels are **not** log-transformed (only ρ and φ are), then z-scored like everything else.
Source: `src/scno/data/dataset.py` → `_log`/`_exp`, `CheetahH5Dataset.__getitem__` (logs ρ, φ), `CheetahDataset.__init__` + `_fit_normalizer` + `_append_features` (z-score, train-only fit), `LogExpDataProcessor.postprocess` (decode); docx §6 (§187–195).

**Where to look**
- docx §6 "Datenrepräsentation & Normalisierung" (§187–195).
- Code: `dataset.py` (`_log`, `LogExpDataProcessor`, `UnitGaussianNormalizer` usage).

---

## §263 — why log ρ and φ

> *"Log on rho and phi is important, because even tho beams are sampled from uniformly distributed parameters, rho and phi come out with heavily skewed log-distributions with long tails. So applying log makes them normally distributed … z-score … on the other input channels is no issue as they are actually part of the parameters that are used for sampling, i.e. uniformly distributed."*

**Rewrite (thesis):**
> ρ and φ are log-transformed because, although the beam parameters are sampled uniformly, the resulting fields are
> strongly right-skewed with long tails (ρ ≥ 0 with ~57 % zeros; φ > 0 with a minimum near e⁵·³ V). The log maps them
> to roughly bell-shaped distributions, so the subsequent z-score is meaningful. The four parameter channels are left
> on their linear scale and only z-scored, because they are the sampling parameters themselves and are already
> uniformly distributed over bounded ranges.

**OK** — keep. → addresses **comment 9 ("Maybe too much")** on "heavily skewed log-distributions with long tails": it is not too much — it is the justification for the log, so keep it, but you can tighten to one sentence and back the "~57 % zeros / min ≈ e⁵·³" with your own histogram. Mild **[CORR]**: "makes them normally distributed" → say *"approximately normal / much less skewed"* (a log transform does not guarantee normality); a log-normal field becomes exactly normal, a general skewed field only approximately.
Source: `src/scno/data/dataset.py` → `_log` comment ("rho ≥ 0, ~57% zeros and phi > 0 (min ~ e^5.3)"); docx §6 (§189).

**Where to look**
- Histograms to support the claim: `exec/tests/hist_raw.py`, `exec/tests/hist_processed.py` (raw vs. post-log/z-score distributions) — cite one of these figures.
- docx §6 (§189).

---

## §264 — training length / stopping / loss drop

> *"trained for 10000 epochs with early stopping and checkpointing, so the best model … is saved and training is stopped, when validation loss rises for 5 consecutive steps … after training the loss is lowered by approx 2 orders of magnitude."*

**Rewrite (thesis) — committed run:**
> The model is trained for 2000 epochs with AdamW. The best epoch by the 32³ H1 validation loss is checkpointed and
> restored at the end (best-model saving); training is not stopped early. Over training the validation H1 loss falls by
> about **[1.5]** orders of magnitude (Fig. **[loss curve]**).

**[CORR 1]** — epochs: committed `epochs = 2000`, not 10000. The inline TODO there reads *"goal: 10000; better 2000, since overfitting starts after 2500."* If you train the long Optuna schedule (~40000), state that number instead — but match it to the model you actually report.
Source: `exec/train_model.py` → `epochs = 2000`.

**[CORR 2]** — "early stopping … stopped when validation loss rises 5 consecutive steps" is **not** implemented. The neuralop `Trainer` runs all `n_epochs`; what you have is *best-checkpoint saving* (`save_best="32_h1"`), i.e. the lowest-validation-loss epoch is kept, but training does not halt. There is no patience-based early stop in the loop. Either implement early stopping and then keep the sentence, or describe it as best-checkpoint saving.
Source: `exec/train_model.py` → `trainer.train(..., save_best="32_h1", ...)`; no `EarlyStopping`/patience in the call. (The "5 consecutive" you may be recalling is the scheduler's `patience=5`, see §267.)

**[CORR 3]** — "approx 2 orders of magnitude": on the committed run (`outputs/tables/2026-11-19_01/history.csv`) the training loss drops 0.999 → ~0.019 (~1.7 orders) and the 32³ H1 validation 0.99 → ~0.04 (~1.4 orders). "About 1.5 orders" is safer; read the exact drop off your final loss-curve figure.
Source: `outputs/tables/<RUN>/history.csv`, `src/scno/evaluate.py::plot_losses`.

**Where to look**
- docx §7 "epochs" (§214–215) and §9.
- Code: `exec/train_model.py` (`epochs`, `save_best`, `Trainer`); `outputs/tables/<RUN>/history.csv`; figure `outputs/figures/<RUN>/loss_curves.png`.

---

## §265 — H1 loss and the force = gradient argument

> *"For training and validation h1-loss is evaluated … H1-loss not only considers the deviation … (L2/MSE) … but also the loss between their gradients … important … because … E_plus_vb_field is computed with the gradient of phi i.e. E = -grad phi * sth."*

**Rewrite (thesis):**
> Training uses the H1 loss (relative, d = 3, with open boundaries: non-periodic in x, y, z). Unlike a pure L2 loss on
> the function values, the H1 loss also penalizes the error of the gradient. This is important here because the force on
> the particles is the gradient of the potential: Cheetah computes the field in `_E_plus_vB_field` from ∇φ,
> **F** = −q∇φ/γ². A loss on φ alone could leave the gradient — the quantity that actually drives the kick — poorly
> matched, which the H1 term prevents.

**OK** — correct. Two additions answer **comment 12 ("Is this the only reason?")** — no, there are two further reasons, both already in your spec §6 (§193):
1. φ is only determined up to an additive constant (the integrated Green's function is fixed only up to a constant), so the *absolute* value of φ is physically irrelevant; only ∇φ matters — which argues for a gradient-sensitive loss over pure L2.
2. Differentiation amplifies high-frequency error, so matching the field well in L2 does not guarantee a good gradient; penalizing the gradient error directly addresses this.
Source: `exec/train_model.py` → `h1loss = H1Loss(d=3, periodic_in_x/y/z=False)`, `train_loss = h1loss`; Cheetah `SpaceChargeKick._E_plus_vB_field` (gradient of φ, `igamma2 = 1/γ²`); docx §6 (§193) and §9 (§235).

**[note on "E = -grad phi * sth"]** — write it properly: the force is **F** = −q∇φ/γ² (your PDF's step-4 equation). Fill the "sth" with −q/γ².

**Where to look**
- docx §9 "Verlustfunktion" (§234–236) and §6 §193 (offset/H1 motivation).
- Code: `exec/train_model.py` (`H1Loss`); Cheetah `SpaceChargeKick._E_plus_vB_field` for the ∇φ/γ² factor (docx §2 §66).

---

## §266 — training/testing resolution, zero-shot super-resolution

> *"Training occurs on 32³-grid, as this is the default grid mostly used in Cheetah applications (SOURCE). Testing is done on 32³ as well as 64³, to test zero-shot super resolution …"*

**Rewrite (thesis):**
> Training is done on a 32³ grid, which is the default grid of Cheetah's `SpaceChargeKick`. The model is evaluated on
> 32³ and, without retraining, on 64³ to test zero-shot super-resolution — the resolution transfer that distinguishes
> neural operators from fixed-grid CNNs. The 32³/64³ comparison on the same beam is shown in Fig. **[res study]**,
> alongside the loss curve.

**OK**, and the "SOURCE" is concrete: Cheetah's `SpaceChargeKick` has `grid_shape=(32,32,32)` as its constructor default (verified). Cite the Cheetah source/docs for that default rather than "mostly used."
Source: Cheetah `SpaceChargeKick.__init__` default `grid_shape=(32,32,32)` (verified via `inspect`); `src/scno/config.py` → `train_resolution=32`, `test_resolutions=[32,64]`; docx §5 (§97).

**Where to look**
- docx §5 (§97) and §9 "Resolution-Study" (§241).
- Code: `src/scno/config.py` (`DATA`); `exec/evaluate_model.py` / `src/scno/evaluate.py` (zero-shot eval, plots). Figure: `outputs/figures/<RUN>/`.

---

## §267 — scheduler

> *"The scheduler gets activated after epoch 5000 and it is a plateau-scheduler, that halves the learning rate after 5 consecutive epochs, where the training loss stagnated."*

**Rewrite (thesis) — two options, pick the truthful one:**
> *(committed)* No learning-rate schedule is used; the learning rate is held constant (AdamW, lr = 1e-3).
>
> *(if you enable it)* A `ReduceLROnPlateau` schedule is used: the learning rate is halved after 5 epochs without
> improvement of the monitored loss.

**[CORR]** — as committed, the scheduler is **off**: `b_scheduler = False` selects `ConstantLR(factor=1.0)`, i.e. a constant learning rate. When enabled it is `ReduceLROnPlateau(patience=5, factor=0.5)` — halve after 5 stagnant epochs. There is **no "activate after epoch 5000"** logic anywhere; drop that. (The scheduler monitors its default metric; it is not tied to "training loss" specifically unless you pass that metric.)
Source: `exec/train_model.py` → `b_scheduler = False` → `ConstantLR`; the disabled branch is `ReduceLROnPlateau(optimizer, patience=5, factor=0.5)`. Same in `exec/tune_model.py`.

→ This also answers **comment 13 ("Maybe also check this with optuna together with scheduler")**: the scheduler is currently not in the Optuna search space (`tune_model.py::suggest` tunes only the 6 model/optimizer params, with `b_scheduler=False`). Adding on/off + patience/factor to the search is a clean, honest extension to mention as future work.

**Where to look**
- docx §7 "scheduler" (§212–213) — already says *"ConstantLR (ReduceLROnPlateau optional)"*, which is the correct statement; §267 contradicts it. Trust §213.
- Code: `exec/train_model.py` and `exec/tune_model.py` (the `b_scheduler` block).

---

## §268 — optimizer

> *"As optimizer AdamW with weight decay was used, as this is a general optimizer which can be used for a broad band of cases."*

**Rewrite (thesis):**
> The optimizer is AdamW (lr = 1e-3, weight_decay = 1e-3). AdamW is used as a robust default: adaptive first/second-moment
> estimates with decoupled weight decay, which works well across a broad range of problems without per-problem tuning,
> and whose weight decay provides mild regularization in addition to the low-rank factorization.

**OK** — add the concrete lr/weight_decay values, and (optionally) one real reason beyond "general": decoupled weight decay (Loshchilov & Hutter, *Decoupled Weight Decay Regularization*, ICLR 2019) is the AdamW-specific property worth naming.
Source: `exec/train_model.py` → `AdamW(model.parameters(), lr=1.0e-3, weight_decay=1.0e-3)`; docx §7 (§211).

**Where to look**
- docx §7 (§210–211).
- Code: `exec/train_model.py` (optimizer line). → relates to **comment 13** (scheduler + optuna, see §267).

---

## §269 — Optuna-tuned hyperparameters

> *"The initial learning rate, weight decay, number of hidden channels, number of hidden layers (n_layers), max. number of k_modes/frequency modes (n_modes) …, domain_padding, were determined by an optuna study (SOURCE)."*

**Rewrite (thesis):**
> Six hyperparameters were selected with an Optuna study: the initial learning rate, the weight decay, the number of
> hidden channels, the number of layers, the number of retained Fourier modes per dimension (n_modes), and the
> domain_padding. n_modes was searched up to the Nyquist limit of the 32³ training grid (≤16). The remaining training
> settings (optimizer family, loss, batch size, epochs, scheduler) were fixed.

**OK for the current study** — the six you list are exactly `tune_model.py::suggest`'s search space. → answers **comment 14 ("Is that all studied parameters?")**: for the committed `tune_model.py`, yes, those six. **Caveat:** an earlier study (the `2026-10-29` best-result file) additionally tuned `gradient_weight` (the weight of the gradient term in the H1 loss) and `epochs`. If you cite that study's result, say so; if you cite the current `tune_model.py`, the six are complete.
Source: `exec/tune_model.py::suggest` — `n_modes` (3..res//2), `hidden_channels` (2..16, log), `n_layers` (1..12), `domain_padding` ∈ {0.125,0.15,0.2,0.25}, `learning_rate` (1e-4..1e-1, log), `weight_decay` (1e-6..1e-2, log). Earlier scope: `outputs/studies/optuna/2026-10-29_03/…best result.txt` (has `gradient_weight`, `epochs`).

**"SOURCE"** = your own study artifacts: `outputs/studies/optuna/<RUN>/` (journal `.log`, `trials.csv`, best-result `.txt`) and the study `2026-11-09_02` referenced in `train_model.py`. Cite the run ID + best value.

**Where to look**
- docx §7 "Optuna-Bestwerte" (§216).
- Code: `exec/tune_model.py` (search space + `work`/`objective`); study outputs under `outputs/studies/optuna/`.

---

# The 15 document comments — quick resolution map

Comments 1–14 are anchored in the **spec chapters** (not Ch. 11), but most overlap the Ch. 11 content above. Where to look / what to do:

| # | Anchor (abridged) | Resolution → see |
|---|---|---|
| 0 | "parameters" ("Which ones exactly?") | The parameters φ depends on: ρ, Δx, Δy, γ·Δτ → the 4 input channels σ_x,σ_y,σ_τ,γ. docx §4/§5; code `dataset.py::FEATURES`. See **§260**. |
| 1 | Tucker prevents overfitting ("Link source") | No source for "identity collapse" (your observation). Citable: neuralop TFNO docs (rank 0.1 ≈ 10 %) + Kossaifi et al. arXiv:2310.00120 (factorization = regularizer). See **§255**. |
| 2 | Hockney doubling ("Maybe explain more") | Expand: doubling each axis (2³=8× points) makes the periodic FFT convolution equal the open-boundary one; crop afterwards. docx §2 §63. See **§256**. |
| 3 | "12.5 % → 1.4× bigger" ("Check numbers") | **Wrong**: committed padding is 0.25; neuralop pads both sides → (1+2p)³. 0.125→~1.95×, 0.25→~3.4×. See **§256 [CORR 2]**. |
| 4 | "kick applied 3× with drifts" ("…or once?") | **Once**, no drifts (`generate.py`, effect_length=0). See **§258 [CORR]**. |
| 5 | "beam" ("Or kick parameters?") | They are beam parameters that fix the kick's cell size/γ; phrase as "beam parameters entering the solver's grid." See **§260**. |
| 6 | channel upscaling ("Maybe leave out") | Duplicate of §5/§6 text; keep the explanation once (data section), drop the repeat. See **§261**. |
| 7 | "sigma_x,_y,_tau" (combine in future?) | Design note, not a correction. Put in Future Work: combining the sigmas vs. keeping them separate so the model learns their relation. docx §10. |
| 8 | "uniformly distributed parameters" ("Can I write that so?") | Yes — current generator is uniform (log path commented out). See **§259**. |
| 9 | "heavily skewed log-distributions" ("Maybe too much") | Keep — it justifies the log. Soften "normally distributed" → "approximately normal." Back with `hist_raw.py`. See **§263**. |
| 10 | "10000 epochs + early stopping…" ("Check numbers!") | Committed: 2000 epochs, no early stopping (best-checkpoint saving). See **§264 [CORR 1,2]**. |
| 11 | "h1-loss" ("…can't argue for l2-loss for validation") | Code currently evaluates **both** h1 and l2 (`eval_losses = {"h1":…, "l2":…}` in `train_model.py`). To match your intent, drop l2 from `eval_losses` (keep only for the final report if wanted). `save_best` already uses `32_h1`. See fix note below. |
| 12 | "E from grad phi" ("Is this the only reason?") | No — also: φ defined up to a constant (only ∇φ matters) and differentiation amplifies high-freq error. docx §6 §193. See **§265**. |
| 13 | AdamW ("check with optuna + scheduler") | Scheduler is off and not in the Optuna search; add it as future work. See **§267/§268**. |
| 14 | optuna params ("Is that all studied?") | Current `tune_model.py`: those 6. Earlier study also tuned gradient_weight + epochs. See **§269**. |

**Comment 11 — concrete code fix** (if you want l2 gone from validation): in `exec/train_model.py`,
`eval_losses = {"h1": h1loss, "l2": l2loss}` → `eval_losses = {"h1": h1loss}`. The same `eval_losses` dict is passed to
`trainer.train(...)`, and `save_best="32_h1"` already keys on H1, so dropping l2 changes only what gets logged, not model selection.
(`exec/tune_model.py` similarly builds `eval_losses` with both — change there too for consistency.)
