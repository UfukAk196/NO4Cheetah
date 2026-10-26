# Plan: Datengenerator für Cheetah-Space-Charge (LLM-gestützt)

Oct 5, 2026 · @Ufuk

## Ziel und Zielschnittstelle

Am Ende steht eine Funktion `load_cheetah_sc(...)`, die sich exakt so aufrufen lässt wie `load_darcy_flow_small(...)` und dieselben drei Objekte zurückgibt. Der Trainingsteil (Schritt 2) bleibt dadurch Zeile für Zeile unverändert.

```python
train_loader, test_loaders, data_processor = load_cheetah_sc(
    data_root="data/cheetah_sc",
    n_train=1000,
    batch_size=16,
    n_tests=[100, 50],
    test_resolutions=[32, 64],
    test_batch_sizes=[16, 4],
    train_resolution=32,
    generate=True,          # statt download=True: fehlende .pt-Dateien selbst erzeugen
)
data_processor = data_processor.to(device)
```

Vertrag, an dem jede Teilaufgabe gemessen wird:

- `train_loader` liefert Batches als Dict `{"x": Tensor, "y": Tensor}` mit Form `(B, C, D, H, W)`.
- `test_loaders` ist ein Dict `{Auflösung: DataLoader}`, ein Eintrag pro Wert in `test_resolutions`.
- `data_processor` ist ein `DefaultDataProcessor` mit an den Trainingsdaten gefitteten Normalisierern.
- Inhalt nach Schnitt A der Schnittstellendefinition: `x` = entdimensionalisierte Ladungsdichte ρ̂ plus Achsenverhältnis-Kanäle, `y` = entdimensionalisiertes Potential φ̂.
- Batchgrößen sind gegenüber Darcy kleiner gewählt, weil 3D-Gitter (32³ = 32 768 Punkte pro Kanal) viel mehr Speicher pro Probe brauchen als 2D-Darcy (16² = 256).

## Architektur nach der Skizze

Neu zu schreiben sind nur drei Dinge: ein Generator, der `.pt`-Dateien erzeugt, die Klasse `CheetahDataset` und die Funktion `load_cheetah_sc`. `PTDataset`, `DefaultDataProcessor` und `DataLoader` werden unverändert aus neuralop bzw. PyTorch übernommen.

| Darcy-Vorlage (neuralop) | Cheetah-Gegenstück | Datei | Neu oder übernommen | Was sich ändert |
| --- | --- | --- | --- | --- |
| Zenodo-Download (`download_from_zenodo_record`) | `generate_cheetah_sc(...)` | `data/generate.py` | neu | Erzeugt die Daten selbst aus Cheetah statt sie herunterzuladen |
| `DarcyDataset(PTDataset)` | `CheetahDataset(PTDataset)` | `data/dataset.py` | neu | `generate`-Flag statt `download`, eigene erlaubte Auflösungen, `channels_squeezed=False` |
| `PTDataset` | `PTDataset` | neuralop `pt_dataset.py` | übernommen | nichts; liest `{name}_train_{res}.pt` und `{name}_test_{res}.pt` |
| `load_darcy_flow_small(...)` | `load_cheetah_sc(...)` | `data/dataset.py` | neu | gleiche Signatur und Rückgabe, Auflösungen frei wählbar |
| `DefaultDataProcessor` | `DefaultDataProcessor` | neuralop `data_processors.py` | übernommen | nichts; kommt über `dataset.data_processor` zurück |
| `torch.utils.data.DataLoader` | `DataLoader` | PyTorch `dataloader.py` | übernommen | kleinere Batches, `num_workers` konfigurierbar |

Datenfluss in einer Zeile: Generator → `.pt`-Dateien → `CheetahDataset.__init__` (via `PTDataset`) → `train_db`, `test_dbs`, `data_processor` → `load_cheetah_sc` baut daraus die `DataLoader` → `Trainer`. Das entspricht der Skizze vom 05.10.2026 (`Datengenerierung_Architektur.svg`), ergänzt um den Generator als Ersatz für den Download.

## Datenformat und Konventionen

Der Generator schreibt pro Auflösung zwei Dateien, die `PTDataset` ohne jede Anpassung einliest. Alles Weitere ergibt sich aus `pt_dataset.py`.

| Element | Festlegung | Grund im Code |
| --- | --- | --- |
| Dateinamen | `cheetah_sc_train_{res}.pt`, `cheetah_sc_test_{res}.pt` | `PTDataset` baut `f"{dataset_name}_train_{train_resolution}.pt"` |
| Inhalt | `torch.save({"x": x, "y": y, "meta": {...}}, pfad)` | gelesen werden nur `data["x"]` und `data["y"]`, weitere Keys werden ignoriert |
| Form `x` | `(N, 4, res, res, res)`: ρ̂ + drei konstante Kanäle `log(E_i/L)`, E_i = Boxkanten (nicht das E-Feld!) | mehrere Kanäle, deshalb `channels_squeezed=False`, `channel_dim=1` |
| Form `y` | `(N, 1, res, res, res)`: φ̂ | Kanal explizit mitspeichern, sonst bricht die Normalisierung |
| Dtype | beide `torch.float32` | `y` wird in `PTDataset` nicht nach float32 gecastet, nur `x` |
| Normalisierung | `encode_input=True`, `encode_output=True`, `encoding="channel-wise"` | `UnitGaussianNormalizer` pro Kanal, gefittet nur auf Trainingsdaten |
| Auflösungen | Training 32³, Test 32³ und 64³ | Zero-Shot-Test der Auflösungsinvarianz, analog Darcy 16 → 32 |
| `meta` | Seed, Strahlparameter je Probe, `grid_extent`, Cheetah-Version, Datum | Reproduzierbarkeit, spätere Auswertung nach Parametern |

Physikalisch bleibt die Entdimensionalisierung aus der ML-Pipeline gültig: `rho_hat = rho · L³/Q` und `phi_hat = phi · 4πε₀L/Q` mit L aus der **Boxgröße**, nicht der Zellgröße. Mit Zellbezug lag der Zero-Shot-Fehler 16³ → 32³ bei 142 % statt 3 %.

Die Testdaten für 64³ werden mit einem eigenen Poisson-Solve auf 64³ erzeugt, nicht durch Hochrechnen der 32³-Daten. Sonst misst der Test nur Interpolation, keine Physik.

Speicherbedarf zur Einordnung: 1000 Proben × 5 Kanäle × 32³ × 4 Byte ≈ 0,66 GB; 50 Proben auf 64³ ≈ 0,26 GB.

## Arbeitspakete

Sechs Pakete, strikt nacheinander. Jedes ist ein eigener LLM-Auftrag mit eigenem Abnahmetest; erst wenn der grün ist, geht's ins nächste.

| # | Paket | Ergebnis | Abnahme |
| --- | --- | --- | --- |
| G1 | Parameter-Sampler | `sample_beam_params(n, seed, ranges)` → Liste von Dicts (σx, σy, στ, Energie, Ladung, Teilchenzahl) | gleicher Seed → identische Liste; alle Werte in den Grenzen aus `config.yaml` |
| G2 | Feld-Paar aus Cheetah | `make_sample(params, res)` → `(x, y)` mit Formen `(4,res,res,res)` und `(1,res,res,res)` | φ aus der Funktion stimmt mit `SpaceChargeKick._solve_poisson_equation` auf < 1e-6 relativ überein |
| G3 | Datei-Writer | `generate_cheetah_sc(root, n_train, n_tests, train_res, test_res, seed)` schreibt die `.pt`-Dateien | Dateien existieren, Keys `x`, `y`, `meta`, Dtype float32, keine NaN/Inf |
| G4 | `CheetahDataset` | Unterklasse von `PTDataset` mit `generate`-Flag | `train_db[0]` liefert Dict mit `x (4,32,32,32)` und `y (1,32,32,32)`; `test_dbs` hat Keys 32 und 64 |
| G5 | `load_cheetah_sc` | gleiche Rückgabe wie `load_darcy_flow_small` | ein Batch aus `train_loader` hat Form `(16,4,32,32,32)`; Normalisierer sind gefittet |
| G6 | Smoke-Test Training | Skript mit 3D-FNO und neuralop `Trainer`, 2 Epochen | läuft ohne Fehler, Train-Loss sinkt, Eval meldet Werte für 32 und 64 |

G2 ist der heikle Teil: Hier muss der Generator dieselbe Rechenkette wie `SpaceChargeKick` treffen (Blöcke 1 bis 7 der Schnittstellendefinition). Die anderen Pakete sind reine Verdrahtung nach Vorlage.

## Prompts für die andere LLM

Ein Kontextpaket am Anfang der Session, danach ein kurzer Prompt pro Paket. So bleibt jede Antwort klein genug zum Prüfen.

**Kontextpaket (einmal hochladen):** `Datengenerierung_Architektur.svg`, `pt_dataset.py`, `darcy.py`, `data_processors.py`, `space_charge_kick.py`, die Abschnitte „Schnitt A“ und „Der Ablauf in elf Blöcken“ aus der Schnittstellendefinition, dieses Dokument. `dataloader.py` (1706 Zeilen PyTorch-Interna) weglassen, es bringt nur Rauschen.

**Start-Prompt:**

```text
Du bist mein Pair-Programmer für einen Datengenerator in Python/PyTorch.
Ziel: load_cheetah_sc(...) mit exakt derselben Rückgabe wie
load_darcy_flow_small(...) aus neuralop (darcy.py anbei), damit
neuralop.training.Trainer unverändert läuft.
Die Architektur steht in der SVG und im Plan-Dokument. Datenformat und
Arbeitspakete G1–G6 stehen im Plan. PTDataset, DefaultDataProcessor und
DataLoader werden NICHT verändert, nur geerbt bzw. aufgerufen.
Regeln:
- Ein Paket pro Antwort. Nur Code für dieses Paket plus ein pytest-Test,
  der das Abnahmekriterium aus dem Plan prüft.
- Keine Annahmen über Cheetah-APIs erfinden: wenn etwas in
  space_charge_kick.py nicht steht, frag nach.
- Formen und Einheiten jedes Tensors als Kommentar angeben.
Bestätige kurz, dass du Plan und Dateien gelesen hast, und nenne die
drei Stellen in pt_dataset.py, die für unser Format kritisch sind.
```

Die letzte Zeile ist eine Kontrollfrage: Erwartet sind `channels_squeezed` (unsqueeze bei Mehrkanal-x), der fehlende float32-Cast für `y` und das Namensschema der Dateien. Fehlt was davon, Kontext nachreichen.

**Prompt G1:**

```text
Paket G1: Schreibe data/sampling.py mit
sample_beam_params(n: int, seed: int, ranges: dict) -> list[dict].
Größen: sigma_x, sigma_y, sigma_tau [m], energy [eV], total_charge [C],
n_particles [int]. Breite Größen log-uniform, Rest uniform.
ranges kommt aus config.yaml (Beispielwerte ergänzen, als TODO markieren).
Test: gleicher Seed -> identische Ausgabe; alle Werte in den Grenzen.
```

**Prompt G2:**

```text
Paket G2: Schreibe data/fields.py mit
make_sample(params: dict, res: int) -> tuple[Tensor, Tensor].
Ablauf: ParticleBeam aus params erzeugen (Cheetah), dann die Blöcke 1–7
aus SpaceChargeKick nachbilden: Gitterausdehnung, Deposition,
_array_rho, _solve_poisson_equation. Rufe die Methoden von
SpaceChargeKick auf, statt sie zu kopieren.
x = [rho_hat (ungepolstert, res^3), log(E_x/L), log(E_y/L), log(E_z/L)] als (4,res,res,res),
E_i = gamma-gestreckte Boxkanten (NICHT das elektrische Feld)
y = [phi_hat] als (1,res,res,res), Entdimensionalisierung laut Plan,
L aus der Boxgröße. float32.
Test: phi aus make_sample (vor Entdimensionalisierung) == phi aus
SpaceChargeKick._solve_poisson_equation, relativer Fehler < 1e-6.
```

**Prompt G3:**

```text
Paket G3: Schreibe data/generate.py mit
generate_cheetah_sc(root, n_train, n_tests, train_resolution,
test_resolutions, seed, ranges).
Schreibt {root}/cheetah_sc_train_{res}.pt und cheetah_sc_test_{res}.pt
als torch.save({"x": x, "y": y, "meta": meta}).
Train- und Test-Seeds getrennt ableiten, damit sich keine Probe doppelt.
Fortschrittsanzeige mit tqdm, Abbruch bei NaN/Inf mit Probenindex.
CLI: python -m data.generate --config config.yaml
Test: kleiner Lauf (n_train=4, n_tests=[2,2], res 16/32) prüft Dateien,
Keys, Formen, dtype und Endlichkeit.
```

**Prompt G4 + G5:**

```text
Pakete G4+G5: Schreibe data/dataset.py nach dem Muster von
darcy.py:
- class CheetahDataset(PTDataset): wie DarcyDataset, aber generate=True
  ruft generate_cheetah_sc für fehlende Dateien statt Zenodo-Download.
  dataset_name="cheetah_sc", channels_squeezed=False, channel_dim=1.
- def load_cheetah_sc(...): Signatur wie load_darcy_flow_small plus
  train_resolution, generate, num_workers. Gibt
  (train_loader, test_loaders, dataset.data_processor) zurück.
  train_loader mit shuffle=True (fehlt in der Darcy-Vorlage).
Test: Formen eines Batches, Keys von test_loaders, Normalisierer gefittet.
```

**Prompt G6:**

```text
Paket G6: Schreibe scripts/smoke_train.py: load_cheetah_sc mit kleinen
Zahlen, neuralop FNO mit n_modes=(8,8,8), in_channels=4, out_channels=1,
Trainer wie im Darcy-Tutorial (Code anbei), 2 Epochen, eval_interval=1,
training_loss=H1Loss(d=3), eval_losses={"h1": H1Loss(d=3), "l2": LpLoss(d=3, p=2)}.
Gib Train-Loss pro Epoche und Eval-Werte pro Auflösung aus.
```

## Tests und Verifikation

Der Generator gilt als fertig, wenn diese fünf Prüfungen grün sind. Die ersten drei laufen als pytest, die letzten zwei schaust du dir einmal von Hand an.

1. **Solver-Treue:** φ aus `make_sample` gegen `SpaceChargeKick` auf 5 zufälligen Strahlen, relativer Fehler < 1e-6. Zusätzlich einmal gegen die analytische geladene Kugel (Referenzwert aus der ML-Pipeline: 0,5 % auf 64³).
2. **Format:** Formen, Dtype float32, Keys, keine NaN/Inf, Train- und Test-Seeds überschneiden sich nicht.
3. **Durchstich:** `load_cheetah_sc` → ein Batch → `data_processor.preprocess` → FNO-Vorwärtslauf ohne Fehler, Ausgabeform `(B,1,32,32,32)`.
4. **Sichtprüfung:** für 3 Proben Mittelschnitt von ρ̂ und φ̂ plotten (z = res/2). φ̂ muss glatt sein und sein Maximum im Ladungsschwerpunkt haben.
5. **Verteilung:** Histogramm von `log10(max|φ̂|)` über den Datensatz. Die Kalibrierdaten streuten über 1,4 Dekaden; deutlich mehr heißt, die Parametergrenzen sind zu weit.

Danach Smoke-Test G6. Der Vergleich gegen die alten Zahlen (Potential 1,3 %, Kräfte 2,7 % auf 96 Proben) ist erst mit dem vollen Datensatz sinnvoll, nicht nach 2 Epochen.

## Fallstricke und offene Entscheidungen

Diese Stellen im neuralop-Code beißen dich, wenn die andere LLM sie übersieht. Gib die Liste beim Review jedes Pakets mit.

| Stelle | Problem | Gegenmaßnahme |
| --- | --- | --- |
| `pt_dataset.py`, `channels_squeezed=True` (Default) | macht aus `x (N,4,D,H,W)` einen 6D-Tensor; `DarcyDataset` reicht den Parameter gar nicht durch | in `CheetahDataset` explizit `channels_squeezed=False` an `super().__init__` übergeben |
| `pt_dataset.py` Z. 134 | `y` wird nicht nach float32 gecastet | `y` schon im Generator als float32 speichern |
| `pt_dataset.py` Z. 141 | prüft `input_subsampling_rate` statt `output_subsampling_rate` | Subsampling auf `None` lassen; Auflösungen immer echt erzeugen |
| `pt_dataset.py` gesamt | lädt alle Daten komplett in den RAM | bis wenige GB unkritisch; darüber eigenes Dataset auf HDF5 oder Memmap |
| `data_processors.py`, `DefaultDataProcessor.forward` | ruft `postprocess(output)` ohne `data_dict` auf → TypeError | nie `data_processor(...)` direkt aufrufen; der `Trainer` nimmt `preprocess` und `postprocess` einzeln |
| `data_processors.py`, Normalisierung | Train-Loss auf normierten, Eval-Loss auf rücktransformierten Werten | Train- und Eval-Zahlen nicht direkt vergleichen |
| `darcy.py`, `load_darcy_flow_small` | `train_loader` ohne `shuffle=True` | in `load_cheetah_sc` ergänzen |

Offene Entscheidungen, jeweils mit Vorschlag:

- [x] **Eingangsgitter:** ρ̂ ungepolstert auf res³ (Vorschlag) oder nullgepolstert auf (2·res)³ wie in Cheetah. Ungepolstert passt zu FNO (Ein- und Ausgabe auf demselben Gitter); die offenen Ränder übernimmt `domain_padding` im FNO.
- [ ] **Zielgröße:** φ̂ (Schnitt A, Vorschlag für den Start) oder Feld mit 3 Kanälen (Schnitt B). Das Format oben trägt beides, nur `out_channels` ändert sich.
- [ ] **Parameterverteilung:** gewürfelte Familien wie in `generate.py` oder Strahlen aus echten Cheetah-Lattice-Läufen. Die zweite Variante ist aussagekräftiger, braucht aber eine Lattice als Quelle.
- [ ] **Datenmenge:** 1000 Proben für den ersten Durchlauf, Ziel 5 000 bis 50 000 laut ML-Pipeline.
