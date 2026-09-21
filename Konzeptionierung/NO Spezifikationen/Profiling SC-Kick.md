# Profiling `SpaceChargeKick`: welche Funktion wo, und was danach verglichen wird

Stand: 21.09.2026 · Paket P2 · Anschluss an `cheetah_test.ipynb` (Teil 2)
Antwortet auf: Durchsicht 2, Leitfrage 11

Alle Codestücke unten wurden gegen cheetah 0.8.4 / torch 2.14 laufen gelassen und
liefern die Zahlen, die weiter unten in der Beispieltabelle stehen (CPU).

## 1. Der Werkzeugkasten — welche Funktion wofür

| Zweck | Funktion | Wo im Notebook |
|---|---|---|
| Blockgrenzen markieren, damit die Messung überhaupt aufschlüsselbar wird | `torch.profiler.record_function` | in einer Unterklasse `ProfiledSpaceChargeKick`, Zelle P2.2 |
| Zeit **und** Speicher je Operation, vorwärts und rückwärts in einem Lauf | `torch.profiler.profile(activities=[...], profile_memory=True)` | Zelle P2.3 |
| Tabelle daraus ziehen | `prof.key_averages()` → `.table(...)` oder die Felder `cpu_time_total`, `device_time_total`, `cpu_memory_usage`, `device_memory_usage` | Zelle P2.3 |
| Belastbare Wandzeit (Aufwärmen, Wiederholungen, Streuung, CUDA-Synchronisierung inklusive) | `torch.utils.benchmark.Timer(...).blocked_autorange()` | Zelle P2.4 |
| Rückwärtszeit je Block einzeln | `torch.autograd.grad(out.sum(), inputs, retain_graph=True)` innerhalb eines `Timer` | Zelle P2.4 |
| Spitzenspeicher auf der GPU, exakt | `torch.cuda.reset_peak_memory_stats()` / `torch.cuda.max_memory_allocated()` | Zelle P2.5 |
| Spitzenspeicher auf der CPU | `resource.getrusage(RUSAGE_SELF).ru_maxrss` — **nur im frischen Unterprozess** je Konfiguration, weil der Wert eine Höchstmarke über die ganze Prozesslaufzeit ist und nicht zurückgesetzt werden kann | Zelle P2.5 |
| Speicherverlauf als Bild (GPU) | `torch.cuda.memory._record_memory_history()` + `._dump_snapshot("sc.pickle")`, hochladen auf pytorch.org/memory_viz | Zelle P2.8 |
| Zeitstrahl als Bild | `prof.export_chrome_trace("sc_trace.json")`, ansehen in chrome://tracing oder ui.perfetto.dev | Zelle P2.3 |
| Gerätewechsel | `beam.to("cuda")`, `kick.to("cuda")`, davor/danach `torch.cuda.synchronize()` | Zelle P2.1 |
| Gitterwechsel 32³ / 64³ | Konstruktorargument `grid_shape=(64, 64, 64)` | Sweep-Schleife, Zelle P2.6 |

Was **nicht** taugt: `time.perf_counter` allein (zu verrauscht, und auf der GPU misst es
nur das Abschicken der Kernel, nicht deren Laufzeit), `tracemalloc` (sieht die
Torch-Tensoren gar nicht, die laufen an Pythons Speicherverwaltung vorbei) und
`%%timeit` (kein Speicher, keine Aufschlüsselung).

## 2. Wo die Blockgrenzen hingehören

Der Knackpunkt: Deposition, Green-Aufbau und FFT stecken alle drei **in derselben
Methode** `_solve_poisson_equation` (Zeilen 293–321 in `space_charge_kick.py`). Marken
an den Methodengrenzen allein reichen also nicht — diese eine Methode muss in der
Unterklasse nachgebaut werden, mit einer Marke um den FFT-Teil.

Die übrigen vier Blöcke bekommen ihre Marke per `super()`-Durchreiche. Weil die
Aufrufe verschachtelt sind (`_compute_forces` ⊃ `_E_plus_vB_field` ⊃
`_solve_poisson_equation` ⊃ {Deposition, Green, FFT}), liefert der Profiler
Bruttozeiten; die Nettozeit je Block kommt durch Abziehen der Kinder heraus:

```
Deposition     = B4-5
Green-Aufbau   = B6
FFT-Faltung    = B7
Differenzen    = B8 − (B4-5 + B6 + B7)
Interpolation  = B9 − B8
Koordinaten + Impulsupdate = track − B9
```

## 3. Die Zellen

### P2.1 — Messumgebung

```python
import gc
import platform
import resource

import pandas as pd
import torch
from torch.profiler import ProfilerActivity, profile, record_function
from torch.utils.benchmark import Timer

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
GRIDS = [(32, 32, 32), (64, 64, 64)]

print(f"{platform.processor()} | torch {torch.__version__}")
for d in DEVICES:
    if d == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}, "
              f"{torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GiB")
```

### P2.2 — Unterklasse mit Blockmarken

```python
from scipy.constants import epsilon_0


class ProfiledSpaceChargeKick(cheetah.SpaceChargeKick):
    """`SpaceChargeKick` mit einer `record_function`-Marke um jeden der elf Blöcke.

    `_solve_poisson_equation` ist nachgebaut (Original: space_charge_kick.py,
    Zeilen 293-321), weil Deposition, Green-Aufbau und FFT sonst in einer
    einzigen Methode stecken und nicht auseinanderzuhalten wären. Gerechnet
    wird Zeile für Zeile dasselbe.
    """

    def _array_rho(self, beam, xp, cell_size, grid_dimensions):
        with record_function("B4-5 Deposition"):
            return super()._array_rho(beam, xp, cell_size, grid_dimensions)

    def _integrated_green_function(self, beam, cell_size):
        with record_function("B6 Green-Aufbau"):
            return super()._integrated_green_function(beam, cell_size)

    def _solve_poisson_equation(self, beam, xp, cell_size, grid_dimensions):
        charge_density = self._array_rho(beam, xp, cell_size, grid_dimensions)
        green = self._integrated_green_function(beam, cell_size)
        with record_function("B7 FFT-Faltung"):
            rho_ft = torch.fft.rfftn(charge_density, dim=[1, 2, 3])
            green_ft = torch.fft.rfftn(green, dim=[1, 2, 3])
            potential = (1.0 / (4 * torch.pi * epsilon_0)) * torch.fft.irfftn(
                rho_ft * green_ft, dim=[1, 2, 3]
            ).real
            return potential[
                ...,
                : charge_density.shape[-3] // 2,
                : charge_density.shape[-2] // 2,
                : charge_density.shape[-1] // 2,
            ]

    def _E_plus_vB_field(self, beam, xp, cell_size, grid_dimensions):
        with record_function("B8 Differenzen"):
            return super()._E_plus_vB_field(beam, xp, cell_size, grid_dimensions)

    def _compute_forces(self, beam, xp, cell_size, grid_dimensions):
        with record_function("B9 Interpolation"):
            return super()._compute_forces(beam, xp, cell_size, grid_dimensions)

    def track(self, incoming):
        with record_function("track gesamt"):
            return super().track(incoming)


# Gegenprobe: die Marken dürfen am Ergebnis nichts ändern
_check = ProfiledSpaceChargeKick(section_length / 3, name="chk").track(incoming)
print("bitgleich zum Original:",
      torch.equal(_check.particles, kick.track(incoming).particles))
```

### P2.3 — Gesamtprofil, vorwärts und rückwärts getrennt

Der Rückwärtslauf braucht etwas, woran ein Gradient hängt. `incoming.particles`
auf `requires_grad_(True)` zu setzen ist der ehrlichste Aufhänger: es ist genau
der Pfad, den ein gradientenbasiertes Verfahren später auch geht.

```python
BLOCKS = ["B4-5 Deposition", "B6 Green-Aufbau", "B7 FFT-Faltung",
          "B8 Differenzen", "B9 Interpolation"]


def profile_track(num_particles, grid, device):
    beam = make_beam(num_particles, device)      # siehe Zelle 10, nur mit .to(device)
    beam.particles.requires_grad_(True)
    kick = ProfiledSpaceChargeKick(section_length / 3, grid_shape=grid, name="SC")

    out = kick.track(beam)                        # Aufwärmrunde, nicht mitzählen
    out.sigma_x.backward()
    beam.particles.grad = None

    activities = [ProfilerActivity.CPU]
    if device.startswith("cuda"):
        activities.append(ProfilerActivity.CUDA)
        torch.cuda.synchronize()

    with profile(activities=activities, profile_memory=True) as prof:
        with record_function("VORWAERTS"):
            out = kick.track(beam)
            loss = out.sigma_x
            if device.startswith("cuda"):
                torch.cuda.synchronize()
        with record_function("RUECKWAERTS"):
            loss.backward()
            if device.startswith("cuda"):
                torch.cuda.synchronize()

    events = {e.key: e for e in prof.key_averages()}

    def ms(key):
        e = events.get(key)
        if e is None:
            return float("nan")
        raw = e.device_time_total if device.startswith("cuda") else e.cpu_time_total
        return raw / 1e3

    dep, green, fft, diff, interp = (ms(b) for b in BLOCKS)
    total = ms("track gesamt")
    return {
        "Deposition": dep,
        "Green-Aufbau": green,
        "FFT-Faltung": fft,
        "Differenzen": diff - dep - green - fft,
        "Interpolation": interp - diff,
        "Koordinaten + Kick": total - interp,
        "vorwaerts": ms("VORWAERTS"),
        "rueckwaerts": ms("RUECKWAERTS"),
    }, prof


row, prof = profile_track(100_000, (32, 32, 32), "cpu")
print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=20))
prof.export_chrome_trace("sc_trace_32_cpu.json")   # für die Abbildung
```

**Wichtige Einschränkung:** `record_function`-Marken greifen nur beim Vorwärtslauf.
Der Rückwärtslauf taucht im Profil als `autograd::engine::evaluate_function: …Backward0`
auf und lässt sich darüber nur grob den Blöcken zuordnen. Für eine saubere
Rückwärts-Aufschlüsselung je Block gibt es Zelle P2.4.

### P2.4 — Block für Block, vorwärts und rückwärts sauber getrennt

Baut direkt auf `probe_space_charge` aus Zelle 13 auf: die Zwischenprodukte liegen
schon vor, jeder Block wird einzeln angeworfen.

```python
def bench_block(label, fn, inputs, min_run_time=1.0):
    """Wandzeit vorwärts und rückwärts für einen einzelnen Block."""
    for t in inputs:
        t.requires_grad_(True)

    t_fwd = Timer(stmt="fn()", globals={"fn": fn}) \
        .blocked_autorange(min_run_time=min_run_time).median * 1e3

    out = fn()
    scalar = out.sum() if torch.is_tensor(out) else sum(o.sum() for o in out)
    t_bwd = Timer(
        stmt="torch.autograd.grad(scalar, inputs, retain_graph=True)",
        globals={"torch": torch, "scalar": scalar, "inputs": inputs},
    ).blocked_autorange(min_run_time=min_run_time).median * 1e3

    print(f"{label:22s} vorwaerts {t_fwd:8.2f} ms   rueckwaerts {t_bwd:8.2f} ms"
          f"   Verhaeltnis {t_bwd / t_fwd:5.2f}")
    return t_fwd, t_bwd


flat, xp = probe["flat"], probe["xp_coordinates"].detach()
cell_size, grid_dimensions = probe["cell_size"], probe["grid_dimensions"]
rho, green = probe["rho_padded"].detach(), probe["green"].detach()

bench_block("Deposition",
            lambda: kick._array_rho(flat, xp, cell_size, grid_dimensions), [xp])
bench_block("Green-Aufbau",
            lambda: kick._integrated_green_function(flat, cell_size), [cell_size])
bench_block("FFT-Faltung", lambda: torch.fft.irfftn(
    torch.fft.rfftn(rho, dim=[1, 2, 3]) * torch.fft.rfftn(green, dim=[1, 2, 3]),
    dim=[1, 2, 3]).real, [rho, green])
bench_block("Poisson gesamt",
            lambda: kick._solve_poisson_equation(flat, xp, cell_size, grid_dimensions), [xp])
bench_block("+ Differenzen",
            lambda: kick._E_plus_vB_field(flat, xp, cell_size, grid_dimensions), [xp])
bench_block("+ Interpolation",
            lambda: kick._compute_forces(flat, xp, cell_size, grid_dimensions), [xp])
```

### P2.5 — Spitzenspeicher

```python
def peak_memory_mib(fn, device):
    """Spitzenspeicher eines Aufrufs. GPU exakt, CPU über die Prozess-Höchstmarke."""
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        fn()
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated() / 2**20
    before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    fn()
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 - before
```

**Fallstrick, im Messlauf schon aufgetreten:** `ru_maxrss` ist eine Höchstmarke über
die gesamte Prozesslaufzeit und lässt sich nicht zurücksetzen. Läuft vorher schon eine
größere Konfiguration, kommt für die kleinere glatt 0 MiB heraus. Für die
CPU-Speicherspalte der Profiltabelle also jede Konfiguration in einem eigenen
Unterprozess messen:

```python
import json
import subprocess
import sys

SNIPPET = """
import json, resource, sys, torch
sys.path.insert(0, ".")
from sc_profiling import make_beam, build_kick     # ausgelagerte Hilfsfunktionen
n, g = int(sys.argv[1]), int(sys.argv[2])
beam = make_beam(n, "cpu"); beam.particles.requires_grad_(True)
kick = build_kick((g, g, g))
out = kick.track(beam); out.sigma_x.backward()
print(json.dumps({"peak_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024}))
"""


def peak_memory_subprocess(num_particles, grid_edge):
    result = subprocess.run(
        [sys.executable, "-c", SNIPPET, str(num_particles), str(grid_edge)],
        capture_output=True, text=True, check=True)
    return json.loads(result.stdout.strip().splitlines()[-1])["peak_mib"]
```

### P2.6 — Der Sweep

```python
rows = []
for device in DEVICES:
    for grid in GRIDS:
        for n in [10_000, 100_000, 1_000_000]:
            row, _ = profile_track(n, grid, device)
            row.update(Geraet=device, Gitter=f"{grid[0]}³", N=n)
            row["Spitzenspeicher / MiB"] = (
                peak_memory_mib(lambda: profile_track(n, grid, device), device)
                if device.startswith("cuda")
                else peak_memory_subprocess(n, grid[0]))
            rows.append(row)

profiltabelle = pd.DataFrame(rows).set_index(["Geraet", "Gitter", "N"])
profiltabelle.round(2).to_csv("profiltabelle_p2.csv")
profiltabelle.round(2)
```

### P2.7 — Die Abbildung

Zwei Tafeln, mehr braucht es nicht: links gestapelte Balken (ein Balken je
Konfiguration, ein Farbabschnitt je Block, vorwärts und rückwärts nebeneinander),
rechts doppelt logarithmisch die Blockzeit über der Gitterkantenlänge, mit
Steigungsgeraden für O(G³) und O(G³ log G) als Orientierung. Speicher als zweite
Achse oder als eigene kleine Tafel darunter.

### P2.8 — Speicherverlauf als Bild (nur GPU)

```python
torch.cuda.memory._record_memory_history(max_entries=100_000)
out = kick.track(beam)
out.sigma_x.backward()
torch.cuda.memory._dump_snapshot("sc_memory_32_cuda.pickle")
torch.cuda.memory._record_memory_history(enabled=None)
# Datei auf https://pytorch.org/memory_viz hochladen -> Zeitstrahl der Belegung
```

## 4. Erste Zahlen (CPU, 4 Kerne, float32, ein Kick)

Zeiten in Millisekunden, Nettoanteile nach der Abzugsregel aus Abschnitt 2.

| Gitter | N | Deposition | Green | FFT | Differenzen | Interpolation | Koord. + Kick | **vorwärts** | **rückwärts** |
|---|---|---|---|---|---|---|---|---|---|
| 32³ | 10 000 | 2,98 | 28,66 | 3,20 | 0,63 | 5,61 | 4,21 | **46,0** | **41,5** |
| 32³ | 100 000 | 10,00 | 20,68 | 1,70 | 0,66 | 47,70 | 13,73 | **95,8** | **120,3** |
| 64³ | 10 000 | 3,31 | 136,76 | 22,71 | 1,42 | 5,22 | 4,31 | **174,4** | **134,4** |
| 64³ | 100 000 | 16,06 | 116,02 | 22,72 | 1,48 | 68,93 | 14,46 | **240,8** | **216,2** |

Drei Dinge fallen sofort auf, und alle drei widersprechen der naheliegenden Vermutung:

1. **Die FFT ist nicht der Kostentreiber.** 1,7 bis 22,7 ms, also zwischen 2 % und 9 %
   des Kicks. Der Aufbau der Green-Funktion kostet das Fünf- bis Zwölffache davon.
2. **Der Green-Aufbau hängt gar nicht an ρ.** Er braucht nur `cell_size` und γ. Bei
   festem Gitter und festem γ über einen Batch ließe er sich einmal rechnen und
   wiederverwenden — eine Zwischenspeicherung, die den Kick bei 64³ fast halbiert.
   Das gehört als Vergleichsvariante in die Baseline-Tabelle von P6, sonst gewinnt das
   Surrogat gegen einen Gegner, der sich selbst im Weg steht.
3. **Rückwärts kostet etwa so viel wie vorwärts**, bei 32³/100k sogar mehr. Für ein
   gradientenbasiertes Verfahren verdoppelt sich der Preis je Kick also ungefähr.

Der Sprung 32³ → 64³ bei gleichem N: vorwärts Faktor 2,5, Green-Aufbau Faktor 5,6,
FFT Faktor 13. Die Gitterblöcke wachsen mit dem achtfachen Zellenanteil, die
teilchenbezogenen Blöcke bleiben fast stehen — genau das Bild, das die Aufschlüsselung
liefern soll.

## 5. Durchsicht 2 — warum der Speicher wächst

Drei Quellen, der Größe nach:

**Erstens das doppelte Gitter (Hockney).** ρ wird auf 64³ genullpolstert, G wird auf
64³ aufgebaut, beide werden per `rfftn` in den Frequenzraum gebracht (64·64·33
komplexe Einträge, also die doppelte Bytezahl je Eintrag), das Produkt ist ein
dritter Tensor dieser Größe. Aus einem physikalischen 32³-Feld werden so
größenordnungsmäßig ein Dutzend Tensoren zu je 64³. Acht Neuntel der Gitterpunkte
tragen keine Physik, sondern halten nur die zyklische Faltung von den echten Rändern
fern. Beim Schritt auf 64³ physikalisch verachtfacht sich dieser Anteil.

**Zweitens die Teilchenseite.** Die Wolke-in-Zelle-Deposition legt für jedes
Makroteilchen acht Eckindizes und acht Gewichte an: bei 100 000 Teilchen sind das
Tensoren mit 800 000 Einträgen, die Indizes davon als int64 mit acht Byte je Eintrag.
Die Rückinterpolation macht dasselbe noch einmal, für drei Feldkomponenten. Das ist
der Posten, der mit N wächst — und im Messlauf oben auch der Posten mit der längsten
Rechenzeit.

**Drittens, und das ist der eigentliche Punkt für diese Arbeit, der Autograd-Band.**
Solange ein Gradient gebraucht wird, hält PyTorch jedes Zwischenergebnis fest, bis der
Rückwärtslauf es abgeholt hat: ρ, ρ̂, G, Ĝ, das Produkt, φ, drei Gradientenfelder, die
Gewichtstensoren der Interpolation. Der Speicher richtet sich also nicht nach dem
größten einzelnen Tensor, sondern nach der Summe aller Zwischenergebnisse entlang des
Pfades. Bei einem Lattice mit mehreren Kicks summiert sich das über alle Kicks, weil
der Band erst am Ende abgewickelt wird. Das ist der Grund, warum ein
Langstreckentracking mit Gradienten an der Speicherwand endet und nicht an der
Rechenzeit.

**Und da liegt der Gewinn des Neural Operators.** Er ersetzt die Blöcke 6 und 7 durch
eine feste Zahl von Schichten:

- Der Band wird kurz und vorhersagbar. Statt ρ̂, Ĝ und dem Produkt auf 64³ stehen dort
  die Zwischenschichten des Netzes auf 32³ — und deren Zahl legt man beim Entwurf fest,
  sie hängt nicht am Gitter.
- Die Nullpolsterung könnte ganz entfallen, wenn der lokale W-Zweig die offenen Ränder
  trägt. Das ist die Hypothese aus dem Motivationsleitfaden, und sie spart ein
  Achtel-zu-Ganzes an Gitterpunkten.
- Der Rückwärtslauf des Operators ist gewöhnliche Netz-Rückrechnung mit festen
  Gewichten, statt einer zweiten FFT-Runde über das große Gitter.
- Die Gewichte kosten einmalig `k_max³ · d_v²` komplexe Einträge je Schicht und
  wachsen nicht mit der Teilchenzahl.

Was die Messung zugleich zeigt: Block 9, die Rückinterpolation, bleibt beim Schnitt A
unangetastet und ist bei 100 000 Teilchen der dickste Posten. Der ehrliche Satz für das
Manuskript lautet deshalb nicht „Space Charge ist der Flaschenhals", sondern: der
Gitterteil des Kicks wächst mit dem Gitter und dominiert bei feiner Auflösung, der
Teilchenteil wächst mit N und dominiert bei großen Wolken — Schnitt A greift den
ersten an, und die Profiltabelle sagt, ab welcher Auflösung sich das auszahlt. Das ist
genau der Grund, warum die Auflösungsstudie aus P7b kein Beiwerk ist, sondern das
Hauptargument.

## 6. Leitfrage 11 — Vergleichspartner und Metriken

### Wogegen verglichen wird

| Vergleichspartner | Wofür er steht | Woher |
|---|---|---|
| `cheetah.SpaceChargeKick`, 32³ | der Gegner, den das Surrogat ablösen soll — gleiche Auflösung, gleiche Daten | P2 |
| dasselbe mit zwischengespeicherter Green-Funktion | fairer Gegner, ohne den vermeidbaren Aufbau je Aufruf | P6 |
| `SpaceChargeKick` bei 64³ und 128³ | die konvergierte Referenz; dagegen wird Genauigkeit gemessen, nicht gegen 32³ | P6 |
| kalter uniformer Bunch, analytische Verdopplung | die einzige echte Wahrheit im ganzen Aufbau — schon in Zelle 11 verdrahtet | P1 |
| Drift ohne Raumladung | die untere Schranke: was passiert, wenn man den Effekt weglässt | P1 |
| CNN oder U-Net bei fester Auflösung | zeigt, was der Operator gegenüber einem gewöhnlichen Netz beiträgt: Auflösungsunabhängigkeit | P6 |
| Kaiser-MLP auf `ParameterBeam` | andere Strahldarstellung, deshalb nur Einordnung im Ausblick, kein Zahlenvergleich | Ausblick |

### Zu den angehängten Metriken

Die Tabelle mit Accuracy, Precision, Recall, F1 und Konfusionsmatrix stammt aus der
**Klassifikation**: sie setzt voraus, dass eine Vorhersage entweder richtig oder falsch
ist, dass es also abzählbare Klassen gibt. Hier wird ein kontinuierliches Feld auf ein
kontinuierliches Feld abgebildet — eine Regression, genauer: eine Operatorapproximation.
Keine der sechs Kennzahlen ist darauf anwendbar. Sie gehören nicht in die
Bewertungstabelle. (Das gilt für klassische Netze genauso — es ist keine Frage der
Architektur, sondern der Aufgabenart.)

Eine einzige Ausnahme, und die ist bewusst konstruiert: wenn man eine Toleranz festlegt
und je Probe fragt „drunter oder drüber", entsteht ein binäres Ereignis, und der Anteil
der Proben unter der Toleranz ist dann eine Trefferquote — praktisch dasselbe wie
Accuracy auf einem abgeleiteten Merkmal. Als eine Zeile „98,3 % der Proben unter 1 %
Feldfehler" ist das gut lesbar. Konfusionsmatrix und F1 tragen darüber hinaus nichts bei,
weil es keine zweite Klasse mit eigener Bedeutung gibt.

### Was stattdessen

**Feldebene** — misst, ob φ stimmt:

- relativer L2-Fehler je Probe, ‖φ̂ − φ‖₂ / ‖φ‖₂, am mittelwertfreien φ. Das ist die
  Standardkennzahl der drei Neural-Operator-Paper und schon in Zelle 20 verdrahtet.
- **als Verteilung berichten, nicht als Mittelwert**: Median, 95. Perzentil, Maximum.
  Ein Operator, der im Mittel gut ist und bei 2 % der Strahlen versagt, ist unbrauchbar,
  und der Mittelwert verrät es nicht.
- relativer L∞-Fehler — wo sitzt der schlimmste Punkt, und sitzt er im Zentrum oder am
  Rand.
- Fehler je Modenband, aus `torch.fft.rfftn` der Differenz. Sagt, welche Skalen das
  Modell verfehlt, und speist direkt die Wahl von `k_max` zurück.

**Strahlebene** — misst, ob es für die Physik reicht, und wiegt schwerer als die
Feldebene:

- Δσ_x, Δσ_px, Δσ_τ nach dem Segment, gegen den exakten Lauf. Steht in Zelle 20 schon.
- Emittanzzuwachs Δε_x, Δε_y — die Größe, an der ein Beschleunigerphysiker das Surrogat
  misst.
- Fehleraufbau über n Kicks: die Kurve, nicht der Endwert.

Der Befund aus Zelle 20 ist hier der Leitsatz: 1 % unkorreliertes Rauschen auf φ
erzeugt 0,01 % Abweichung in σ, die Modenabschneidung dagegen schlägt voll durch.
**Der Feldfehler allein taugt nicht als Gütemaß** — er muss immer zusammen mit einer
Strahlgröße berichtet werden.

**Gradientenebene** — die eigene Forschungsfrage, und die Ebene, für die es in der
angehängten Tabelle schlicht keine Entsprechung gibt:

- Kosinusähnlichkeit zwischen ∂L/∂θ aus dem Surrogat und aus dem exakten Löser.
- relativer Fehler der Gradientennorm.

Der Kalibrierlauf hat dort 0,06 bei 1 % Feldfehler ergeben — ein Modell kann das Feld
gut treffen und den Gradienten komplett verfehlen. Ohne diese beiden Zahlen steht das
Wort „differenzierbar" im Titel und ist unbelegt.

**Übertragung:**

- derselbe relative L2-Fehler bei 64³, ohne Nachtraining (Auflösungsstudie P7b). Das ist
  die Kennzahl, die ein CNN nicht liefern kann — das Alleinstellungsmerkmal in
  Zahlenform.
- Fehler außerhalb der Trainingsbereiche von γ, Ladung und Bunchform.

**Kosten** — die Spalten, die aus dieser Profiltabelle kommen:

- Millisekunden je Kick und Spitzenspeicher in MiB, vorwärts und rückwärts getrennt,
  CPU und GPU.
- Beschleunigungsfaktor **bei gleicher Genauigkeit**, nicht bei gleicher Auflösung. Ein
  Surrogat mit 1 % Fehler gegen einen Löser auf grobem Gitter mit 3 % Fehler zu stellen,
  wäre kein Vergleich.
- Einmalkosten: Datenerzeugung plus Training, als Zahl von Kick-Auswertungen ausgedrückt.
  Daraus folgt, ab wie vielen Simulationen sich die Sache trägt — die Zahl, nach der in
  der Konsultation als erstes gefragt wird.