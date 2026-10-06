
# ML-Pipeline für den Space-Charge-Surrogaten: Skelett und erste Messwerte
 
Stand: 30.08.2026. Gehört zum Skelett-Repo `sc-neuralop` (als ZIP in der
Konversation geliefert).
 
## Was gebaut wurde
 
Lauffähige Pipeline von der Datengenerierung bis zur Auswertung, zugeschnitten
auf Cheetahs `SpaceChargeKick`. Ersetzt wird ausschließlich der Poisson-Solve
(die FFT-Faltung mit der integrierten Greenschen Funktion). Deposition,
Gradientenbildung und Interpolation bleiben klassischer Code — die sind billig,
exakt und bereits differenzierbar.
 
Module: `physics.py` (klassischer Solver, standalone in Torch, ohne
Cheetah-Abhängigkeit), `normalization.py`, `losses.py`, `models/fno.py`,
`models/exact.py`, `data/generate.py`, `data/dataset.py`, `train.py`,
`evaluate.py`, `cheetah_element.py`.
 
## Zwei Design-Entscheidungen
 
### Entdimensionalisierung
 
```
rho_hat = rho * L^3 / Q          L = (Ex*Ey*Ez)^(1/3), Kanten der Gitterbox
phi_hat = phi * 4*pi*eps0*L / Q  Q = Gesamtladung im Gitter
  =>  laplace(phi_hat) = -4*pi*rho_hat
```
 
Übrig bleiben die Achsenverhältnisse `E_i/L` als konstante Eingabekanäle. Ein
Modell deckt damit alle Strahlgrößen, Ladungen und Energien ab.
 
**Wichtig:** L muss an der Boxgröße hängen, nicht an der Zellgröße. Mit
Zellbezug bildet dieselbe physikalische Dichte je nach Auflösung auf ein anderes
`phi_hat` ab, und die Auflösungsinvarianz bricht zusammen. Gemessen: Training
auf 16³, Test auf 32³ ergab 142 % Fehler mit Zellbezug gegenüber 3 % mit
Boxbezug.
 
Restliche Streuung von `phi_hat` bei den Beispieldaten: 1,4 Dekaden, getrieben
von der gamma-Streckung der longitudinalen Achse. Offener Punkt.
 
### H1-Verlust
 
Der Kick ist ein Gradient des Potentials. Ein Modell mit gutem L2 auf `phi` kann
schlechte Kräfte liefern, weil Ableiten hohe Frequenzen verstärkt. Voreinstellung
ist deshalb Feld- plus Gradientenfehler; die Auswertung berichtet beide getrennt.
 
## Erste Messwerte (Kalibrierlauf, kein Ergebnis für die Thesis)
 
96 Proben, Gitter 16³, 60 Epochen, 4,7 Mio. Parameter, CPU, ~7 Minuten:
 
| Größe | Wert |
|---|---|
| Solver gegen analytische geladene Kugel (64³) | 0,5 % |
| Potential, relativer L2 | 1,3 % |
| Kräfte, relativer L2 | 2,7 % |
| Potential, Zero-Shot auf 32³ | 3,0 % |
| Kräfte, Zero-Shot auf 32³ | 23 % |
| Gradiententreue (Kosinus zum exakten Solver) | **0,06** |
 
## Das zentrale Zwischenergebnis
 
Ein Modell, das das Potential auf ein Prozent trifft, hat trotzdem eine
praktisch unbrauchbare Jacobi-Matrix (Kosinusähnlichkeit 0,06 zwischen
`d(Ziel)/d(rho)` durch das Netz und durch den exakten Solver). Da die Arbeit
gradientenbasierte Optimierung zum Ziel hat, ist genau das die Größe, die
verbessert und berichtet werden muss — und kein Standard-Benchmark für Neural
Operators misst sie, weil dort niemand nach Ableitungen fragt.
 
Der Messaufbau ist abgesichert: der exakte Solver, verpackt als Modell
(`models/exact.py`), läuft durch dieselbe Auswertung mit Fehler 8e-8 und
Gradienten-Kosinus 1,000000.
 
## Offene Punkte, nach Ertrag sortiert
 
1. Gradiententreue anheben: Sobolev-Training mit höherem Gradientengewicht,
   direktes Lernen des Feldes statt des Potentials (drei Ausgabekanäle),
   Jacobian-Matching auf zufälligen Richtungen, Architektur mit eingebauter
   Green-Struktur.
2. Restliche Skalenstreuung: zweiter Kopf für `log|phi_hat|`, Hauptzweig nur
   für die Form.
3. Datenmenge: 96 Proben sind ein Spielzeug, realistisch sind 5 000 bis 50 000.
   Verteilungen aus echten Cheetah-Läufen wären aussagekräftiger als die
   gewürfelten Familien in `generate.py`.
4. Ablation gegen ein CNN/U-Net mit gleicher Parameterzahl. Ohne diesen
   Vergleich ist die Wahl des Neural Operators eine Behauptung.
5. Laufzeit ehrlich messen, vorwärts und rückwärts getrennt. Auf GPU ist der
   FFT-Solver sehr flott; der Gewinn liegt vermutlich im Rückwärtslauf und bei
   großen Gittern.
6. `cheetah_element.py` fertigstellen: Vektor-/Broadcast-Logik des Originals
   fehlt, Achsenreihenfolge in `grid_sample` gegen die Schleifenvariante
   gegentesten.
7. Physik-Residuum (`loss.physics_weight > 0`) ausprobieren. Erwartung: hilft
   bei wenig Daten, bringt bei viel Daten nichts. Beides ist berichtenswert.
 
## Werkzeuge
 
- `neuraloperator` (offizielle FNO-Implementierung der Autoren, im
  PyTorch-Ecosystem) — für die Thesis-Ergebnisse der Eigenbau-FNO vorziehen,
  Umschalten über `model.name: neuralop_fno`.
- Kern: torch, numpy, scipy, h5py, pyyaml, matplotlib.
- Empfohlen: wandb (Läufe mitschreiben), hydra (Configs), einops.
- NVIDIA PhysicsNeMo als Zweitquelle für Trainings-Setups.
 

