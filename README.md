# Space Charge Neural Operator

Skelett fuer eine Masterarbeit: ein differenzierbarer Neural Operator, der den
FFT-Poisson-Solver in Cheetahs `SpaceChargeKick` ersetzt.

Der Code laeuft. Er ist nicht fertig -- er ist ein Geruest mit einer
funktionierenden Pipeline von der Datengenerierung bis zur Auswertung, damit du
am ersten Tag Zahlen siehst statt Importfehler. Alles Interessante ist als
offene Frage markiert.

```
python scripts/verify_physics.py     # 30 Sekunden: stimmt der Solver?
python scripts/smoke_test.py         # 2 Minuten: laeuft alles durch?
```

---

## Was hier gelernt wird

```
                Ladungsdichte auf dem Gitter
                          |
                          v
        [ FFT-Faltung mit Greenscher Funktion ]   <-- klassisch, O(N log N)
        [        Neural Operator (FNO)        ]   <-- das hier
                          |
                          v
                    Potential phi
                          |
                 zentrale Differenzen
                          |
                          v
                 Kraefte -> Impuls-Kick
```

Ersetzt wird ausschliesslich der mittlere Kasten. Deposition, Gradient und
Interpolation bleiben der klassische Code -- die sind billig, exakt und schon
differenzierbar. Der teure Teil, und der einzige, bei dem ein Surrogat etwas
gewinnen kann, ist der Poisson-Solve.

**Warum das ueberhaupt eine Arbeit wert ist:** Cheetah ist gradientenfaehig,
und Space Charge ist der Schritt, der das Tracking langsam macht. Ein
Surrogat, der nicht nur richtig rechnet, sondern auch richtig *ableitet*,
macht gradientenbasierte Optimierung von Strahlparametern ueber lange
Strecken erst praktikabel. Das zweite "richtig" ist der eigentliche Kern der
Arbeit -- und, so viel vorweg, der schwierige Teil (siehe unten).

---

## Struktur

```
src/scno/
  physics.py           klassischer Solver: CIC-Deposition, IGF, Hockney-FFT
                       -> erzeugt die Trainingsdaten UND ist die Referenz
  normalization.py     Entdimensionalisierung -- lies das zuerst
  losses.py            relativer L2, H1 (Gradient!), Poisson-Residuum
  models/fno.py        FNO-3D, eigene Implementierung + Anbindung an neuraloperator
  models/exact.py      der exakte Solver als "Modell" -- die Kontrolle
  data/generate.py     Strahlverteilungen wuerfeln, loesen, nach HDF5 schreiben
  data/dataset.py      Laden, Splits, Vergroebern fuer Aufloesungsstudien
  train.py             Trainingsschleife, eine Datei, keine Framework-Magie
  evaluate.py          Genauigkeit, Aufloesung, Gradiententreue
  cheetah_element.py   NeuralSpaceChargeKick -- der Einbau ins echte Lattice

scripts/
  verify_physics.py    Solver gegen die analytisch bekannte geladene Kugel
  smoke_test.py        ganze Pipeline in Miniaturgroesse
  inspect_scaling.py   Diagnose: streut phi_hat noch ueber Groessenordnungen?

configs/default.yaml   alle Regler an einer Stelle
```

---

## Loslegen

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
export PYTHONPATH=src                               # Windows: set PYTHONPATH=src

# 1. Solver pruefen -- immer zuerst
python scripts/verify_physics.py

# 2. Daten erzeugen (klein anfangen)
python -m scno.data.generate --out data/train.h5 --num-samples 512 \
    --grid 32 32 32 --num-particles 200000

# 3. Trainieren
python -m scno.train --config configs/default.yaml

# 4. Auswerten
python -m scno.evaluate --checkpoint outputs/run_default/best.pt --data data/train.h5
```

Ueberschreiben ohne die Config anzufassen:

```bash
python -m scno.train --config configs/default.yaml \
    --override training.epochs=300 model.width=64 loss.gradient_weight=2.0
```

---

## Die zwei Entscheidungen, an denen alles haengt

### 1. Entdimensionalisierung

Rohe Ladungen streuen ueber vier Dekaden, Zellgroessen ueber zwei, gamma ueber
drei. Ein Netz, das das direkt frisst, lernt Groessenordnungen statt Physik.
Also wird analytisch herausgezogen, was analytisch bekannt ist:

```
rho_hat = rho * L^3 / Q          L = (Ex*Ey*Ez)^(1/3), Kanten der Gitterbox
phi_hat = phi * 4*pi*eps0*L / Q  Q = Gesamtladung im Gitter

     laplace(phi_hat) = -4*pi*rho_hat
```

Uebrig bleiben die Achsenverhaeltnisse `E_i/L` als drei konstante Kanaele. Ein
einziges Modell deckt damit alle Strahlgroessen, Ladungen und Energien ab.

**Der Stolperstein, der mich beim Bauen eine Runde gekostet hat:** L muss an
der *Boxgroesse* haengen, nicht an der Zellgroesse. Mit Zellgroesse als Bezug
bildet dieselbe physikalische Dichte je nach Aufloesung auf ein anderes
`phi_hat` ab -- und die Zero-Shot-Super-Resolution bricht zusammen, obwohl am
Netz nichts falsch ist. Gemessen: mit Zellbezug 142 % Fehler auf 32^3 nach
Training auf 16^3, mit Boxbezug 3 %.

`scripts/inspect_scaling.py` misst, wie weit `phi_hat` danach noch streut. Bei
den Beispieldaten bleiben 1,4 Dekaden uebrig -- getrieben von der
gamma-Streckung. Das ist der erste Punkt auf der Liste unten.

### 2. Der Verlust muss den Gradienten kennen

Der Kick ist der Gradient des Potentials. Ein Modell mit hervorragendem L2 auf
`phi` kann trotzdem schlechte Kraefte liefern, weil Ableiten hohe Frequenzen
verstaerkt. Deshalb ist H1 (Feld plus Gradient) die Voreinstellung, und die
Auswertung berichtet beide Fehler getrennt.

---

## Was beim Bauen herauskam

Mini-Lauf zum Kalibrieren der Erwartung, 96 Proben, Gitter 16^3, 60 Epochen,
4,7 Mio. Parameter, CPU, gut sieben Minuten insgesamt:

| Groesse | Wert |
|---|---|
| Solver gegen analytische Kugel (64^3) | 0,5 % |
| Potential, relativer L2 | 1,3 % |
| Kraefte, relativer L2 | 2,7 % |
| Potential, Zero-Shot auf 32^3 | 3,0 % |
| Kraefte, Zero-Shot auf 32^3 | 23 % |
| Gradiententreue (Kosinus zum exakten Solver) | **0,06** |

Die ersten Zeilen sind fuer 96 Trainingsproben ordentlich und werden mit mehr
Daten deutlich besser. Die letzte Zeile ist das eigentliche Ergebnis: ein
Modell, das das Potential auf ein Prozent trifft, hat trotzdem eine praktisch
unbrauchbare Jacobi-Matrix. Genau darum geht es in dieser Arbeit -- und genau
das misst kein Standard-Benchmark, weil dort niemand nach Ableitungen fragt.

Zur Absicherung laeuft `models/exact.py` durch dieselbe Auswertung: Fehler
8e-8, Gradienten-Kosinus 1,000000. Der Messaufbau stimmt also, die 0,06 sind
echt.

---

## Offene Punkte, ungefaehr nach Ertrag sortiert

1. **Gradiententreue anheben.** Das Hauptthema. Ansaetze: Sobolev-Training mit
   hoeherem Gewicht auf dem Gradiententerm, direktes Lernen des Feldes statt
   des Potentials (drei Ausgabekanaele), Jacobian-Matching auf zufaelligen
   Richtungen, oder eine Architektur mit eingebauter Green-Struktur.
2. **Restliche Skalenstreuung.** Ein zweiter Kopf, der `log|phi_hat|` getrennt
   vorhersagt, waehrend der Hauptzweig nur die Form liefert. Sollte die 1,4
   Dekaden erledigen.
3. **Datenmenge und -vielfalt.** 96 Proben sind ein Spielzeug. Rechne mit
   5 000 bis 50 000. Die Verteilungsfamilien in `generate.py` sind ein
   Anfang; realistischer waeren Verteilungen aus echten Cheetah-Laeufen nach
   einigen hundert Metern Strecke.
4. **Ablation gegen ein CNN mit gleicher Parameterzahl.** Ohne diesen Vergleich
   ist "wir haben einen Neural Operator verwendet" eine Behauptung, keine
   Aussage. Das U-Net gewinnt bei fester Aufloesung oefter, als man denkt --
   der Unterschied zeigt sich beim Aufloesungswechsel.
5. **Laufzeit ehrlich messen.** Auf GPU ist der FFT-Solver sehr flott. Der
   Gewinn liegt vermutlich weniger im Vorwaertslauf als im Rueckwaertslauf
   und bei sehr grossen Gittern. Miss beides, bevor du Beschleunigung
   behauptest.
6. **`cheetah_element.py` fertig machen.** Die Vektor-/Broadcast-Logik des
   Originals fehlt, und die Achsenreihenfolge in `grid_sample` gehoert einmal
   gegen die Schleifenvariante gegengetestet -- dort baut man sich die
   haesslichsten stillen Fehler ein.
7. **Physik-Residuum ausprobieren.** `loss.physics_weight > 0` schaltet es an.
   Erwartung: hilft bei wenig Daten, bringt bei viel Daten nichts. Beides ist
   ein berichtenswertes Ergebnis.

---

## Bibliothek statt Eigenbau

`models/fno.py` bringt eine eigene, rund 80 Zeilen kurze FNO-Implementierung
mit -- zum Lesen, Debuggen und um das Projekt lauffaehig zu halten. Fuer die
Ergebnisse in der Thesis nimm besser die offizielle:

```bash
pip install neuraloperator
# in configs/default.yaml:  model.name: neuralop_fno
```

Das ist der Code der FNO-Autoren, inzwischen Teil des PyTorch-Ecosystems, mit
Tensor-Faktorisierung und einigem Feinschliff. Ein Ergebnis, das auf einer
gepflegten Referenzimplementierung steht, ist in der Verteidigung leichter zu
halten als eines auf Eigenbau.

---

## Literatur

- Li et al., *Fourier Neural Operator for Parametric PDEs* -- das Kernpaper.
  Lies es parallel zu `models/fno.py`, der Sprung "Spektralfaltung" wird erst
  am Code klar.
- Kovachki et al., *Neural Operators: Learning Maps Between Function Spaces* --
  der theoretische Rahmen, Diskretisierungsinvarianz.
- Azizzadenesheli et al., *Neural Operators for Accelerating Scientific
  Simulations and Design* -- Ueberblick, gut fuer die Einleitung.
- [neuraloperator.github.io](https://neuraloperator.github.io) -- Doku,
  Beispiel-Notebooks. Das Darcy-Flow-Beispiel ist mathematisch der Nachbar
  dieses Problems (auch elliptisch).
- Qiang et al., PRST-AB 9, 044204 -- die IGF-Methode, die `physics.py` umsetzt.
- Cheetah: [github.com/desy-ml/cheetah](https://github.com/desy-ml/cheetah)

---

## Nachtrag zum Arbeiten damit

Bau die Pipeline immer erst mit laecherlich kleinen Daten und einem winzigen
Modell komplett durch, bis hinten eine Zahl herauskommt -- genau dafuer ist
`smoke_test.py` da. Erst dann skalieren. Und wenn eine Auswertung seltsam
aussieht: erst `models/exact.py` durch dieselbe Auswertung schicken. Kommt
dort nicht Fehler null heraus, liegt es nicht am Netz.
