# Normalisierung der Trainingsdaten: Skalare, rho und phi

Nov 17, 2026 · @Ufuk

Der geplante Ansatz (Z-Score auf die Skalare, Log plus Z-Score auf rho und phi) passt im Großen und Ganzen. Der Haken liegt bei der Deutung des rho-Histogramms: Der Spike bei etwa -16 ist kein Ausreißer, und ein anderer Scaler bügelt ihn deshalb auch nicht glatt. Dazu kommt eine Grundsatzfrage: Der Log auf rho und phi zerschießt die Linearität der Poisson-Gleichung.

## Skalare (gamma, sigma\_x, sigma\_y, sigma\_tau)

Für gleichverteilte Größen mit bekannten Sampling-Grenzen passt Min-Max auf \[-1, 1\] mit genau diesen Grenzen am natürlichsten. Der Wertebereich bleibt dann fest und hängt nicht davon ab, welche Samples zufällig im Trainingsset gelandet sind.

Z-Score klappt genauso: Auf einer Gleichverteilung landet man damit bei rund ±1,73 (±√3). Der Unterschied zwischen beiden Varianten ist marginal, hier lohnt kein Aufwand.

- Grenzen aus der Sampling-Konfiguration (oder Statistiken nur vom Trainingsset) verwenden
- Parameter abspeichern, damit die Rücktransformation reproduzierbar bleibt

## rho: Der Spike bei ca. -16 sind leere Zellen

Der Spike ist log(0 + eps): Gitterzellen ohne ein einziges Teilchen, deren Wert allein vom gewählten eps abhängt. Daher auch die Lücke zwischen -16 und -15. Max-Abs- oder Robust-Scaling bringen hier nichts, weil es um keine Extremwerte geht.

Drei Gegenmittel:

1. **eps relativ wählen**, etwa `eps = 1e-6 * rho.max()` pro Sample. Der Boden rückt dann direkt unter die echte Verteilung, statt weit abseits zu hängen.
2. **Leere Zellen aus Mittelwert und Standardabweichung rauslassen**, damit der Spike die Statistik nicht verzerrt.
3. **Binären Masken-Kanal dazupacken** (belegt vs. leer). „Leer“ wird so explizit, und das Netz muss es nicht aus einer magischen Zahl herausfriemeln.

## Grundsatzfrage: Überhaupt logarithmieren?

Die Poisson-Gleichung ist **linear in rho**: doppelte Ladung, doppeltes phi. Die Spektral-Layer eines FNO sind ebenfalls lineare Operatoren, Physik und Architektur passen also zusammen. Wer Input und Target logarithmiert, wirft das über Bord, und das Netz muss exp → Faltung → log erst mühsam lernen.

Die physikalisch begründete Alternative ist eine **Normierung pro Sample**:

```latex
\rho_n = \frac{\rho}{Q} \quad \text{oder} \quad \rho_n = \frac{\rho}{\max \rho}
```

```latex
\phi_n = \frac{\phi}{s}, \qquad s \sim \frac{Q}{4 \pi \varepsilon_0 \, \sigma}
```

Damit verschwindet der Großteil der Streuung zwischen den Samples, auch der lange Schwanz im rohen phi-Histogramm. Der stammt nämlich hauptsächlich von unterschiedlichen Ladungen und Strahlgrößen. Der Operator bleibt linear, und der Skalierungsfaktor wandert am Ende exakt wieder rein.

**Vorsicht bei den Diagnose-Plots:** Histogramme über alle Pixel gepoolt führen bei Feldern in die Irre. Dass ein einzelner Pixelwert nahe 0 liegt, sagt über das Feld nichts aus. Der FNO sieht die räumliche Struktur, und ein Gauß-Blob in \[0, 1\] mit vielen Werten nahe null ist problemlos lernbar. Das heavy-tailed Pixel-Histogramm vom rohen rho ist für einen FNO ein deutlich kleineres Problem als bei tabellarischem ML.

## Loss: Wo die Wahl wirklich zählt

Im Log-Raum gewichtet der Loss den **relativen** Fehler überall gleich, auch in den äußeren Ausläufern, wo phi winzig und physikalisch nahezu bedeutungslos ist. Auf die Teilchen wirkt aber E = −∇phi im Kern des Strahls. Im linear normierten Raum konzentriert sich der Loss auf genau die Zonen, die für den Kick relevant sind.

## Empfehlung und Ablation

| Größe | Baseline | Vergleichslauf |
| --- | --- | --- |
| gamma, sigma\_x, sigma\_y, sigma\_tau | Min-Max auf \[-1, 1\] über Sampling-Grenzen | Z-Score |
| rho | Normierung pro Sample (Q bzw. max), kein Log | log(rho + eps\_rel) + Z-Score, optional Masken-Kanal |
| phi | gleicher Faktor wie rho × charakteristische Skala, kein Log | log(phi) + Z-Score |

Mit dem bestehenden Pipeline-Skelett ist der Vergleich eine fixe Ablation. Für die Thesis gibt das außerdem einen sauberen Diskussionspunkt her: physikalisch motivierte gegen statistische Normierung.
