# Schnittstellendefinition: wo das Surrogat in `SpaceChargeKick` einsteigt

Stand: 08.09.2026 · Rohbau für Paket P3 · Grundlage: `space_charge_kick.py`,
`cheetah/utils/cloud_in_cell.py`, Kaiser et al. 2024 Abschnitt III E

Zugehörige Abbildung: Artifact „Space-Charge-Kick Schnittstellen" (Blockdiagramm
mit den drei Schnitten und der Kaiser-Figur).

## Der Ablauf in elf Blöcken

| # | Funktion | Was passiert | Ausgabe |
|---|---|---|---|
| 1 | `track(incoming)` | Prüfen, `vector_shape` broadcasten, Batch flachklopfen | `flattened_incoming.particles (B, N, 7)`, s-Koordinaten |
| 2 | — (in `track`) | `grid_dimensions = grid_extent · σ`; `cell_size = 2·gd/grid_shape`; `dt = L/(β·c)` | `(B,3)` m, `(B,3)` m, `(B,)` s |
| 3 | `to_xyz_pxpypz()` | s-Fixpunkt → t-Fixpunkt | `xp_coordinates (B, N, 7)`, Spalten 0,2,4 = x,y,τ |
| 4 | `cloud_in_cell_charge_deposition()` | Makroteilchen trilinear aufs Gitter | `charge_grid (B,32,32,32)` C |
| 5 | `_array_rho()` | ÷ Zellvolumen → Dichte, Nullpolsterung auf 2× | `ρ (B,64,64,64)` C/m³ |
| 6 | `_integrated_green_function()` | 8× `_integrated_potential()`, Oktanten-Spiegelung, dτ = dz·γ | `G (B,64,64,64)` |
| 7 | `_solve_poisson_equation()` | `rfftn(ρ)·rfftn(G) → irfftn`, ÷ 4πε₀, zurückschneiden | `φ (B,32,32,32)` V |
| 8 | `_E_plus_vB_field()` | zentrale Differenzen auf φ, × (−1/γ²) | `grad_x/y/tau`, je `(B,32,32,32)` V/m |
| 9 | `_compute_forces()` | trilineare Interpolation Gitter → Teilchen, × Elementarladung | `forces (B, N, 3)` N |
| 10 | — (in `track`) | `xp[…,1] += Fx·Δt`, `xp[…,3] += Fy·Δt`, `xp[…,5] += Fz·Δt` | `xp_coordinates (B, N, 7)` |
| 11 | `from_xyz_pxpypz()` | t → s, Batch zurückformen | `outgoing ParticleBeam (…, N, 7)` |

**Aufrufreihenfolge ≠ Datenfluss.** Der Code ruft von außen nach innen:
`track` → `_compute_forces` → `_E_plus_vB_field` → `_solve_poisson_equation` →
{`_array_rho` → `cloud_in_cell…`, `_integrated_green_function`}. Die Daten fließen
umgekehrt. Praktisch: für Schnitt A überschreibst du `_solve_poisson_equation()`,
für Schnitt B `_E_plus_vB_field()`.

## Schnitt A — ρ ↦ φ (Kern der Arbeit)

Ersetzt Blöcke 6 und 7.

- **Eingang:** `charge_density (B,64,64,64)` C/m³ (bereits nullgepolstert) ·
  `cell_size (B,3)` m (dritte Komponente ohne γ) · `gamma (B,)`
- **Ausgang:** `potential (B,32,32,32)` V — dieselbe Form und Einheit wie heute
- **Klassisch bleibt:** Blöcke 1–5 und 8–11
- **Dafür:** exakte Referenz vorhanden, φ direkt vergleichbar, engster Ersatz.
  Die Nullpolsterung könnte entfallen, wenn der lokale W-Zweig die offenen Ränder
  trägt — ein Achtel der Gitterpunkte.
- **Dagegen:** die zentralen Differenzen aus Block 8 bleiben im Gradientenpfad.
  Genau dort saß im Kalibrierlauf der Gradienten-Kosinus von 0,06 bei 1 % Feldfehler.

## Schnitt B — ρ ↦ Feld (Ausbaustufe)

Ersetzt Blöcke 6, 7 und 8.

- **Eingang:** wie A, aber `gamma` zwingend als Kanal (die Skalierung −1/γ² steckt
  sonst nirgends mehr)
- **Ausgang:** `grad_x, grad_y, grad_tau`, je `(B,32,32,32)` V/m, bereits mit
  −1/γ² skaliert — oder ein Tensor `(B,3,32,32,32)`, beim Rückgeben aufspalten
- **Dafür:** das Netz lernt direkt die Größe, die den Kick bestimmt; der
  Differenzenstern fällt aus dem Gradientenpfad
- **Dagegen:** kein vergleichbares φ mehr, drei Kanäle, γ-Konditionierung nötig

## Schnitt C — Teilchen ↦ Kraft (Ausblick)

Ersetzt Blöcke 4 bis 9.

- **Eingang:** `xp_coordinates[…, [0,2,4]] (B,N,3)` m · `particle_charges (B,N)` C ·
  `gamma (B,)`
- **Ausgang:** `forces (B,N,3)` N
- **Dagegen:** keine Abbildung zwischen Funktionen mehr, sondern über eine
  Punktwolke — kein Neural Operator im engeren Sinn. Bräuchte eine
  permutationsinvariante Architektur, und die Auflösungsunabhängigkeit fällt weg.

## Der Kaiser-Weg liegt auf einer anderen Ebene

Kaiser et al. 2024, Abschnitt III E, Abb. 8: ein NN-erweiterter Quadrupol.

```
b_out = f_linear(b_in | l_Q, k_Q)  +  ΔΣ_SC(b_in | l_Q, k_Q)
                 ^ klassisch                ^ MLP
```

- **MLP-Eingang:** `x = (σx, σx′, σy, σy′, στ, σδ, E, q)` — logarithmiert, dann auf
  Einheitsnormal skaliert
- **MLP-Ausgang:** `y = (Δσx, Δσx′, Δσy, Δσy′, Δστ, Δσδ)` — Residuum auf die
  lineare Lösung
- **Architektur:** MLP, 4 verdeckte Schichten à 256, Sigmoid, Adam, lr 2e-5,
  Batch 32, Frühabbruch mit Geduld 10, Split 60/20/20
- **Daten:** 100 000 Ocelot-Läufe, zweite Ordnung, ein Quadrupol, SC-Gitter 63³,
  Schrittweite 2 cm; log-uniforme Bereiche siehe Tab. II des Papers
- **Ergebnis:** 370 µs statt 1,36 s, gut drei Größenordnungen

**Entscheidender Unterschied:** dieses Surrogat arbeitet auf einem `ParameterBeam`
und kann keine veränderte Teilchenwolke zurückgeben, nur die zweiten Momente.
`SpaceChargeKick` arbeitet auf einem `ParticleBeam` und aktualisiert jeden
Makroteilchen-Impuls einzeln. Die Autoren schreiben selbst, dass eine Übertragung
auf Teilchenwolken eine Architektur wie PointNet bräuchte und noch aussteht.

Daraus der Abgrenzungsabsatz für die Motivation: der Kaiser-Weg ersetzt ein
*Element bei fester Strahldarstellung*, der Neural Operator ersetzt den
*physikalischen Kern* eines Elements und lässt die Element-Schnittstelle
unangetastet — deshalb bleibt er in beliebigen Lattices komponierbar und
funktioniert auf Teilchenwolken.

## Empfehlung

**A als Kern und Vergleichsanker, B als Ausbaustufe, C und Kaiser in den Ausblick.**

- A zuerst, weil für φ eine exakte Referenz vorliegt und der ersetzte Schritt
  buchstäblich dieselbe Rechenvorschrift ist wie eine Fourier-Schicht.
- B danach, weil das Problem laut Kalibrierlauf nicht die Feldgenauigkeit ist,
  sondern der Gradient. Der Vergleich A gegen B ist dann selbst ein Ergebnis.
- C und Kaiser in den Ausblick, weil beide die Auflösungsunabhängigkeit aufgeben —
  also genau das Argument, mit dem die Wahl des Neural Operators begründet wird.

Nächster Schritt in P3: die Signatur von A mit Formen und Einheiten hinschreiben.
Sie ist der Anfang des Methodikkapitels und legt zugleich fest, wie die
Trainingsdaten aus P4 aussehen müssen.