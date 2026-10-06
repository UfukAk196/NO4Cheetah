# Schnittstellendefinition für Datengenerierung: wo das Surrogat in `SpaceChargeKick` einsteigt

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