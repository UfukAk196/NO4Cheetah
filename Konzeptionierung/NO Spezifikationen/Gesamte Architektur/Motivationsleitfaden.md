
# Motivation: Neural Operator für den Space-Charge-Kick in Cheetah
 
Stand: 30.08.2026 · Abgeleitet aus Li et al. (2021), FNO, und aus `space_charge_kick.py`
Vollständige Fassung mit Quellenkorrekturen: `Li_2021_Fourier Neural Operator for Parametric PDEs.docx`
 
## Die Andockstelle im Code
 
`SpaceChargeKick.track` läuft so ab:
 
1. `to_xyz_pxpypz` – Koordinatenwechsel
2. `cloud_in_cell_charge_deposition` – Makroteilchen auf uniformes Gitter (Standard 32³, ±3σ)
3. `_array_rho` – Nullpolsterung auf 64³ (Hockney, für offene Ränder)
4. `_integrated_green_function` – Aufbau der integrierten Green-Funktion auf 64³ aus 8 `_integrated_potential`-Auswertungen (je 6 atan/asinh-Terme) plus Spiegelkopien in alle Oktanten; longitudinale Zellgröße mit γ skaliert
5. `_solve_poisson_equation` – `rfftn(ρ) · rfftn(G) → irfftn` = die Faltung
6. `_E_plus_vB_field` – zentrale Differenzen auf φ, Skalierung mit −1/γ²
7. `_compute_forces` – trilineare Rückinterpolation
8. Impulsupdate über dt = effect_length/(β·c)
 
**Ersatzkandidat: Schritte 3–5 (wahlweise 3–6).** Abbildung Feld → Feld auf uniformem Gitter, ausgeführt als Multiplikation im Frequenzraum – dieselbe Rechenvorschrift, die eine Fourier-Schicht parametrisiert. Unterschied: Cheetah verwendet die analytische Green-Funktion, ein FNO gelernte Gewichte R im k-Raum.
 
## Argumentationskette (fünf Glieder)
 
1. **Differenzierbarkeit ist der Daseinszweck von Cheetah.** PyTorch-Implementierung, damit Gradienten durchs Tracking laufen: Beamline-Optimierung, inverse Probleme, Systemidentifikation, RL-Umgebung. [Beleg: PhysRevAccelBeams 27, 054601]
2. **Space Charge ist die teure Stelle.** `is_skippable = False`, lässt sich nicht in Transfermatrizen einfalten. Pro Aufruf: Deposition + kompletter Green-Funktions-Neuaufbau auf 64³ + 3 FFT-Durchläufe + Differenzen + Rückinterpolation, und das pro Kick entlang der Strecke.
3. **Rückwärts ist teurer als vorwärts.** Die ganze Kette liegt im Autograd-Graphen, mit allen 64³-Zwischenfeldern pro Kick.
4. **Warum ein Operator, kein CNN.** Ersetzt wird eine Feld-auf-Feld-Abbildung, gebraucht über verschiedene `grid_shape`/`grid_extent` und Strahlgrößen. CNN-Surrogat hinge an einer Auflösung – genau die Schwäche, die Li et al. mit dem wachsenden FCN-Fehler belegen (Tab. 3, 4).
5. **Warum Fourier.** Der ersetzte Schritt *ist* eine FFT-Faltung auf uniformem Gitter; der induktive Bias passt zur Physik. Zusätzlich: FNO braucht dank des lokalen W-Zweigs keine Periodizität (Li, Kap. 5.5) – prüfbare Hypothese, dass die Hockney-Gitterverdopplung entfallen kann (ein Achtel weniger Gitterpunkte).
 
## Zielsatz
 
> Ziel dieser Arbeit ist ein differenzierbares, gitterauflösungsunabhängiges Neural-Operator-Surrogat für den Poisson-Löser des `SpaceChargeKick`-Elements in Cheetah, das die Faltung mit der integrierten Green-Funktion durch gelernte Gewichte im Frequenzraum ersetzt, um gradientenbasierte Optimierung über Strecken mit Raumladungseinwirkung zu beschleunigen, ohne die Komponierbarkeit der Cheetah-Elemente aufzugeben.
 
## Vier Zuschnitte
 
| Variante | Abbildung | Dafür | Dagegen |
|---|---|---|---|
| A | ρ ↦ φ | Engster Ersatz, exakte Referenz vom analytischen Löser | Zentrale Differenzen bleiben im Gradienten |
| B | ρ ↦ (F_x, F_y, F_τ) | Spart Differenzenbildung, glatterer Gradient | Keine Vergleichsgröße φ; γ muss als Eingangskanal rein |
| C | Teilchen ↦ Impulsänderung | Spart auch Deposition/Interpolation | Kein Feld→Feld-Mapping, also kein NO im engeren Sinn |
| D | Driftabschnitt als Raum-Zeit-Operator | Größter Zeitgewinn, Analogie FNO-3D | Bricht die Element-Struktur |
 
Empfehlung: **A** als Kern und Vergleichsanker, **B** als Ausbaustufe, C/D in den Ausblick.
 
## Offene Punkte (die eigentlichen Forschungsfragen)
 
- **Ist der Vorwärtsgewinn da?** 3 FFTs auf 64³ sind auf GPU nicht viel; ein 4-Schicht-FNO macht selbst 8. Der Gewinn müsste aus dem Wegfall des Green-Funktions-Aufbaus und dem billigeren Backward-Pass kommen. Muss gemessen werden.
- **Genauigkeit über viele Kicks.** Emittanzwachstum ist integral; 1 % Feldfehler pro Kick ≠ 1 % Endfehler.
- **Entdimensionalisierung.** ρ und φ laufen über viele Größenordnungen mit Ladung und γ. Kandidat: ρ auf Gesamtladung normieren, φ auf analytische Skala, γ als konstanter Eingangskanal.
- **Datenabdeckung.** Der bestehende Löser ist der Datengenerator (Vorteil ggü. Li et al.). Frage ist die Abdeckung, nicht die Menge – rein gaußsche Strahlen reichen vermutlich nicht (Halo, Mikrobunching, Asymmetrien).
- **Baseline.** Nicht „ohne Space Charge", sondern der Hockney-Löser bei gleicher und feinerer Auflösung, plus mindestens ein CNN-Surrogat auf festem Gitter.
- **Speicher in 3D.** R hat k_max³·d_v² komplexe Einträge; bei k_max = 12, d_v = 32 sind das ~1,77 Mio. pro Schicht. Früh durchrechnen.
 
## Abgrenzung zur cheetah-demo
 
Das Demo-Surrogat ersetzt eine *ganze feste Beamline* (Maschinenparameter → Strahlgrößen) und ist an diese Konfiguration gebunden. Ein Neural Operator ersetzt den *physikalischen Kern eines Elements* und bleibt beliebig in Lattices komponierbar, weil die Element-Schnittstelle erhalten bleibt. Das ist der Zusatznutzen.
 
## Nächster konkreter Schritt
 
Profiling von `SpaceChargeKick.track`: Laufzeit und Speicher, vorwärts und rückwärts getrennt, aufgeschlüsselt nach Deposition / Green-Funktion / FFT / Differenzen / Interpolation, bei 32³ und 64³, CPU und GPU. Ohne diese Zahlen ist „Space Charge ist der Flaschenhals" eine Vermutung; mit ihnen ist es der erste belastbare Beitrag der Arbeit – und das Profil zeigt gleich, welche der vier Varianten sich lohnt.
 

Datei kann nicht geöffnet werden.