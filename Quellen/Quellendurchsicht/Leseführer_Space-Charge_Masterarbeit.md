# 🧭 Leseführer: Simulationsmodelle für Space Charge

**Wegweiser durch den Wissensspeicher · Stand 27.07.2026 · Anmeldung Masterarbeit zum 01.09.2026**

Vier Quellen liegen vor, und sie ziehen erstaunlich gut an einem Strang: eine physikalische Grundvorlesung, ein Code-Paper, eine numerische Detailstudie und der Quelltext selbst. Zusammen decken sie die Kette *Physik → Algorithmus → Implementierung → ML-Ersatzmodell* fast lückenlos ab. Fast. Wo es hakt, steht unten ein ⚠️.

---

## 🎨 Farbcodierung der Themencluster

| Marker | Cluster | Worum es geht |
|---|---|---|
| 🟦 | **Physik & Analytik** | Selbstfelder, Tune-Shift, Bildladungen, geschlossene Formeln |
| 🟩 | **Numerik & Löser** | PIC, Poisson, Integrated Green Function, FFT, Gitter |
| 🟨 | **Differenzierbarkeit & Speicher** | Autodiff, Reverse-Mode, Speicherskalierung, Checkpointing |
| 🟥 | **Machine Learning** | Surrogate, PINN, Trainingsdaten, RL, Bayesische Optimierung |
| 🟪 | **Werkzeug & Implementierung** | Cheetah-API, PyTorch, Ocelot, ASTRA, Schnittstellen |
| ⬛ | **Lücke im Material** | Musst du selbst nachschlagen |

---

## 🗺️ Die Landkarte auf einen Blick

| # | Datei | Was für ein Ding | Cluster | Umfang |
|---|---|---|---|---|
| 1️⃣ | `PhysRevAccelBeams_27_054601.pdf` | Journal-Paper (PRAB 27, 054601, 2024) – Cheetah | 🟪🟥🟨 | 17 S. |
| 2️⃣ | `p305.pdf` | CAS-Vorlesungsskript, K. Schindl (CERN) – *Space Charge* | 🟦 | 16 S. |
| 3️⃣ | `3815572_3815749.pdf` | Konferenz-Paper (PASC '26) – Speicherskalierung Reverse-Mode-AD | 🟩🟨 | 11 S. |
| 4️⃣ | `space_charge_kick.py` | Quelltext, Cheetah-Element `SpaceChargeKick` | 🟩🟪 | 608 Zeilen |

**Die Kernbeobachtung:** Drei der vier Dateien kreisen um **dasselbe Werkzeug – Cheetah**, allerdings in drei unterschiedlichen Reifegraden und aus drei verschiedenen Blickwinkeln. Schindl steht bewusst daneben und liefert das physikalische Fundament, auf dem der ganze Rest aufsitzt, ohne selbst eine Zeile Code zu enthalten.

---

# 📖 Empfohlene Bearbeitungsreihenfolge

## Station 1 — 🟪🟥 `PhysRevAccelBeams_27_054601.pdf`
### *Bridging the gap between machine learning and particle accelerator physics with high-speed, differentiable simulations*
**Kaiser, Xu, Eichler, Santamaria Garcia — DESY / KIT / TUHH, PRAB 27, 054601 (2024)**

**Was das ist:** Das Vorstellungs-Paper von Cheetah, einem auf PyTorch aufgesetzten, differenzierbaren Code für lineare Strahldynamik. Beteiligt sind DESY, KIT — und mit Annika Eichler auch die TUHH. Das ist also ziemlich sicher dein direktes Umfeld.

**Was drinsteht:**
- 🟪 Aufbau: zwei Klassen, `Beam` und `Element`, beide von PyTorch-`Module` abgeleitet. `ParticleBeam` (jedes Makroteilchen mit 7er-Vektor) gegen `ParameterBeam` (nur Mittelwertvektor µ plus Kovarianzmatrix Σ, Gauß-Annahme).
- 🟪 Tempo-Tricks: Zusammenfassen von Transfermatrizen, statisch und dynamisch, dazu Ersetzen inaktiver Elemente durch Drifts. Berichtete Beschleunigung gegenüber Ocelot/ASTRA/Bmad-X um Größenordnungen — bei bewusst in Kauf genommenem Genauigkeitsverlust.
- 🟪 Import von Strahlen aus ASTRA, Lattices aus Bmad, beidem aus Ocelot. Eigenes Format auf LatticeJSON-Basis.
- 🟥 Fünf Anwendungsbeispiele: Reinforcement Learning, gradientenbasiertes Tuning, gradientenbasierte Systemidentifikation, Cheetah als physikalischer Prior für Bayesische Optimierung — und, für dich das Herzstück, **modulares NN-Ersatzmodell für Space Charge** (Abschnitt III E).
- 🟥 Der Space-Charge-Surrogat im Detail: Residuum-Ansatz statt Rundumschlag. Also `b_out = f_linear(b_in | l_Q, k_Q) + ΔΣ_SC(b_in | l_Q, k_Q)`, wobei nur das Δ vom Netz kommt. Eingang: (σx, σx′, σy, σy′, στ, σδ, E, q), logarithmiert und normiert. Ausgang: die sechs Δσ. Netz: MLP, 4 verdeckte Schichten, Breite 256, Sigmoid, Adam, Lernrate 2·10⁻⁵, Batch 32, Early Stopping mit Geduld 10, 959 Epochen. Hyperparameter über 303 Trainingsläufe per Bayesischer Optimierung ausgesucht, gefahren mit PyTorch Lightning und Weights & Biases.
- 🟥 Trainingsdaten: **100 000 Proben aus Ocelot**, Space Charge mit Gitter 63³, Schrittweite 2 cm, Tracking zweiter Ordnung durch ein einzelnes Quadrupol. Ziehung gleichverteilt über Länge und Stärke des Quadrupols, log-gleichverteilt über die Strahlparameter (Tabelle II liefert die Spannen: σx 10 µm–1 mm, q 1 pC–5 nC, E 1 MeV–1 GeV usw.). Aufteilung 60/20/20.
- 🟥 Ergebnis: Ocelot braucht im Mittel 1,36 s pro Quadrupol-Durchlauf mit Space Charge, das NN-gestützte Cheetah-Element 370 µs. Über drei Größenordnungen weniger Rechenzeit, und das Ganze bleibt vollständig differenzierbar.
- 🟥 Der Verwandtschafts-Abschnitt (II A) ist deine Fundgrube für weiterführende Literatur: Fujita (IEEE Access 9, 164017, 2021) mit einem **physikinformierten Netz** für das Space-Charge-Feld im Vakuumkammerquerschnitt, ein NN für die transversale Emittanz niederenergetischer Strahlen, sowie ein differenzierbares selbstkonsistentes Space-Charge-Modell auf Basis abgeschnittener Potenzreihenalgebra.

**Warum du hier anfängst:** Dieses Paper ist deine Aufgabenstellung im Kleinformat. Abschnitt III E ist buchstäblich ein Machbarkeitsnachweis für genau das, was du bauen willst — die Autoren nennen es selbst einen Proof of Concept und benennen im Fazit die offenen Enden, die du beerben kannst. Es liefert dir außerdem in einem Rutsch den Grund, warum Python gesetzt ist (Punkt 2c deiner Mail) und wo dein Werkzeugkasten steht.

**Der Absatz, den du zweimal liest:** Das Ende von III E. Dort steht die Einschränkung, dass der gezeigte Ansatz auf **parametrisch beschriebene Strahlen** gemünzt ist, und dass eine Ausdehnung auf Teilchenwolken über Architekturen wie PointNet gedacht, aber noch nicht gebaut ist. Genau dort klafft eine Tür auf, durch die eine Masterarbeit passt.

> 📝 **Deine Notizen — Station 1**
> - Kandidat für Themeneingrenzung:
> - Referenzen, die ich mir holen muss:
> - Was mir an III E unklar geblieben ist:
> -

---

## Station 2 — 🟦 `p305.pdf`
### *Space Charge* — K. Schindl, CERN Accelerator School

**Was das ist:** Ein klassisches Vorlesungsskript, geschrieben für Leute, die kollektive Effekte zum ersten Mal anfassen. Vollständig analytisch, keine Numerik, kein Code, kein ML. Genau deshalb wertvoll.

**Was drinsteht:**
- 🟦 Woher der Effekt kommt: Coulomb-Abstoßung gegen magnetische Anziehung, vollständige Aufhebung erst bei Lichtgeschwindigkeit, übrig bleibt der berühmte 1/γ²-Faktor.
- 🟦 Selbstfelder für den Fall des runden, gleichmäßig geladenen Strahls (Gauß'sches Gesetz und Stokes ausgeschrieben, Gl. 1–12). Die Kraft wächst linear mit dem Radius — Space Charge wirkt wie eine über die ganze Maschine verschmierte Zerstreuungslinse, und zwar in beiden Ebenen gleichzeitig, anders als ein Quadrupol.
- 🟦 Hill'sche Gleichung mit Zusatzterm K_sc (Gl. 14–16), daraus der inkohärente Tune-Shift ΔQ ∝ N / (ε β²γ³). Wächst mit der Intensität, verschwindet praktisch bei Elektronenmaschinen.
- 🟦 Elliptischer Querschnitt (Gl. 20–25) und bi-gaußförmige Verteilung (Gl. 26–30). Merksatz von dort: eine gleichmäßige Verteilung erzeugt den kleinstmöglichen Tune-Shift, eine gaußähnliche Verteilung dagegen einen amplitudenabhängigen Tune-**Spread**, den du mit externer Fokussierung nicht mehr wegdrehen kannst.
- 🟦 Indirekte Space Charge: Bildladungen an leitenden Wänden, unendliche Spiegelreihe zwischen Parallelplatten, Laslett-Koeffizienten (Tabelle 1) für die verschiedenen Geometrien. Wichtig hierbei: Bildeffekte fallen nur mit 1/γ ab, nicht mit 1/γ², und bleiben deshalb auch bei hohen Energien am Leben.
- 🟦 Unterschied inkohärent gegen kohärent, gebündelte gegen ungebündelte Strahlen, Bunching-Faktor, Formfaktoren, praktische Formel Gl. 43.
- 🟦 Space-Charge-Grenze am Beispiel des CERN PS Booster inklusive „Krawatten"-Diagramm im Tune-Raum, plus die realen Gegenmaßnahmen (längerer Linac bei FNAL, Booster-Synchrotron bei CERN und BNL).

**Warum das für dich zählt:** Zwei Gründe. Erstens brauchst du diese Begriffe für die Einleitung deiner Ausarbeitung, und zwar in der richtigen Reihenfolge hergeleitet — das schreibt sich nicht aus dem Kopf. Zweitens gibt dir Schindl die **physikalischen Prüfsteine**, an denen du ein trainiertes Netz messen kannst: Skaliert die Vorhersage mit 1/(β²γ³)? Bleibt die Kraft im gleichmäßig gefüllten Strahl linear im Radius? Verdoppelt sich der Effekt beim Übergang von gleichmäßig auf gaußförmig? Ein Netz, das diese analytischen Grenzfälle trifft, hat mehr gelernt als Kurvenanpassung.

**Der Absatz, den du zweimal liest:** Abschnitt 2.6 mit der Linearisierung der Kraft nahe der Strahlachse. Das ist der sauberste Testfall, den du für ein physikinformiertes Netz überhaupt bekommen kannst.

> 📝 **Deine Notizen — Station 2**
> - Analytische Grenzfälle als Validierungstests für mein Netz:
> - Formeln, die in die Einleitung wandern:
> - Begriffe, die ich noch nicht sicher greife:
> -

---

## Station 3 — 🟩🟨 `3815572_3815749.pdf`
### *The Memory Scaling of Reverse-Mode Differentiation in Particle Accelerator Simulations with Space Charge*
**Dhamrait, Zoni, Huebl, Qiang, Mitchell (LBNL), Roussel (SLAC), Kaiser (DESY), Xu (Argonne), Vay, Lehe — PASC '26**

**Was das ist:** Die direkte Fortsetzung von Station 1, zwei Jahre später und von einem viel größeren Konsortium. Kernaussage: Cheetah hat jetzt einen **echten** Space-Charge-Löser, keinen Surrogat. Und die Autoren rechnen aus, was einen das im Speicher kostet, sobald man Gradienten mitziehen will.

**Was drinsteht:**
- 🟩 Der Aufbau eines Tracking-Codes mit Space Charge im Allgemeinen: Strang-Splitting zweiter Ordnung, Element in N_slices zerlegt, halber Drift → Impuls-Kick → halber Drift. Der Kick selbst über Particle-In-Cell.
- 🟩 Die sechs Schritte eines Space-Charge-Kicks, sauber aufgedröselt (Abschnitt 3.2): s-nach-t-Transformation, Ladungsdeposition per Cloud-In-Cell mit Splines erster Ordnung, Feldlöser, Feldrückholung, Impulsaktualisierung, t-nach-s zurück.
- 🟩 Der Löser: eine **modifizierte Poisson-Gleichung im Laborsystem** mit dem 1/γ²-Term vor der z-Ableitung (Gl. 2), gelöst über die **Integrated Green Function** (Gl. 3–5) und die **Hockney-Methode** — Arrays doppelt so groß anlegen, Faltung per FFT, offene Randbedingungen. Mathematisch gleichwertig zum Lösen im Ruhesystem und Zurücktransformieren, was andere Codes machen.
- 🟨 Das Hauptergebnis: Der Speicherbedarf im Vorwärtsdurchlauf wächst **linear** mit der Zahl der Makroteilchen und der Zellen und ist proportional zur Zahl der Kicks. Für Cheetah gemessen: rund 1,4 kB je Makroteilchen, dominiert von Ladungsdeposition und Feldrückholung, sowie rund 1,2 kB je Zelle, dominiert von der Auswertung des IGF-Ausdrucks.
- 🟨 Ein hübsches Detail: Die FFT selbst verbraucht im Vorwärtsdurchlauf **keinen** zusätzlichen Speicher, weil der Rückwärtsschritt einer FFT nur die durchzureichenden Gradienten braucht und keine Zwischendaten.
- 🟨 Entschärfung, falls es eng wird: **Gradient Checkpointing** — nur ausgewählte Zwischenstände aufheben und den Rest im Rückwärtsdurchlauf neu ausrechnen. Rechenzeit gegen Speicher getauscht.
- 🟩 Anhang B enthält zwei Genauigkeitstests: das Feld eines relativistischen Gauß-Bunches gegen die analytische Integraldarstellung (10⁶ Makroteilchen, 128³ Zellen, Gitter über ±6σ, γ von 10 bis 10 000) und die freie Ausdehnung eines kalten, gleichmäßig gefüllten Bunches.
- 🟩 Ausblick der Autoren: Iterative Löser im Ortsraum (Multigrid, GMRES, Krylov mit Multigrid-Vorkonditionierung) sind in Tracking-Codes verbreitet, weil sie mit verwickelten Strahlrohrgeometrien umgehen. Deren Speicherverhalten weicht vermutlich vom linearen Gesetz ab — ⬛ **wurde hier aber ausdrücklich nicht untersucht.**

**Warum das für dich zählt:** Hier steht die **Konkurrenz zu deinem Ersatzmodell** in Zahlen. Wenn Cheetah inzwischen einen echten differenzierbaren PIC-Löser mitbringt, musst du in deiner Aufgabenbeschreibung sauber begründen, wozu es daneben noch ein neuronales Modell braucht. Das Paper liefert dir die Begründung gleich mit: Der Speicher wächst mit jedem Kick weiter, und bei vielen Umläufen oder langen Beschleunigern läuft dir der Grafikspeicher voll. Ein Surrogat, das einen ganzen Kick oder gar eine ganze Sektion in einem Aufruf ersetzt, hat einen **konstanten** Fußabdruck. Das ist ein starkes, quantitativ belegbares Argument für dein Thema — und es kommt aus dem Papier deiner eigenen Leute.

**Der Absatz, den du zweimal liest:** Das Fazit ab „thousands of turns or more". Dort steht die Motivationslücke, in die deine Arbeit hineinpasst, in zwei Sätzen formuliert.

> 📝 **Deine Notizen — Station 3**
> - Rechnung: Speicherbedarf meines Anwendungsfalls mit echtem Löser vs. Surrogat:
> - Argumente für die Aufgabenbeschreibung:
> - Was ich zu iterativen Lösern noch brauche:
> -

---

## Station 4 — 🟩🟪 `space_charge_kick.py`
### Cheetah-Element `SpaceChargeKick` — 608 Zeilen PyTorch

**Was das ist:** Die Implementierung dessen, was Station 3 beschreibt. Der Quelltext, mit dem du vermutlich deine Trainingsdaten erzeugst und in den dein Modell später eingehängt wird.

**Was drinsteht:**
- 🟪 Klasse `SpaceChargeKick(Element)`. Ändert **nur die Impulse**, die Positionen bleiben unberührt — gedacht zur Kombination mit einem `Drift`, der die Positionen fortschreibt. Das ist das Strang-Splitting aus Station 3, in Code gegossen.
- 🟪 Voreinstellungen, die du kennen musst: `grid_shape=(32,32,32)`, und die Gitterausdehnung wird in Vielfachen von σ angegeben, voreingestellt jeweils 3.0. Das PASC-Paper rechnet seine Genauigkeitstests dagegen mit 128³ Zellen über ±6σ. Der Unterschied zwischen Voreinstellung und Testaufbau ist ein realer Konvergenzhebel, den du im Blick halten musst, sobald du Daten erzeugst.
- 🟩 Der Ablauf in Methoden: `_array_rho` (Ladungsdeposition), `_integrated_green_function` (mit den ganzen Spiegelungen über `flip`, um das doppelt so große Hockney-Array aufzubauen), `_solve_poisson_equation` (`torch.fft.rfftn` → Multiplikation → `irfftn`), `_E_plus_vB_field` (zentrale Differenzen mit Nullrand, Skalierung mit 1/γ²), `_compute_forces`, `track`.
- 🟪 Der Docstring nennt die Herkunft: IGF-Methode nach PhysRevSTAB 9, 044204, Vorgehen ähnlich Ocelot, aber Poisson wird im Laborsystem gelöst statt im Strahlsystem (Physics of Plasmas 15, 056701).
- 🟪 Harte Einschränkung, ausdrücklich per `assert`: Tracking läuft **nur mit `ParticleBeam`**. Der NN-Surrogat aus Station 1 arbeitet dagegen auf `ParameterBeam`. Diese Kluft zwischen Momenten- und Teilchendarstellung zieht sich durch dein ganzes Thema.
- 🟪 Kräftige Vektorisierung über Broadcasting und Flatten der Batch-Dimensionen. Wenn du Trainingsdaten stapelweise erzeugen willst, ist das dein Freund.

**Warum das für dich zählt:** Das ist deine Datenquelle und dein Einbauort in einem. Wer hier die Signatur von `track` und die Bedeutung von `effect_length`, `grid_shape` und den drei `grid_extent_*` verstanden hat, kann sein eigenes Element schreiben, das dieselbe Schnittstelle bedient — und genau das ist die technische Kernaufgabe eines modularen Surrogats.

**Der Absatz, den du zweimal liest:** `_solve_poisson_equation` zusammen mit `_integrated_green_function`. Wenn du je über einen Fourier-Neural-Operator nachdenkst, ist das hier die Stelle, an der bereits im Frequenzraum gearbeitet wird. Diese Parallele ist kein Zufall und einen eigenen Gedanken wert.

> 📝 **Deine Notizen — Station 4**
> - Setup, mit dem ich meine ersten Daten ziehe:
> - Schnittstelle, die mein Element bedienen muss:
> - Fragen an den Betreuer zum Code:
> -

---

# 🔗 Wie die Dateien ineinandergreifen

```
                    🟦 p305 (Schindl)
                    Physik, analytisch
                    Tune-Shift, Laslett
                            │
                            │ liefert Begriffe, Grenzfälle, Validierungsphysik
                            ▼
    ┌───────────────────────────────────────────────────┐
    │              C H E E T A H                        │
    │                                                   │
    │  1️⃣ PRAB 2024 ──────────► 3️⃣ PASC '26            │
    │  Code + 5 Anwendungen     echter PIC-Löser        │
    │  🟥 NN-Surrogat für SC     🟨 Speicher & Autodiff  │
    │  (ParameterBeam)          (ParticleBeam)          │
    │         │                        │                │
    │         └────────┬───────────────┘                │
    │                  ▼                                │
    │            4️⃣ space_charge_kick.py                │
    │            Das Ganze als Quelltext                │
    └───────────────────────────────────────────────────┘
```

**Überschneidungen:**
- 1️⃣ und 3️⃣ teilen sich einen Autor (Jan Kaiser) und dasselbe Werkzeug. 3️⃣ verweist für die Code-Details ausdrücklich auf 1️⃣.
- 3️⃣ und 4️⃣ beschreiben **exakt dasselbe** — einmal als Formelfolge, einmal als PyTorch. Lies sie nebeneinander, das spart dir Stunden.
- 1️⃣ und 2️⃣ berühren sich beim 1/γ²-Verhalten und bei der bi-gaußförmigen Ladungsdichte, die Fujitas PINN laut Verwandtschafts-Abschnitt als Annahme hineinsteckt.

**Was jeweils nur eine Datei hergibt:**
- **Nur 2️⃣:** Ringmaschinen, Tune-Shift und Tune-Spread, Bildladungen, Laslett-Koeffizienten, Space-Charge-Grenze. Die drei Cheetah-Dokumente kennen ausschließlich Einzeldurchläufe mit offenen Rändern.
- **Nur 1️⃣:** Der komplette ML-Werkzeugkasten, Trainingsdatenrezept, Hyperparameter, Anschluss an RL und Bayesische Optimierung, plus die Literaturliste zu PINNs und Nachbararbeiten.
- **Nur 3️⃣:** Die Kostenseite der Differenzierbarkeit. Speicherskalierung, Gradient Checkpointing, Genauigkeitstests des Lösers.
- **Nur 4️⃣:** Voreingestellte Werte, Signaturen, die tatsächliche Schnittstelle. Kein Paper verrät dir, dass die Gitterausdehnung ab Werk bei 3σ liegt.

**Zeitliche Schichtung, die du im Kopf behalten musst:** Das PRAB-Paper von 2024 schreibt ausdrücklich, dass kollektive Effekte in Cheetah ab Werk weggelassen werden. Das PASC-Paper von 2026 und der Quelltext widersprechen dem — dort gibt es den Löser. Beim Zitieren also aufpassen, welcher Stand gemeint ist.

---

# 🔁 Wiederkehrende Themen quer durch das Material

**1. Der Preis-Genauigkeits-Handel** 🟪🟩
Zieht sich durch alles. Cheetah tauscht Genauigkeit gegen Tempo (1️⃣, ausdrücklich als bewusste Entscheidung benannt). Der Surrogat tauscht Allgemeingültigkeit gegen einen Faktor 3600 (1️⃣). Gradient Checkpointing tauscht Rechenzeit gegen Speicher (3️⃣). Splines erster Ordnung tauschen Feldauflösung gegen Rechenaufwand (3️⃣/4️⃣). Deine Arbeit wird eine weitere Position auf dieser Achse besetzen, und du wirst begründen müssen, welche.

**2. Momente gegen Teilchen** 🟥🟩
`ParameterBeam` mit µ und Σ gegen `ParticleBeam` mit N Makroteilchen. Der bestehende Surrogat kann nur ersteres, der echte Löser nur letzteres. Ein Modell, das auf Teilchenwolken oder auf Feldern arbeitet — Neural Operator, PointNet, Graphennetz — würde diese Kluft schließen. Genau dort liegt aus meiner Sicht der interessanteste Zipfel deines Themas.

**3. Differenzierbarkeit als durchgehende Anforderung** 🟨
Alle drei Cheetah-Dokumente bestehen darauf. Es reicht nicht, dass ein Modell richtig rechnet — es muss auch sauber Gradienten durchreichen, weil es sonst weder für RL noch für gradientenbasiertes Tuning noch als BO-Prior taugt. Das schränkt deine Architekturwahl ein und gehört früh in die Aufgabenbeschreibung.

**4. Der Faktor 1/γ²** 🟦🟩
Taucht in jeder einzelnen Datei auf: bei Schindl als Ergebnis der Kraftbilanz, im PASC-Paper als Koeffizient vor ∂²φ/∂z², im Code als `igamma2`, im PRAB-Paper als der Grund, warum der Surrogat bei niedrigen Energien größere Korrekturen ausgibt. Wenn dein Netz eine physikalische Vorgabe eingebaut bekommt, ist das der erste Kandidat.

**5. Ocelot als stiller Bezugspunkt** 🟪
Trainingsdaten kommen aus Ocelot (1️⃣), Cheetahs Berechnungen lehnen sich an Ocelot an (1️⃣), der Docstring vergleicht sich mit Ocelot (4️⃣). Falls du selbst Daten erzeugst, wirst du dich entscheiden müssen: Ocelot als etablierte Referenz oder Cheetahs eigener Löser, der dafür flott und differenzierbar ist.

**6. Validierung bleibt dünn** 🟥
Der Surrogat in 1️⃣ wird über einen einzigen Energieschwenk geprüft, bei ansonsten festgehaltenen Parametern. Der Löser in 3️⃣ wird an zwei idealisierten Fällen geprüft. Ein belastbares Prüfprotokoll für Space-Charge-Ersatzmodelle existiert im vorliegenden Material schlicht nicht.

---

# ⬛ Lücken — hier musst du selbst nachschlagen

| ⚠️ | Was fehlt | Warum das weh tut | Wo du suchst |
|---|---|---|---|
| ⬛ **1** | **Kein einziges PINN-, NODE- oder Neural-Operator-Paper** | Deine Mail nennt genau diese drei Familien als das, wofür dein TUHH-Betreuer offen ist. Im Wissensspeicher liegt dazu nur ein Verweis zweiter Hand: Fujita, IEEE Access 9, 164017 (2021). | Fujita direkt besorgen; dazu Raissi et al. zu PINNs, Chen et al. zu Neural ODEs, Li et al. zu Fourier Neural Operators und DeepONet von Lu et al. |
| ⬛ **2** | **Nichts zu Julia** | Punkt 2d deiner Mail. Kein Material zu SciML, Lux.jl, DiffEqFlux, NeuralOperators.jl oder zu Python-Anbindung über PythonCall/juliacall. | Ökosystem-Dokumentation; außerdem klären, ob Cheetah selbst überhaupt aus Julia heraus ansprechbar wäre |
| ⬛ **3** | **Kein Prüfprotokoll für Ersatzmodelle** | Woran misst du Erfolg? Fehler in σ? Emittanzwachstum? Erhaltungsgrößen? Verhalten außerhalb des Trainingsgebiets? | Fehlt komplett — das ist ein eigenständiger Beitrag, den du leisten kannst |
| ⬛ **4** | **Speicherverhalten iterativer Löser** | Vom PASC-Paper ausdrücklich ausgeklammert. Relevant, sobald echte Strahlrohrgeometrien ins Spiel kommen. | ImpactX, Xsuite und die dort zitierte Multigrid-Literatur |
| ⬛ **5** | **Kein Datensatz, kein Erzeugungsskript** | 1️⃣ nennt Spannen und Stichprobenzahl, nicht aber den Code. Du beginnst bei der Datenerzeugung bei null. | Cheetah-Repository und das verlinkte Beispiel-Repository der Autoren |
| ⬛ **6** | **Ringe, Bildladungen, Randbedingungen numerisch** | Schindl behandelt Laslett-Koeffizienten und leitende Wände ausführlich — der Code kennt ausschließlich offene Ränder über Hockney. Zwischen beiden klafft ein Loch. | Falls dein Anwendungsfall ein Synchrotron ist, wird das zur Kernfrage. Falls es ein Linac bleibt, kannst du es sauber ausschließen |
| ⬛ **7** | **Rechenzeit des echten Lösers** | 3️⃣ misst Speicher, nicht Laufzeit. Für die Aufwandsrechnung „Surrogat gegen Löser" fehlt damit die halbe Achse. | Selbst messen — das ist ein überschaubares erstes Experiment |
| ⬛ **8** | **Keine Aufgabenbeschreibung, keine formalen Vorgaben** | Umfang, Bewertungskriterien, Abgabeform, Rolle der TUHH gegenüber DESY. | Betreuer fragen, am besten zusammen mit deinem Entwurf |
| ⬛ **9** | **Nichts zum SASE-Thema** | Passt zu deiner Entscheidung für Thema 2, aber falls du dich noch einmal umentscheidest, stehst du ohne Material da. | Nur relevant, falls die Entscheidung wieder aufgemacht wird |

---

# 🎯 Rückbindung an deine Mail

| Was du geschrieben hast | Was das Material dazu hergibt |
|---|---|
| „Effekt ist bereits ganz gut verstanden" | Bestätigt sich. 2️⃣ liefert die Analytik, 3️⃣ und 4️⃣ das etablierte numerische Vorgehen. Du musst keine Physik erfinden. |
| „es gibt einige Paper dazu und Anhaltspunkte, wo ich anfangen kann" | Der beste Anhaltspunkt ist Abschnitt III E in 1️⃣ samt dem dort formulierten Ausblick. Der Rest der Literatur liegt aber ⬛ **außerhalb** des Wissensspeichers. |
| „offen für PiNNs, NODEs und Neural Operators" | ⬛ **Lücke 1.** Das Material zeigt einen schlichten MLP-Residuum-Ansatz und erwähnt PINNs nur beiläufig. Zu NODEs und Neural Operators steht hier nichts. Diese Vorauswahl triffst du auf eigener Grundlage. |
| „Programmierung in Python, da die Simulation auch in Python geschrieben ist" | Klar gedeckt. Cheetah ist PyTorch bis in die Grundmauern, Modelle und Simulation teilen sich denselben Autodiff-Graphen. Ein Bruch an dieser Stelle würde die Differenzierbarkeit kosten, die alle drei Papers als zentral behandeln. |
| „prüfe vorher eine Implementierung in Julia mit Interfaces für Python" | ⬛ **Lücke 2.** Null Material. Aus dem Vorliegenden spricht viel dagegen: Wenn dein Modell als Cheetah-`Element` laufen soll, muss es im PyTorch-Graphen sitzen. Eine Julia-Umsetzung müsste an dieser Nahtstelle Gradienten über die Sprachgrenze reichen. Nachprüfen, aber mit Skepsis. |
| „Aufgabenbeschreibung, Themeneingrenzung, Arbeitspakete, Gliederung entwerfen" | Die drei tragfähigsten Zuschnitte, die aus dem Material herausfallen, stehen unten. |
| „Anmeldung zum 01.09." | Fünf Wochen. Der Vorschlag darunter ist darauf gerechnet. |

## Drei mögliche Themenzuschnitte aus dem Material

**A — Vom Moment zur Wolke** 🟥🟩
Den Surrogat aus 1️⃣ von `ParameterBeam` auf `ParticleBeam` heben. Direkt an den Ausblick der Autoren angedockt, technisch klar umrissen, gut prüfbar. Architekturfrage: PointNet, Graphennetz oder Neural Operator.

**B — Physik ins Netz einbauen** 🟥🟦
Ein physikinformiertes Modell, das die modifizierte Poisson-Gleichung aus 3️⃣ als Verlustterm mitführt, statt sie zu lösen. Fujitas Arbeit als Ausgangspunkt, Schindls analytische Grenzfälle als Prüfsteine. Riskanter, dafür der eigenständigere Beitrag.

**C — Der Kick als Operator** 🟥🟨
Einen ganzen Space-Charge-Kick durch einen Fourier Neural Operator ersetzen und gegen den echten Löser antreten lassen — in Genauigkeit, Laufzeit und Speicher. Die inhaltliche Nähe zwischen FNO und der ohnehin FFT-basierten Faltung in `_solve_poisson_equation` macht das reizvoll. Liefert zugleich die Zahlen, die in Lücke 7 fehlen.

## Fahrplan bis zum 01.09.

| Woche | Was ansteht | Ergebnis |
|---|---|---|
| **28.07 – 03.08** | Station 1 und 2 durcharbeiten | Begriffe sitzen, Zuschnitt eingegrenzt |
| **04.08 – 10.08** | Station 3 und 4, dazu Lücke 1 schließen (PINN/NODE/FNO beschaffen und querlesen) | Architekturentscheidung vorbereitet |
| **11.08 – 17.08** | Cheetah installieren, `SpaceChargeKick` laufen lassen, Lücke 7 selbst messen | Erste eigene Zahlen |
| **18.08 – 24.08** | Aufgabenbeschreibung und Arbeitspakete schreiben, Julia-Frage abhaken | Entwurf zur Abstimmung |
| **25.08 – 31.08** | Gliederung, Rückmeldung einholen, nachschärfen | Anmeldefertige Unterlagen |

---

## 📌 Freies Notizfeld

**Offene Fragen an den Betreuer:**
- Was soll im Rahmen der Masterarbeit NUN erarbeitet werden?
- Neural Operator Modell für Space Charge Simulationen?
- Oder sind vergleichbare Modelle auch ok?
- Z.B. PiNN, NODE
- Oder ein Vergleich aller möglichen Surrogate-Modelle?
- Implementierung inkl. Test + Benchmark oder theoretisches Konzept ausreichend?
- Implementierung in Julia ok oder zwingend Python?
- In bisherigen Quellen sind andere Ansätze im Ausblick genannt
- [Ansätze benennen]
- Soll ich diese stattdessen verfolgen?
- Vergleich des Surrogate zur bisherigen Cheetah-Simulation
- (Benchmarking, Rechenzeit, Speicheraufwand)
- Genauigkeit/ Feldauflösung vs. Rechenzeit/ Rechenaufwand vs. Speicher: Was davon soll ich priorisieren?
- Neural Operator (altn. PiNN oder NODE) soll bisheriges Modell von ParameterBeam (Moment) auf ParticleBeam (Wolke) heben und so mit ganzen Teilchenwolken/ -feldern arbeiten?
- Differenzierbarkeit Pflicht!
- \fraq{1}{\gamma^2} als Faktor (für was?) in Netzarchitektur berücksichtigen
- Was als Datenquelle benutzen: Ocelot oder Cheetah?
- Rechenzeit des Cheetah-Lösers noch selber zu messen?
- Allgemeine Aufgabenstellung: modulares/ Parametriertes Surrogate-Modell für Anwendungen in der Hochenergiephysik?
- (Daher auch Neural Operators, um einen generischen „Funktionsfinder“ zu haben statt eines NN, dass ein spezifisches Problem löst?)
- PiNN („anteilig“ in Neural Operator) mit Poisson-Gleichung aus 2026: Dhamrait et al. Als Verlustterm + weiteren Randbedingungen etc. aus Schindl: Space Charge (CERN-Paper)
- Space-Charge-Kick durch einen Fourier Neural Operator ersetzen
- und gegen den echten Löser (Cheetah?) antreten lassen
- hinsichtlich Genauigkeit, Laufzeit und Speicher
- Wie wurde das bisherige neuronale Netz in Cheetah trainiert?
- D.h. mit welchen Daten? (Aus Simulationen/ Messungen?)
- Line 308 space_charge_kick.py: Hier ggf. einhaken für Fourier Neural Operator
- Space Charge ist nur bei niedrigen Energien (z.B. 1 MeV vs. 250 MeV relevant). Vgl. Durchläufe mit sc_walktrough bzw. shapes.py, wo 1/γ² bei geringen Energien um viele Größenordnungen größer ist.

**Entscheidungen, die ich getroffen habe (mit Datum):**
- (17.08.2026) Arbeitstitel: Neural-Operator-Surrogate für Space Charge Effekte im Strahlsimulationsframework Cheetah
- (17.08.2026) Aufgabenstellung:
    - Option 1:
    Innerhalb von Cheetahs SpaceChargeKick.py wird der bestehende Feldlöser _integrated_green_function() (IGF) durch einen (Fourier) Neural Operator abgelöst und mit diesem verglichen — in Genauigkeit, Laufzeit und vor allem Speicherverhalten (Forward Pass bei Rückwärtsdifferentiation, wie in Dhamrait et al.).
    - Option 2:
    Neural Operator Surrogate ähnlich zu Neural Network Surrogate, dass in doi: 10.1103/PhysRevAccelBeams.27.054601 Abschnitt III.E vorgestellt wurde, trainieren und validieren


**Literatur, die ich noch beschaffen muss:**
- J. Kaiser, O. Stein, and A. Eichler, Learning-based optimisation
of particle accelerators under partial observability
without real-world training, in Proceedings of the 39th
International Conference on Machine Learning (ICML-
2022), Baltimore, Maryland, edited by K. Chaudhuri, S.
Jegelka, L. Song, C. Szepesvari, G. Niu, and S. Sabato
(PMLR, Baltimore, 2022), Vol. 162, pp. 10575–10585.
-J. Kaiser, A. Eichler, S. Tomin, and Z. Zhu, Machine
learning for combined scalar and spectral longitudinal
phase space reconstruction, in Proceedings of the 14th
International Conference for Particle Accelerators (2023),
https://dx.doi.org/10.18429/JACoW-IPAC2023-THPL019.
- A. Paszke et al., PyTorch: An imperative style, highperformance
deep learning library, in Advances in Neural
Information Processing Systems 32, edited by H. Wallach,
H. Larochelle, A. Beygelzimer, F. d’Alch´e Buc, E. Fox, and
R. Garnett (Curran Associates, Inc., Vancouver, 2019),
pp. 8024–8035.
- K. Flöttmann, ASTRA: A space charge tracking algorithm,
1997.
- I. Agapov, G. Geloni, S. Tomin, and I. Zagorodnov,
OCELOT: A software framework for synchrotron light
source and FEL studies, Nucl. Instrum. Methods Phys.
Res., Sect. A 768, 151 (2014).
- G. Iadarola et al., Xsuite: An integrated beam physics
simulation framework, 2023.
- F. Mayet, M. Hachmann, K. Floettmann, F. Burkart, H.
Dinter, W. Kuropka, T. Vinatier, and R. Assmann, Predicting
the transverse emittance of space charge dominated
beams using the phase advance scan technique and a fully
connected neural network, Phys. Rev. Accel. Beams 25,
094601 (2022).
- A. Scheinker, F. Cropp, S. Paiagua, and D. Filippetto, An
adaptive approach to machine learning for compact particle
accelerators, Sci. Rep. 11, 19187 (2021).
- J. Qiang, Differentiable self-consistent space-charge simulation
for accelerator design, Phys. Rev. Accel. Beams 26,
024601 (2023).
- J. Gonzalez-Aguilera et al., Towards fully differentiable
accelerator modeling, in Proceedings of the 14th
International Particle Accelerator Conference, IPAC-
2023, Venice, Italy (JACoW, Geneva, Switzerland,
2023), pp. 2797–2800.
- O. Stein, J. Kaiser, and A. Eichler, Accelerating linear
beam dynamics simulations for machine learning applications,
in Proceedings of the 13th International Particle
Accelerator Conference IPAC-2022, Bangkok, Thailand
(JACoW, Geneva, Switzerland, 2022).
- K. L. Brown, A first and second order matrix theory for the
design of beam transport systems and charged particle
spectrometers, Adv. Part. Phys. 1, 71 (1968).
- W. Falcon and The PyTorch Lightning team, PyTorch
Lightning, 2019.
- L. Biewald, Experiment tracking with Weights and Biases,
wandb.com (2020).
R. T. Sandberg, R. Lehe, C. E. Mitchell, M. Garten, A.
Myers, J. Qiang, J.-L. Vay, and A. Huebl, Synthesizing
particle-in-cell simulations through learning and GPU
computing for hybrid particle accelerator beamlines,
arXiv:2402.17248.
- . K. Birdsall and A. B. Langdon. 2004. Plasma Physics via Computer Simulation.
CRC Press. doi:10.1201/9781315275048
- @article{kaiser2024cheetah,
    title        = {Bridging the gap between machine learning and particle accelerator physics with high-speed, differentiable simulations},
    author       = {Kaiser, Jan and Xu, Chenran and Eichler, Annika and Santamaria Garcia, Andrea},
    year         = 2024,
    month        = {May},
    journal      = {Phys. Rev. Accel. Beams},
    publisher    = {American Physical Society},
    volume       = 27,
    pages        = {054601},
    doi          = {10.1103/PhysRevAccelBeams.27.054601},
    url          = {https://link.aps.org/doi/10.1103/PhysRevAccelBeams.27.054601},
    issue        = 5,
    numpages     = 17
}
- @inproceedings{stein2022accelerating,
    title        = {Accelerating Linear Beam Dynamics Simulations for Machine Learning Applications},
    author       = {Stein, Oliver and Kaiser, Jan and Eichler, Annika},
    year         = 2022,
    booktitle    = {Proceedings of the 13th International Particle Accelerator Conference}

**Ideen, die mir beim Lesen gekommen sind:**
- EASy in Cheetah implementieren und Cheetah Simulation auf EASy Case-Study "Untersuchung, ob Strahlexposition in FLASH1 zu Messsystemausfall geführt haben kann" (Ausblick)
- Fördermöglichkeiten recherchieren unter
    - http://www.sbb-stipendien.de/weiterbildungsstipendium.html
	- http://www.e-fellows.net/show/detail.php/5789
	- http://www.stipendiumplus.de
- Ausschlaggebende Punkte für die Masterarbeit:
	- Methodik:
		- **Spezialwissen (Fakten, Theorien, Methoden) aus dem Studium** zur Bearbeitung der Aufgabenstellung anwenden
		- geeignete Methoden überhaupt finden und ggf. weiterentwickeln
		- sogar auf ggf. komplexe/ unvollständige Problemstellungen anwenden
	- **kritische** Auseinandersetzung mit/ Beurteilung von
		- **in der Tiefe zu erklärenden** Ansätzen, Terminologien und aktuellen Entwicklungen
		- der im **eigenen Fachgebiet** einzuordnen **Forschungsfrage**
		- ... sowie dem **aktuellen Forschungsstand**
		- **neue** wissenschaftliche Erkentnisse aus der eigenen Arbeit
	- Präsentation/ Verteidigung:
		- wissenschaftliche Fragestellung für ein **Fachpublikum** strukturiert, verständlich und **sachlich richtig** präsentieren
		- Fragen **fachkundig** beantworten
		- eigene Einschätzung **überzeugend** vertreten
	- Selbstständiges rojektorientieres Arbeiten:
		- Projekt in **sinnvoll realisierbare** Arbeitspakete **strukturieren**
		- diese **selbstständig** abarbeiten
		- d.h. auch unter **Einhaltung der Zeitvorgaben**
		- und ggf. **Delegieren/ Priorisieren** von Arbeitspaketen
		- sowie dem **bedarfsorientierten Entscheiden**, ein Arbeitspaket trotz unvollständiger Zielerfüllung als abgeschlossen zu werten, wenn der Stand für das Gesamtprojekt ausreicht
		- Vertiefte Einarbeitung in teilweise unbekannte Arbeitsgebiete
		- und Erschließung dafür benötigter Informationen
		- **korrekte und gründliche** Anwendung der Techniken des wissenschaftlichen Arbeitens
- "many-query setting nennen: Situationen, in denen dieselbe Gleichung mit tausenden verschiedenen Eingaben gelöst werden muss. Davon gibt es im Wesentlichen vier Sorten — inverse Probleme, Unsicherheitsquantifizierung, Design-Optimierung und Echtzeitregelung." [Zitat Claude Opus 5 Hoch: cse_01DCLY9K6zmANfJkuuZXosJU (Frage: Was ist ein praktisches Beispiel für einen Lösungsoperatoren von parametrischen PDEs? Wie kann ich mir das in einer praktischen Anwednung, wie den von Li aufgeführten sowie im Rahmen von Cheetah vorstellen?)]