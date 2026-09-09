# Arbeitspakete, Einarbeitung und Bewertung des Bearbeitungsplans

Stand: 07.09.2026 · Grundlage: handschriftliche Blätter vom 06.09. und 09.09.2026,
`Bearbeitungsplan.xlsx`, `space_charge_kick.py`

Zugehörige Ergebnisse: interaktive Themenkarte (Artifact) und
`Bearbeitungsplan_ueberarbeitet.xlsx`.

## Rahmen

- **Zieltermin 24.03.2027** = Anmeldung am 24.09.2026 plus sechs Monate
  Bearbeitungszeit. Noch nicht angemeldet, also Planungsziel, keine Frist.
- Arbeitstage Mo–Do, jeder Freitag blockiert (Familie) → 4 Arbeitstage je Woche.
- Von 07.09.2026 bis 24.03.2027 bleiben 112 Arbeitstage, davon 17 als Puffer
  ausgewiesen (15 %).
- Julia ist gestrichen; statt Implementierung ein begründender Absatz im Manuskript.

### Vor dem Festzurren zu klären

1. **Läuft die Frist ab der Anmeldung oder ab der Ausgabe des Themas?** In vielen
   Prüfungsordnungen zählt die Ausgabe, die einige Tage bis Wochen später liegt.
   Frage ans Prüfungsamt.
2. **Dürfen Ergebnisse aus P1 und P2** — also von vor der Ausgabe des Themas — in
   der Arbeit verwendet werden? Frage an den Betreuer, Antwort ins Protokoll.

## Bewertung des bisherigen Vorankommens

**Inhaltlich weiter als der Plan, strukturell dahinter.** Die Ist-Balken zeigen ein
klares Muster: alles, was mit Neural Operators und dem Schreiben der Motivation zu
tun hat, wurde vorgezogen abgearbeitet (lila am 27.08. bis 04.09., Aufgaben aus den
Phasen 3 und 5, geplant für Mitte September bzw. Ende Oktober). Gleichzeitig steht
bei zwei Aufgaben gar kein Ist:

- „Datengenerierung mit Cheetah verstehen" (Soll 22.–26.08.)
- „Testdurchläufe mit Cheetah" (Soll 26.–27.08.)

Genau diese beiden blockieren die gesamte Phase 4. Ohne laufende Cheetah-Testläufe
keine Daten, ohne Daten kein Training. Das Risiko ist nicht mangelnder Einsatz,
sondern dass die Cheetah-Seite unangetastet bleibt, bis es weh tut.

Qualitativ ist die Vorarbeit stark: Zielsatz, vier Zuschnitte und die
Argumentationskette in der Motivationsnotiz sind ein Ergebnis, das viele erst in
Woche 6 haben.

## Strukturelle Schwächen des alten Plans

1. **Phase 4 grob unterschätzt.** 14.09.–11.10. für Datengenerierung, Training,
   Evaluation, Pipeline und Vergleich. Bei 4 Arbeitstagen je Woche sind das
   16 Tage; realistisch ist etwa das Doppelte.
2. **Kein Puffer.** Nirgends im Plan. Bei einer Masterarbeit sind 15–20 % das Minimum.
3. **Profiling fehlt vollständig** — obwohl die eigene Motivationsnotiz es als
   ersten belastbaren Beitrag benennt.
4. **Einarbeitungspakete tauchen als Aufgaben nicht auf**, obwohl sie Zeit kosten.
5. **Schreiben startet zu spät** (1. Fassung erst ab 14.10. als Block). Methodik
   entsteht beim Bauen und muss dann mitgeschrieben werden.
6. **Tagesgenaue Balken über ein halbes Jahr** sind Fiktion; schon in Woche 1
   verrutscht. Besser: Wochenziele plus Fertig-Kriterium je Paket.
7. **Abgabetermin nicht an die Anmeldung gekoppelt.** Der 01.03.2027 setzte eine
   Anmeldung zum 01.09.2026 voraus, die nicht stattgefunden hat. Korrigiert.

Gut und beizubehalten: Phasenstruktur, Soll/Ist-Doppelzeilen, zweiwöchige
Konsultationen, eingetragene blockierte Tage.

## Arbeitspakete

| ID | Paket | Beantwortet | Tage | Braucht vorher |
|---|---|---|---|---|
| P1 | Cheetah lauffähig, einen Kick tracken | Durchsicht 1+3, Leitfrage 5 | 3 | — |
| P2 | Profiling `SpaceChargeKick` (Zeit & Speicher, vorwärts/rückwärts) | Durchsicht 2, Leitfrage 11 | 3 | P1 |
| P3 | Zuschnitt & Schnittstelle festlegen (Variante A: ρ → φ) | Leitfrage 6 | 2 | P2 |
| P4a | Datengenerator & Entdimensionalisierung | Leitfragen 7, 7.5, 8 | 5 | P3 |
| P4b | Datensatz erzeugen, prüfen & einfrieren | Leitfrage 8 | 2 | P4a |
| P5 | Modell aufsetzen & trainieren | Leitfrage 9 | 8 | P4b, E3, E4 |
| P6 | Baselines: Hockney (gleich + feiner) und CNN-Surrogat | Leitfrage 11 | 4 | P2, P5 |
| P7a | Genauigkeit & Fehlerstreuung | Leitfrage 10 | 2 | P5, P6, E5 |
| P7b | Auflösungsstudie (Zero-Shot) | Leitfrage 12 | 2 | P7a |
| P7c | Gradiententreue & Differenzierbarkeit | eigene Forschungsfrage | 2 | P7a |
| P7d | Rechenzeit- & Speichervergleich | Leitfrage 11 | 2 | P2, P6 |
| P8 | Einbau ins Lattice & Langstreckentest | Leitfrage 11 | 5 | P7a–d |
| P9 | Ergebnisse konsolidieren | — | 3 | P8 |

**P1 und P2 sind Türsteher.** Ohne sie ist alles Weitere Spekulation — und genau sie
fehlen im bisherigen Ist-Stand.

## Einarbeitungspakete

Prinzip: **lernen, wenn die Arbeit es braucht**, nie am Stück auf Vorrat. Jeder Block
endet an einem Kriterium, nicht an einer Seitenzahl.

| ID | Thema | Quelle | Tage | Fertig, wenn |
|---|---|---|---|---|
| E1 | Faltung & FFT (auffrischen) | 3Blue1Brown zur Faltung; NumPy-FFT-Doku; zirkuläre gegen lineare Faltung selbst provozieren | 2 | du erklärst den Faltungssatz und warum Cheetah auf 64³ polstert |
| E2 | PDE & Poisson (auffrischen) | Numerik-Lehrbuch, Kapitel elliptische Probleme; Jackson Kap. 1–2, nur Poisson | 1 | du schreibst die SC-Gleichung hin und benennst die Randbedingung |
| E3 | PyTorch (neu) | „Learn the Basics" → `torch.autograd` → `nn.Module`, `DataLoader`, Trainingsschleife. Stevens/Antiga/Viehmann nur als Nachschlagewerk | 3 | du erklärst das Skelett-Repo Zeile für Zeile und tauschst eine Schicht |
| E4 | ML: Training & Regularisierung (neu) | Goodfellow/Bengio/Courville, **nur** Kap. 5, 7, 8 | 2 | du begründest Lernrate, Batchgröße, Regularisierung, Abbruch mit je einem Satz |
| E5 | ML: Bewertung & Metriken (neu) | kein Lehrbuch — Metrikabschnitte der drei NO-Paper auswerten | 1 | Tabelle: Kennzahl, was sie misst, wo ihre Grenze liegt |
| E6 | Neural Operators vertiefen | Azizzadenesheli (Überblick) → Li (FNO, parallel zum Code) → Kovachki (selektiv); dazu `neuraloperator`-Doku | 3 | Leitfragen 1–3 schriftlich beantwortet, mit Quellen |

Aus der Bibliothek zu leihen: ein PyTorch-Lehrbuch und ein ML/DL-Lehrbuch — beide
als Nachschlagewerk, nicht als Lesestoff von vorn.

## Terminraster (Fassung vom 07.09.2026)

| Zeitraum | Inhalt |
|---|---|
| 07.09.–28.09. | P1, P2, P3 · E1, E2, E3 · Gliederung · **24.09. Anmeldung** |
| 29.09.–13.10. | P4a, P4b · E4 |
| 14.10.–02.11. | P5 · E6 |
| 03.11.–10.11. | P6 · E5 |
| 11.11.–24.11. | P7a bis P7d |
| 25.11.–08.12. | P8, P9 |
| 09.12.–17.12. | Puffer · Start 1. Fassung |
| 21.12.–30.12. | Puffer (Feiertage, reduzierte Wochen) |
| 04.01.–21.01. | 1. Fassung fertigstellen · **28.01. Konsultation dazu** |
| 25.01.–28.01. | Puffer, Rückmeldung abwarten |
| 01.02.–18.02. | 2. Fassung |
| 22.02.–02.03. | Korrekturlesen & Verbessern |
| 03.03.–09.03. | Ergebnisgestaltung Typoskript |
| 10.03.–18.03. | Puffer Schlussphase |
| 22.03.–23.03. | Vervielfältigung & Buchbindung |
| **24.03.2027** | **Abgabe** |

## Spielregeln für den neuen Plan

1. Jeden Montag 30 Minuten: Ist-Zeile der Vorwoche füllen, eine Zeile notieren, was
   gebremst hat. Das ist später das Material für das Kapitel Vorgehen.
2. Ein Paket ist fertig, wenn sein Kriterium erfüllt ist — nicht wenn die Zeit um ist.
3. Einarbeitung immer an dem Paket, das sie gerade braucht.
4. Jeder Trainingslauf mit Konfiguration und Ergebnis ins Protokoll, am selben Tag.
5. Läuft ein Paket über, wird Puffer verbraucht, nicht der nächste Termin verschoben.
   Ist der Puffer leer, wird der Umfang gekürzt und das in der Konsultation angesprochen.