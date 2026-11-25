# Fischerei und Wild: Wie viel entnehmen? Quote, Aufwand oder Schwelle – Streamlit-Demo

**[→ Demo live ausprobieren](https://sebastianhanisch-fischerei-entnahme-demo.streamlit.app/)**

Fall-Demo für die Website „Sebastian Hanisch – Operations Research und Machine Learning“ (`demos.html`). Ein Bestand (Fisch oder Wild) wächst nach einer logistischen Wachstumsfunktion mit **Wachstumsrauschen**, und jedes Jahr wird ein Teil entnommen. Verglichen werden vier Entnahmeregeln: eine feste **Quote** (Menge je Jahr), ein fester **Aufwand** (Anteil des Bestands), eine **Escapement-Schwelle** (alles, was der gemessene Bestand über einer Schwelle liegt) und eine **gedämpfte Schwelle** (ein Anteil davon, zwei Regler). Die Schwellenregeln lesen einen **Survey** mit **Messrauschen**, Quote und Aufwand lesen ihn nicht. Jede Regel wird fair auf getrennten Trainingspfaden abgestimmt und auf frischen Testpfaden bewertet, immer mit dem **Kollaps** neben dem Fang. Als Referenz dient die Wertiteration (das Optimum bei vollem Wissen über den Bestand), und ein vorgerechnetes Raster zeigt, **wo die Reihenfolge von Escapement und festem Aufwand kippt**. Ein Tab „Von Hand“ rechnet das Modell ohne Zufall in exakten Brüchen.

Alle Zahlen sind **Annahmen**: kein Bestand, keine Fischerei- oder Jagddaten wurden gemessen oder nachgeschlagen. Die Demo zeigt, was in diesem Modell passiert, und gibt keine Bewirtschaftungsempfehlung.

## Warum dieses Problem

Eine Entnahmeregel ist ein Regelkreis: was heute entnommen wird, bestimmt den Bestand, der morgen wächst. Eine feste Quote (die Lehrbuchregel „Quote am MSY“, Maximum Sustainable Yield, größter Dauerertrag) sieht gut aus, solange nichts schwankt, und bricht in diesem Modell schon bei winzigem Wachstumsrauschen ein. Eine Schwelle auf dem gemessenen Bestand erreicht bei exaktem Survey in den Messungen das Ziel der DP-Politik, also der Wertiteration auf dem exakten Messwert simuliert (Differenz höchstens 0,0003; die Wertiteration rechnet ohne Kapazitätsgrenze, die Übereinstimmung ist gemessen, nicht bewiesen), aber ein verrauschter Survey kann sie hinter einen festen Aufwand zurückwerfen, der den Survey gar nicht liest. **Kernfragen:** Hält die gewählte Regel den Bestand? Versteckt das Ziel allein den Kollaps? Und: Bei welchem Messrauschen liegt die Schwelle hinter dem festen Aufwand?

## Modell

Bestand $B_t$ in Einheiten der Tragfähigkeit (Carrying Capacity) $K=1$, Jahrestakt, $B_0=K$, 200 Jahre je Pfad. **Reihenfolge in jedem Jahr:** Messwert $Y_t=B_t\exp(\sigma_m\xi_t-\sigma_m^2/2)$ (lognormal, Mittel 1), Entnahme $H_t$ nach der Regel mit $0\le H_t\le h_{\max}B_t$, Rest $S_t=B_t-H_t$, Wachstum $B_{t+1}=Z_t\max\{0,S_t+rS_t(1-S_t)\}$ mit $Z_t=\exp(\sigma_g\eta_t-\sigma_g^2/2)$ (Mittel 1). Die Entnahme geschieht **vor** dem Wachstum. Regeln: Quote $H=Q$, Aufwand $H=eB$, Escapement $H=\max(0,Y-S^*)$, gedämpfte Schwelle $H=f\max(0,Y-S^*)$ (jeweils begrenzt auf $h_{\max}B$). Ziel: gleichwertiger Jahresfang $J=\frac{1-\beta}{1-\beta^T}\sum_t\beta^tH_t$ (bei $\beta=1$ das Mittel der Jahre 50 bis 199). **Kollaps:** der Bestand fällt irgendwann in den Jahren 1 bis 200 unter die Kollapsgrenze (Voreinstellung 0,1 K); nicht absorbierend. Wird das Wachstum bei sehr großem Rest $S_t$ nach der Entnahme negativ (Rest über $(1+1/r)K$), setzt das Modell den Bestand $B_{t+1}$ auf 0 („Überschießen“, siehe „Befunde und Korrekturen“). Alle Formeln stehen im 📐-Expander der App (vollständiges Modell mit Anfangswerten, Reihenfolge, Zufallsprozess und Randfällen). Kern: `fis_algorithm.py`, `fis_dp.py`, `fis_evaluation.py`.

## Methodik

**Abstimmung ohne Strohmann:** jede Regel bekommt ihren Reglerwert von einem Gitter, bestimmt auf getrennten Trainingspfaden (1 000 Pfade, Zufalls-Seed 11) und bewertet auf Testpfaden (5 000 Pfade, Zufalls-Seed 22); alle Regeln sehen dieselben Zufallszahlen, Differenzen sind pfadweise gepaart. Gitter: Quote 0 bis 0,4 in Schritten von 0,005, Aufwand 0 bis 0,9 in 0,01, Schwelle 0 bis 1 in 0,01, gedämpfte Schwelle $S^*$ in 0,02 mal $f$ von 0,1 bis 1 in 0,1. Zwei Abstimmungen: **frei** (größtes mittleres Ziel) und **mit Kollapsgrenze** (größtes Ziel, wenn höchstens 10 % der Trainingspfade kollabieren; ist nichts zulässig, gilt das kleinste Kollapsrisiko und die Zeile heißt „nicht zulässig“); bei Gleichstand gilt der kleinste Index im Gitter (bei der gedämpften Schwelle zuerst der kleinste Wert von $S^*$, dann von $f$). Dazu die **Lehrbuch-Werte** (Quote $rK/4$, Aufwand $e=r/(2+r)$, Schwelle $K/2$), die **DP-Politik auf dem Messwert** und die **Wertiteration** als Referenz (801 Gitterpunkte, 41 Gauß-Hermite-Knoten, Wert $V(B)=\max_S[(B-S)+\beta\,\mathbb E\,V(Zf(S))]$). **Zustandsraum der Wertiteration:** der Bestand $B$ liegt auf einem gleichmäßigen Gitter in $[0, 2]$ (Schritt 0,0025), ein Bestand nach dem Wachstum über 2 wird auf 2 gesetzt, ein negativer Wert wirkt wie Bestand 0 ($V(0)=0$, wie bei $\max\{0,\cdot\}$), $V$ dazwischen linear interpoliert; die Iteration bricht ab, wenn sich $V$ zwischen zwei Durchläufen um weniger als 10⁻¹¹ ändert (höchstens 20 000 Durchläufe); Messrauschen und Kapazitätsgrenze kennt sie nicht. **Aufwandkosten** der Wertiteration (Regler „Kosten c“, Tab „Das Optimum“): Kosten $c\ln(B/S)$ je Jahr bei Preis 1 für die Entnahme von $B$ auf den Rest $S$ (bei $B>0$ ist $S=0$ ausgeschlossen); $c=0$ heißt ohne Kosten. **Kippstelle:** das Messrauschen, bei dem die gepaarte Differenz (Regel minus Aufwand) vom Positiven ins Nichtpositive wechselt, linear zwischen den Rasterpunkten (0 / 0,05 / 0,1 / 0,15 / 0,2 / 0,25 / 0,3 / 0,35 / 0,375 / 0,4 / 0,5 / 0,6 / 0,8) interpoliert; gezählt wird der erste Wechsel von positiv zu höchstens null, ein späterer zählt nicht (`data/kipp.csv`, `python fis_grid.py`, die Laufzeit hängt vom Rechner ab und liegt im Bereich von Minuten bei mehreren Prozessen, 13 Einstellungen mal 13 Werte des Messrauschens). Der Aufwand liest den Survey nicht: seine Spalten sind über das Messrauschen gleich, per Konstruktion und kein Fehler (Gegenprobe: jede Regel mit absichtlich schlechtem Reglerwert zeigt Kollaps, und im Grundfall des Rasters hat Escapement (frei abgestimmt) bei exaktem Survey 0 % und beim größten Messrauschen 100 % Kollaps).

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen sind Mittel über die genannten Pfade und gelten im Modell oben mit den Voreinstellungen (r = 0,5, σ_g = 0,2, σ_m = 0,2, β = 0,95, Kapazitätsgrenze 0,9, Kollapsgrenze 0,1 K), wenn nichts anderes steht. Ziel in K je Jahr, Kollaps = Anteil der 5 000 Testpfade. Die Zahlen der Tabellen prüft `tests/test_claims.py` als **Band** (nicht als Ziffern: Plattformen lösen Gleichstände der Abstimmung und die letzte Stelle verschieden auf), das Raster wird aus `data/kipp.csv` gelesen. Standardfehler des Ziels auf den 5 000 Testpfaden im Grundfall: 0,0000 bis 0,0003, gepaarte Differenzen ±0,0001; im Raster (`data/kipp.csv`) bis 0,0004, bei σ_g = 0,4 bis 0,0006 (Messung dieser README, die Bänder prüft der Test).

**Grundfall: alle Regeln, abgestimmt auf 1 000 Trainingspfaden, getestet auf 5 000 Testpfaden** (`test_grundfall_tabelle`)

| Regel | Abstimmung | Reglerwert | Ziel | Kollaps |
|---|---|---|---|---|
| Quote | frei | Q = 0,10 | 0,0839 | 98,6 % |
| Quote | mit Kollapsgrenze | Q = 0,06 | 0,0598 | 6,0 % |
| Aufwand | frei | e = 0,23 | 0,1303 | 10,8 % |
| Aufwand | mit Kollapsgrenze | e = 0,22 | 0,1302 | 4,7 % |
| Escapement-Schwelle | frei | S* = 0,47 | 0,1374 | 36,8 % |
| Escapement-Schwelle | mit Kollapsgrenze | S* = 0,63 | 0,1285 | 9,0 % |
| Gedämpfte Schwelle | frei (gleich mit Kollapsgrenze) | S* = 0,38, f = 0,7 | 0,1400 | 5,6 % |
| Quote am MSY (Lehrbuch) | – | Q = 0,125 | 0,0821 | 100,0 % |
| Aufwand am MSY (Lehrbuch) | – | e = 0,20 | 0,1281 | 0,9 % |
| Escapement bei K/2 (Lehrbuch) | – | S* = 0,50 | 0,1370 | 29,3 % |
| DP-Politik auf dem Messwert | – | – | 0,1373 | 41,8 % |

Referenz der Wertiteration bei vollem Wissen: 0,1448 (Schwelle 0,4525; `test_grundfall_referenz_und_anteile`). Anteil an dieser Referenz (aus den ungerundeten Werten; 0,1400 / 0,1448 gibt gerundet 96,7 %): gedämpfte Schwelle 96,6 %, Escapement frei 94,9 %, Aufwand mit Kollapsgrenze 89,9 %, Escapement mit Kollapsgrenze 88,7 %, Quote frei 57,9 %, Quote mit Kollapsgrenze 41,3 %. Das Ziel der Escapement-Schwelle bei exaktem Survey (0,1453, Zeile unten) liegt um 0,0005 über diesem exakten Wert: das liegt im Bereich der Streuung (Standardfehler der Testschätzung bis 0,0003; die Gegenprobe der Politik auf 4 000 Pfaden gibt 0,14507 ± 0,00028 gegen 0,14485), eine einzelne Ursache ist nicht gemessen. Standardfehler eines Kollapsanteils $p$ auf 5 000 Testpfaden: $\sqrt{p(1-p)/5\,000}$, bei 9,0 % etwa 0,4 Prozentpunkte (Rechnung dieser README, nicht im Test); dazu kommt die Streuung der Abstimmung auf den Trainingspfaden. Die Referenz gilt für **volle** Beobachtung; ein Optimum für den verrauschten Survey ist nicht gerechnet. Gepaarte Differenz Escapement minus Aufwand: +0,0071 bei freier Abstimmung, **−0,0017** mit Kollapsgrenze (Standardfehler 0,0001); gedämpfte Schwelle minus Aufwand +0,0097 frei und +0,0098 mit Kollapsgrenze (`test_grundfall_gepaarte_differenzen`).

| Frage | Befund | Test |
|---|---|---|
| Hängen Quote und Aufwand am Messrauschen? | nein, per Konstruktion: gleiches Ziel und gleicher Kollaps bei σ_m = 0 und 0,5 | `test_aufwand_und_quote_sind_ueber_das_messrauschen_konstant_per_konstruktion` |
| Exakter Survey (σ_m = 0, sonst Grundfall) | Escapement frei S* = 0,45: Ziel 0,1453, kein Kollaps (0,0 %); gedämpfte Schwelle f = 1,0: 0,1453; DP-Politik 0,1453; Aufwand frei 0,1303 mit 10,8 % Kollaps; Quote mit Kollapsgrenze 0,0598 | `test_exakter_survey_und_ohne_rauschen` |
| Ohne Rauschen (σ_g = σ_m = 0) | Quote am MSY 0,1250, Aufwand frei 0,1403, Escapement 0,1451, DP-Politik 0,1451, jeweils 0 % Kollaps | `test_exakter_survey_und_ohne_rauschen` |
| Quote am MSY (Q = 0,125) gegen das Wachstumsrauschen, 5 000 Testpfade | Kollaps 0,0 % bei σ_g = 0, **40,2 % bei 0,01**, 73,6 % bei 0,02, 96,7 % bei 0,05, 100,0 % bei 0,2 | `test_quote_am_msy_kollabiert_schon_bei_winzigem_rauschen` |
| Aufwand am MSY (e = 0,2), 5 000 Testpfade | Kollaps 0,0 % bei σ_g = 0, 0,05 und 0,1 (die drei gemessenen Werte) und 0,9 % bei 0,2; bei der Kollapsgrenze θ = 0,2 K (statt 0,1 K) aber 43 % bei σ_g = 0,2 | `test_aufwand_am_msy_ist_stabil_aber_die_kollapsgrenze_entscheidet` |
| Deterministische Bahnen (Quote 0,125 und Aufwand 0,2, 100 Jahre, r = 0,5) | Quote: Start 0,62 K endet bei 0,618 K, Starts 0,55 K, 0,45 K und 0,2 K enden praktisch bei 0, nach 100 Jahren unter 10⁻⁶ K (Kapazitätsgrenze 0,9); Aufwand: jeder dieser Starts endet bei 0,625 K | `test_deterministische_bahnen_quote_instabil_aufwand_kehrt_zurueck` |
| Escapement gegen DP-Politik bei exaktem Survey über σ_g = 0 / 0,05 / 0,1 / 0,2 / 0,3 / 0,4 | gleiches Ziel auf vier Stellen (Differenz höchstens 0,0003): 0,1451 / 0,1452 / 0,1453 / 0,1453 / 0,1443 / 0,1414 | `test_escapement_gleicht_bei_exaktem_survey_der_dp_politik_ueber_das_wachstumsrauschen` |
| Versteckt die Diskontierung den Kollaps? | die frei abgestimmte Quote (Q = 0,10) kollabiert in 98,6 % der Pfade und bringt 0,0839; bei β = 1 (Mittel der Jahre 50 bis 199) wird sie vorsichtiger: Q = 0,065, Kollaps 13,4 %, Ziel 0,0599 | `test_ziel_allein_versteckt_den_kollaps_und_beta_eins_macht_die_quote_vorsichtiger` |
| Kapazitätsgrenze 0,9 gegen keine (1,0), Escapement frei | 0,9: Ziel 0,1374, Kollaps 36,8 %, Aussterben 0,0 %; 1,0: Ziel 0,1322, Kollaps 57,4 %, Aussterben 32,5 % | `test_ohne_kapazitaetsgrenze_kann_escapement_den_bestand_aussterben_lassen` |
| Kollapsgrenze 0,05 / 0,1 / 0,2 K (feste Regler) | Aufwand e = 0,22: 0,1 / 4,7 / 74,0 %; Escapement S* = 0,63: 0,0 / 9,0 / 65,2 %; gedämpfte Schwelle 0,38 / 0,7: 0,1 / 5,6 / 44,8 %; Quote 0,06: 6,0 / 6,0 / 7,7 % | `test_kollapsgrenze_aendert_die_anteile_stark` |
| Wertiteration mit Aufwandkosten c = 0 / 0,02 / 0,05 / 0,1 / 0,2 (Gitter von 801 Punkten) | Schwelle 0,4525 / 0,4625 / 0,4775 / 0,5025 / 0,555; gleichwertiger Jahresfang 0,1448 / 0,1398 / 0,1324 / 0,1206 / 0,0988 (nur die Schwelle ist gerechnet, nicht die Reihenfolge der Regeln unter Kosten) | `test_kosten_heben_die_schwelle_der_wertiteration` |
| Wertiteration gegen die Formel $S^*=\frac12(1-\frac{1/\beta-1}{r})$, $\sigma_g=0$, r = 0,3 / 0,5 / 1,0, β = 0,9 / 0,95 / 0,98 | Schwelle weicht um höchstens 0,0005 ab (Gitter von 2 001 Punkten), $V(1)$ um höchstens 2·10⁻⁵ (zum Beispiel 2,90131 gegen 2,90132); Orakeltest zusätzlich mit Toleranz von einem Gitterschritt | `test_wertiteration_gegen_formel_in_neun_faellen` |
| Wertiteration gegen Simulation der Politik (σ_g = 0,1 / 0,2 / 0,4, 4 000 Pfade, Zufalls-Seed 777) | gleichwertiger Jahresfang (1 − β) V(1) der Wertiteration 0,14506 / 0,14485 / 0,14031, simuliert 0,14517 ± 0,00014 / 0,14507 ± 0,00028 / 0,14082 ± 0,00057; die Schwelle wächst mit σ_g (0,4475 / 0,4525 / 0,4775) und liegt ab σ_g = 0,2 über der Formel (0,4474); bei σ_g = 0,1 ist sie auf dem Gitter (Schritt 0,0025) von der Formel nicht zu unterscheiden | `test_wertiteration_gegen_simulation_der_politik` |
| Gitterfeinheit der Wertiteration (σ_g = 0,2): 401 / 801 / 1 601 Punkte | Schwelle 0,4500 / 0,4525 / 0,4512, (1 − β) V(1) = 0,14485 in allen; Gitterschritte 0,005 / 0,0025 / 0,00125: die Schwelle ist nur auf einen Gitterschritt genau und nicht monoton in der Feinheit, die vierte Stelle ist nicht belastbar | `test_gitterfeinheit_der_wertiteration` |
| Diskontfaktor 0,9 / 0,95 / 0,98 / 1,0: Escapement minus Aufwand (beide mit Kollapsgrenze) | −0,0019 / −0,0017 / +0,0003 / +0,0011; die gedämpfte Schwelle liegt bei allen vier um mehr als 0,007 vor dem Aufwand | `test_diskontfaktor_aendert_die_reihenfolge_von_escapement_und_aufwand_nur_knapp` |
| Wachstumsrate r = 1,5 | gedämpfte Schwelle 1,9 % unter dem Aufwand (Aufwand 0,3628), beide mit Kollapsgrenze | `test_bei_r_eins_komma_fuenf_liegt_die_gedaempfte_schwelle_hinter_dem_aufwand` |
| Überschießen ohne Entnahme bei r = 0,5 | σ_g = 0,4: 16,9 % der 5 000 Testpfade stürzen durch Überschießen auf 0; σ_g = 0,2: kein Pfad | `test_ueberschiessen_ohne_entnahme_bei_grossem_wachstumsrauschen` |
| Presets | Escapement bei exaktem Survey: 0,0 % Kollaps; verrauschter Survey (σ_m = 0,3, S* = 0,47): Ziel 0,1297, Kollaps 94,6 %; Schneller Bestand (r = 1, e = 0,35): Ziel 0,2466, 0,0 % Kollaps; Lehrbuch (Quote am MSY, σ_g = 0,01): Kollaps 40,2 % | `test_zahlen_der_presets`, `test_jedes_preset_zeigt_die_erwartete_zeile_der_ersten_kernfrage` |

**Wo kippt es? Kippstelle des Messrauschens** (aus `data/kipp.csv`, β = 0,95, 1 000 Trainings- und 5 000 Testpfade, Zufalls-Seeds 11 und 22; lineare Interpolation zwischen den 13 Rasterpunkten 0 bis 0,8; `test_kippstellen_des_rasters`). „kein“: kein Vorzeichenwechsel im Raster.

| Einstellung (sonst Grundfall) | Escapement gegen Aufwand, frei | Escapement gegen Aufwand, mit Kollapsgrenze | gedämpfte Schwelle gegen Aufwand, mit Kollapsgrenze |
|---|---|---|---|
| Grundfall | 0,292 | **0,193** | 0,379 |
| Kollapsgrenze 0,05 K | 0,292 | 0,292 | 0,419 |
| Kollapsgrenze 0,2 K | 0,292 | 0,153 | 0,401 |
| Kapazitätsgrenze 0,7 | 0,424 | 0,415 | 0,452 |
| Kapazitätsgrenze 1,0 (keine) | 0,208 | 0,156 | 0,377 |
| σ_g = 0,05 | 0,171 | 0,158 | 0,255 |
| σ_g = 0,1 | 0,199 | 0,167 | 0,301 |
| σ_g = 0,3 | 0,453 | 0,247 | 0,526 |
| σ_g = 0,4 | 0,703 | kein (Aufwand nicht zulässig) | kein |
| r = 0,2 | 0,746 | kein | 0,688 |
| r = 0,3 | 0,466 | 0,222 | 0,534 |
| r = 0,8 | 0,206 | 0,195 | 0,279 |
| r = 1,0 | 0,176 | 0,176 | 0,233 |

- **Mit Kollapsgrenze liegt die Kippstelle in den sieben Einstellungen mit Vorzeichenwechsel unter den neun, die nur r oder σ_g ändern** (Grundfall und je vier Werte von σ_g und r) (Grundfall, σ_g = 0,05 / 0,1 / 0,3, r = 0,3 / 0,8 / 1,0) **zwischen 0,158 und 0,247**; über alle 11 Einstellungen mit Vorzeichenwechsel zwischen 0,153 und 0,415 (kein Wechsel: σ_g = 0,4, dort ist der Aufwand nicht zulässig, und r = 0,2; in den übrigen 12 von 13 Einstellungen ist der Aufwand zulässig). Ohne Kollapsgrenze hat jede der 13 Einstellungen einen Vorzeichenwechsel, zwischen 0,171 und 0,746 (`test_mit_kollapsgrenze_liegt_die_kippstelle_in_sieben_einstellungen_zwischen_016_und_025`).
- **Im Grundfall** (`test_tabelle_der_differenzen_im_grundfall`): Escapement minus Aufwand frei +0,0151 bei σ_m = 0, +0,0071 bei 0,2, −0,0006 bei 0,3 (dort aber 94,6 % Kollaps: der gleich große Mittelwert ist mit fast sicherem Kollaps erkauft); mit Kollapsgrenze −0,0017 bei 0,2 und −0,0534 bei 0,3, wo die Abstimmung die Kollapsgrenze nicht mehr einhalten kann (nicht zulässig bei 0,3 und 0,4). Die gedämpfte Schwelle (mit Kollapsgrenze) bringt 0,1400 bei σ_m = 0,2 und 0,1354 bei 0,3, bleibt bis σ_m = 0,8 bei höchstens 11 % Kollaps und liegt +0,0052 (0,3), +0,0010 (0,375) und −0,0047 (0,4) vom Aufwand entfernt.
- **Die Kippstelle hängt an den Definitionen:** Kollapsgrenze 0,05 / 0,1 / 0,2 K gibt 0,292 / 0,193 / 0,153; Kapazitätsgrenze 1,0 gibt 0,156, 0,9 gibt 0,193, **bei 0,7 gibt es bis σ_m = 0,4 keinen Vorzeichenwechsel** (Differenz bis dahin positiv, +0,0009 bei 0,4) und im Raster bis 0,8 einen bei 0,415 (`test_aufwand_ist_bei_sigma_g_vier_zehntel_nicht_zulaessig_und_bei_r_zwei_zehntel_ist_die_schwelle_vorn`). Die Kippstelle ist also keine feste Zahl: eine strengere Kapazitätsgrenze (0,7: 0,415) oder die Dämpfung f < 1 (gedämpfte Schwelle: 0,379 gegen 0,193 bei Escapement) schiebt sie nach oben, ohne sie aufzuheben. Eine naheliegende, nicht gemessene Deutung: ein zu hoch gemessener Bestand führt zu Überentnahme.
- **σ_g = 0,4 und r = 0,2:** bei σ_g = 0,4 ist kein Aufwand zulässig (Kollaps über 10 % schon ohne Fang; das Überschießen allein trifft ohne Entnahme 16,9 % der Pfade, mehr als die Schranke von 10 %, siehe unten), bei r = 0,2 liegt Escapement mit Kollapsgrenze bis σ_m = 0,8 vorn; der Aufwand muss dort auf e = 0,06 sinken (vermutlich der Grund, nicht einzeln gemessen).
- **Der Vorsprung des Surveys fällt mit der Wachstumsrate:** gedämpfte Schwelle gegen Aufwand (beide mit Kollapsgrenze, σ_m = 0,2) +70 % bei r = 0,2, +23 % bei 0,3, +7,5 % bei 0,5, +3,2 % bei 0,8, +1,4 % bei 1,0, −1,9 % bei 1,5 (`test_vorsprung_der_gedaempften_schwelle_gegen_den_aufwand_faellt_mit_der_wachstumsrate`, r = 1,5 in `test_bei_r_eins_komma_fuenf_liegt_die_gedaempfte_schwelle_hinter_dem_aufwand`; Deutung offen, bei r über 1 ist die Wachstumsfunktion nicht monoton).
- **Bei hohem Wachstumsrauschen wird der feste Aufwand teuer** (σ_m = 0,2, mit Kollapsgrenze; Aufwand / Escapement / gedämpfte Schwelle): σ_g = 0,05: 0,1398 / 0,1321 / 0,1413; 0,1: 0,1379 / 0,1315 / 0,1410; 0,2: 0,1302 / 0,1285 / 0,1400; 0,3: 0,0946 / 0,1233 / 0,1381 (der Aufwand muss auf e = 0,13 sinken; `test_wachstumsrauschen_vergroessert_den_vorsprung_der_schwelle_vor_dem_aufwand`).
- **Überschießen im Raster** (`test_raster_ueberschiessen_nur_bei_sigma_g_vier_zehntel_nennenswert`): außer bei σ_g = 0,4 stürzen in den abgestimmten Regeln höchstens 0,5 % der Testpfade durch Überschießen auf 0; bei σ_g = 0,4 bis 17 % (der Aufwand ohne Fang) und bis 10 % (gedämpfte Schwelle bei σ_m = 0,8).

**Handbeispiel** (exakte Brüche, r = 1/2, K = 1, Kapazitätsgrenze 9/10, ohne Zufall; `test_handbeispiel_zahlen_der_readme`, `tests/test_hand.py`): Quote 1/8 von B_0 = 1 gibt B_1 = 119/128, von B_0 = 11/20 gibt sie B_1 = 1751/3200 (unter 0,55: der Bestand fällt, nach 100 Jahren unter 10⁻⁶ K); Aufwand 1/5 von 11/20 gibt B_1 = 352/625 und kehrt auf das Gleichgewicht 5/8 mit Fang 1/8 zurück. Optimum bei β = 19/20: S* = 17/38 und V(1) = 441/152 (gleichwertiger Jahresfang 441/3040 = 0,14507); die Wertiteration findet dasselbe auf einen Gitterschritt (Orakeltest).

**Messungen ohne Test** (Messungen dieser README, nicht im Test): Die Kippstelle im Grundfall mit anderen Zufalls-Seed-Paaren (Training / Test): 11 / 22 frei 0,292 und mit Kollapsgrenze 0,193; 41 / 52: 0,295 und 0,193; 61 / 72: 0,296 und 0,197. Feineres Gitter der gedämpften Schwelle (S* in Schritten von 0,01, f in 0,05; Grundfall, mit Kollapsgrenze, Training Zufalls-Seed 11, Test 5 000 Pfade) gegen das Gitter der Demo (0,02 und 0,1): Ziel 0,1401 gegen 0,1400 bei σ_m = 0,2 (Kollaps 2,8 % gegen 5,6 %), gleich bei 0,3 (0,1354) und 0,375 (0,1312), 0,1286 gegen 0,1255 bei 0,4 (Differenz zum Aufwand −0,0016 statt −0,0047). Die Kippstelle der gedämpften Schwelle gegen den Aufwand liegt damit bei etwa 0,385 statt 0,379 (lineare Interpolation zwischen 0,375 und 0,4); das grobe Gitter ändert den Vorsprung bis σ_m = 0,375 nicht, und 0,1400 im Grundfall ist eine untere Schranke des besten Wertes mit zwei Reglern (bis auf den Standardfehler der Testschätzung; das feinere Gitter bestätigt die Richtung: 0,1401 gegen 0,1400).

## Ehrliche Grenzen

- **Alle Zahlen sind Annahmen, keine Messdaten:** Wachstumsrate r (Voreinstellung 0,5; Regler 0,2 bis 1,5), Wachstumsrauschen σ_g (0,2; 0 bis 0,4), Messrauschen σ_m (0,2; 0 bis 1,0), Diskontfaktor β (0,95; zur Wahl 0,9 / 0,95 / 0,98 / 1,0), Kapazitätsgrenze h_max (0,9; 0,5 bis 1,0), Kollapsgrenze (0,1 K; 0,05 bis 0,2 K), Anfangsbestand B_0 = K, 200 Jahre je Pfad. Es wurde kein Fisch- und kein Wildbestand gemessen oder nachgeschlagen; keine Zahl dieser Demo gilt für einen echten Bestand, und die Demo gibt keine Empfehlung für eine Bewirtschaftung.
- **Das Modell ist ein Wachstumsmodell vom Schaefer-Typ** (logistisches Wachstum, Entnahme vor dem Wachstum). Die Namen Schaefer und Reed sowie die Aussage, die Schwellenregel mit Dämpfung ähnle den „Hockeyschläger“-Fangregeln der Fischereiberatung, stammen aus dem Gedächtnis und sind **nicht geprüft**; es wurde keine Fachquelle gelesen. Das Lehrbuchwissen (Quote r K / 4, Aufwand r / (2 + r), Schwelle K / 2) wird hier gemessen, nicht übernommen. Für r ≤ 1 ist die Wachstumsfunktion auf [0, 1] monoton; bei r über 1 ist sie es nicht, die Demo zeigt dort Beobachtungen ohne Deutung.
- **Das Rauschen ist lognormal mit Mittel 1 und unabhängig** über Jahre und Pfade; der Messfehler des Surveys ist unverzerrt und unabhängig vom Wachstumsrauschen. Verzerrung und zeitliche Korrelation des Messfehlers sind nicht untersucht, ebenso wenig Altersstruktur, Allee-Effekt (der Kollaps ist hier nicht absorbierend, solange der Rest positiv bleibt), Zeitverzug zwischen Entscheidung und Fang, mehrere Arten, Rückwürfe und Pfade über 200 Jahre.
- **Das Ziel ist eine Menge, kein Geld:** der gleichwertige Jahresfang (in K je Jahr), bei β = 0,95 mit Start B_0 = K ab dem ersten Jahr diskontiert; bei β = 1 das Mittel der Jahre 50 bis 199. Preis und Kosten gibt es nur in der Wertiteration (Kostenregler, Tab „Das Optimum“), nicht in den simulierten Regeln; mit Aufwandkosten ist der gleichwertige Jahresfang der Fang minus die Kosten bei Preis 1, also ein Wert in Einheiten des Fangs und keine reine Menge. Wegen der Diskontierung ist ein Kollaps in ferner Zukunft für das Ziel billig: das Ziel allein sagt nichts über die Sicherheit, deshalb steht der Kollapsanteil neben jedem Ziel.
- **Kollaps** heißt: der Bestand fällt irgendwann in den Jahren 1 bis 200 unter die Kollapsgrenze (Voreinstellung 0,1 K); **Aussterben** heißt: der Bestand fällt irgendwann in den Jahren 1 bis 200 unter 10⁻⁴ K. Die Kippstelle (Tab „Wo kippt es?“) hängt an dieser Definition, an der Kapazitätsgrenze und an der Abstimmungsregel; sie ist deshalb als Raster über diese Einstellungen gezeigt, nicht als eine Zahl.
- **Die Abstimmung ist ein Gitter auf getrennten Trainingspfaden** (Voreinstellung 1 000 Pfade, Zufalls-Seed 11); bewertet wird auf frischen Testpfaden (5 000 Pfade, Zufalls-Seed 22). Alle Regeln sehen dieselben Zufallszahlen, Differenzen sind gepaart. „Frei“ wählt das größte mittlere Ziel; „mit Kollapsgrenze“ das größte Ziel unter der Bedingung, dass höchstens 10 % der Trainingspfade kollabieren (ist nichts zulässig, gilt das kleinste Kollapsrisiko, die Zeile heißt „nicht zulässig“). Die gedämpfte Schwelle hat zwei Regler auf einem groben Gitter (S* in Schritten von 0,02, f in Schritten von 0,1): ihr Ziel ist eine untere Schranke des besten Wertes mit zwei Reglern (bis auf den Standardfehler der Testschätzung).
- **Das Optimum der Wertiteration gilt für volles Wissen** über den Bestand (kein Messrauschen, keine Kapazitätsgrenze). Ein Optimum für den verrauschten Survey (Bayes-Filter über den Bestand) ist **nicht** gerechnet: die gedämpfte Schwelle ist deshalb „besser als die einfachen Regeln“, nicht „optimal“, und die Kippstelle gegen dieses unbekannte Optimum kann anders liegen. Die „DP-Politik auf dem Messwert“ wendet die Schwelle des vollen Wissens auf den verrauschten Messwert an.
- **Zufall und Genauigkeit:** alle Zufallsgrößen kommen aus festen Zufalls-Seeds; Mittelwerte stehen mit ihrem Standardfehler. Neu gerechnete Zahlen können sich plattformabhängig in der letzten Stelle unterscheiden, die Tests prüfen deshalb Bänder.

## Befunde und Korrekturen gegenüber der Vormessung

Die Vormessung (`fischerei-planung/vormessung/`, eine unabhängig geschriebene Fassung desselben Modells) und diese Demo stimmen im Grundfall auf vier Stellen und in den Kippstellen bis auf 0,001 überein (Vergleich beim Bau, nicht im Test; die Abweichung betrifft σ_g = 0,4 ohne Kollapsgrenze: 0,704 in der Vormessung gegen 0,703 hier, vermutlich wegen der Begrenzung des Überschießens). Abweichungen:

- **Überschießen:** die Wachstumsfunktion wird für einen Rest $S$ über $(1+1/r)K$ negativ. Die Vormessung ließ negative Bestände zu (sie zählten als Kollaps, und eine Aufwandsregel hätte dann negativ „gefangen“); beim Bau fiel das in der Zählung auf: ohne Entnahme sind bei r = 0,5 und σ_g = 0,4 17 % der Testpfade betroffen, bei r = 1,5 und σ_g = 0,2 gut die Hälfte (2 605 von 5 000 Pfaden; Messung mit der Fassung ohne Begrenzung beim Bau, nicht im Test). Die Demo setzt den Bestand dann auf 0 und weist den Anteil dieser Pfade aus. In den Einstellungen der Befunde ändert das die Zahlen nicht (höchstens 0,5 % der Pfade, außer σ_g = 0,4); **die Zeile σ_g = 0,4 des Rasters („Aufwand nicht zulässig“) ist wesentlich ein Merkmal der Wachstumsfunktion**, nicht ein Ergebnis der Regel: allein das Überschießen ohne Entnahme (16,9 %) übersteigt die Schranke von 10 %; ob ohne Überschießen ein Aufwand zulässig wäre, ist nicht gemessen. Bei r = 1,5 (nicht im Raster) stürzen ohne Entnahme 52 % der Testpfade durch Überschießen auf 0 (2 605 von 5 000), mit dem abgestimmten Aufwand und der abgestimmten gedämpften Schwelle (beide mit Kollapsgrenze) keiner (`test_bei_r_eins_komma_fuenf_liegt_die_gedaempfte_schwelle_hinter_dem_aufwand`).
- **Mittlerer Bestand und Fang-Streuung** zählen die Jahre 50 bis 199 mit dem Bestand vor der Entnahme (die Vormessung nahm den Bestand nach dem Wachstum); die Tabelle der README nennt beide Größen nicht.
- **Kostenvariante** der Wertiteration mit dem Gitter von 801 statt 401 Punkten: Schwellen 0,4625 / 0,4775 / 0,5025 / 0,555 statt 0,460 / 0,480 / 0,505 / 0,555 bei c = 0,02 / 0,05 / 0,1 / 0,2 (ein Gitterschritt Unterschied).
- **Raster und Kapazitätsgrenze 0,7:** die Vormessung rechnete bis σ_m = 0,4 (kein Vorzeichenwechsel); das Raster reicht bis 0,8 und zeigt einen bei 0,415.
- **Kollapsanteil bei der doppelten Grenze** (die Vormessung zeigte eine zweite Spalte „K20“) ist ersetzt durch den Regler Kollapsgrenze und die Kollapsgrenzen-Zeilen des Rasters.
- **Offene Punkte der Vormessung:** (a) Bayes-Filter als Referenz für verrauschten Survey: nicht gerechnet (siehe „Bewusst nicht umgesetzt“); (b) feineres Gitter der gedämpften Schwelle: gemessen (Abschnitt „Messungen ohne Test“); (c) Quellenlage der Fangregeln: nicht geprüft, die Demo sagt nichts über „die Praxis“.

## Verifikation

- **Kern gegen Handrechnung:** Messung, Wachstum, jede Entnahmeregel an Zahlen von Hand, Gewichte des Ziels, Mini-Instanz ohne Zufall gegen exakte Brüche (`tests/test_algorithm.py`, `tests/test_hand.py`); Abstimmung (Auswahl des besten Reglers, Grenze inklusive, kein zulässiger Wert) und gepaarte Differenz von Hand.
- **Randfälle und Definitionen:** Entnahme vor Wachstum (r = 1, e = 1/2 gibt 3/4), Kollaps strikt unter der Grenze genau auf dem Wert, Anfangsbestand zählt nicht, Aussterben, Reglerwert 0 gibt das reine Wachstum, Kapazitätsgrenze, Überschießen, exakter Survey, β = 1 (`tests/test_edge_cases.py`).
- **Orakel:** `tests/test_oracle_fischerei_entnahme.py` (50 Fälle): Wertiteration gegen die geschlossene Formel in neun Fällen, Gleichgewicht des Aufwands, Quote am MSY gegen exakte Brüche, Eigenschaftstests (Entnahme ≤ h_max B, B ≥ 0, Mittel der Rauschfaktoren 1, Unabhängigkeit), DP-Politik simuliert gegen den Wert (3 Standardfehler), Escapement gegen die Politik, gemeinsame Zufallszahlen, Reihenfolge der Kippstelle, **Vollaufzählung aller 256 Vorzeichenpfade einer 4-Jahres-Instanz gegen eine Schleife in reinem Python** (12 Kombinationen aus Regel und Szenario, auch β = 1), Bellman-Gleichung mit anderer Quadratur und feinem Suchgitter (scipy), Momente der Gauß-Hermite-Knoten.
- **Verdikte** als reine Funktionen mit jeder Schwelle inklusive der Grenze und jeder Zahl im Text (`tests/test_verdicts.py`); je Preset die erwartete Zeile der ersten Kernfrage (`tests/test_app_content.py`).
- **Behauptungen:** jede Zahl der Tabellen oben als Band (`tests/test_claims.py`); die README behauptet nicht, dass jede Ziffer in den Tests steht.
- **Presets, Permalink, Layout:** vollständig, Werte im Reglerbereich und auf dem Raster der Regler, einzigartig; Permalink begrenzt und rastet ein; jeder Regler der Seitenleiste in `SETTING_SPECS`, jede Änderung hebt die Markierung auf (ausgegraute Regler über den Session State, weil auch ein Nutzer sie nicht bedienen kann).
- **Oberfläche:** AppTest-Rauchtest (jeder Knopf, jeder Regler an Minimum und Maximum), Footer, Formelblöcke (15, je einer pro `st.markdown` im Quelltext, kein ungeschütztes Prozentzeichen), Laufzeit-Importe.
- **Nicht geprüft:** ein Durchlauf im echten Browser (Layout, Plotly-Diagramme, KaTeX-Formeln); der Browser des Arbeitsplatzes war für diesen Bau nicht erreichbar.
- **Testzahl:** 272 Tests (`pytest --collect-only -q tests`).

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore` | venv, Caches, Streamlit-Secrets |
| `.streamlit/config.toml` | Streamlit-Einstellungen (keine Nutzungsstatistik) |
| `app.py` | Streamlit-App: sechs Tabs, Kernfragen, Expander, PDF-Export |
| `data/kipp.csv` | vorgerechnetes Raster „Wo kippt es?“ (`python fis_grid.py`) |
| `fis_algorithm.py` | Kern: Messung, Entnahmeregeln, Wachstum, Simulation, Abstimmung, Bewertung |
| `fis_constants.py` | Regler-Grenzen, feste Annahmen, Gitter, Raster-Einstellungen, Presets |
| `fis_dp.py` | Wertiteration (mit Aufwandkosten), geschlossene Formel |
| `fis_evaluation.py` | Pfadbündel, Vergleich der Regeln, Kippstelle, Verdikte |
| `fis_grid.py` | Raster „Wo kippt es?“ bauen und lesen |
| `fis_hand.py` | Handbeispiel in exakten Brüchen |
| `fis_layout.py` | Fester Kopf (Stylesheet) |
| `fis_pdf_export.py` | PDF-Export |
| `fis_presets.py` | Permalink (Begrenzen, Einrasten), Presets (Schnellstart in der Seitenleiste, aktives Preset hervorgehoben), Zufalls-Seed-Knopf |
| `fis_scenario.py` | Szenario (Modellgrößen) und Zufallszahlen der Pfade |
| `fis_texts.py` | Texte, die App und README gemeinsam führen |
| `fis_visualization.py` | Plotly-Abbildungen (Achsen gesperrt) |
| `requirements-dev.txt` | Laufzeit plus pytest und scipy (Gegenproben) |
| `requirements.txt` | Laufzeit-Bibliotheken (ungepinnt) |
| `tests/` | Kern, Hand, Randfälle, Orakel, Behauptungen, Verdikte, Presets, Raster, Layout, AppTest, Footer, Laufzeit-Importe |

## Bewusst nicht umgesetzt

- **Ein Optimum für den verrauschten Survey** (Bayes-Filter über den Bestand): würde zeigen, ob die gedämpfte Schwelle nahe am Optimum liegt und wo die Kippstelle gegen dieses Optimum liegt; nicht gerechnet, die Aussagen über die gedämpfte Schwelle sagen darum nur „besser als die einfachen Regeln“.
- **Verzerrter und zeitlich korrelierter Messfehler, Altersstruktur, Allee-Effekt, Zeitverzug, mehrere Arten, Rückwürfe, Preise und Kosten in den simulierten Regeln:** nicht modelliert und nicht gemessen; Kosten gibt es nur in der Wertiteration (Schwellenhöhe).
- **Quellen:** Schaefer, Reed und die Fangregeln der Fischereiberatung sind nur dem Namen nach aus dem Gedächtnis genannt, nicht gelesen und nicht geprüft; die Demo behauptet nichts über die Praxis.
- **Ein Regler für die Kollapsgrenze der Abstimmung** (fest 10 % der Trainingspfade) und ein frei wählbares Raster (das Raster ist vorgerechnet, 13 Einstellungen).

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Gebaut mit Streamlit, Plotly, NumPy und fpdf2; die Tests nutzen zusätzlich scipy (Gegenprobe der Bellman-Gleichung).

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html).
