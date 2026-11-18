"""Verdikte an von Hand gesetzten Werten (DEMO-PLAYBOOK §12): jede Schwelle (inklusive Grenze) und jede Zahl im Text steht hier. Ein Verdikt ist eine reine Funktion
(Art, Text) in fis_evaluation.py und bekommt Zeilen, nicht Seeds."""

import pytest

import fis_evaluation as ev


def _row(rule, mode, obj=0.1, collapse=0.0, feasible=True, name=None):
    return {"rule": rule, "mode": mode, "name": name or ev.LABELS[rule], "obj": obj, "collapse": collapse, "feasible": feasible}


def _cmp(rows, d_free=0.0, d_cap=0.0, se_cap=0.0001, cap=0.10, n_test=5000):
    return {"rows": rows, "cap": cap, "n_test": n_test, "diffs": {("escape", "frei"): (d_free, 0.0001), ("escape", "kollaps"): (d_cap, se_cap)}}


# ---------------------------------------------------------------------------------------------------------------- Kernfrage 1: Kollaps der Regel
@pytest.mark.parametrize(
    "collapse, kind",
    [(0.0, "success"), (0.10, "success"), (0.1001, "info"), (0.5, "info"), (0.5001, "warning"), (1.0, "warning")],
)
def test_verdict_collapse_schwellen_inklusive_der_grenzen(collapse, kind):
    assert ev.verdict_collapse(collapse, 0.1, 500, "Aufwand")[0] == kind


def test_verdict_collapse_nennt_regel_anteil_pfadzahl_und_grenze():
    kind, text = ev.verdict_collapse(0.07, 0.15, 500, "Gedämpfte Schwelle")
    assert kind == "success"
    assert "„Gedämpfte Schwelle“" in text and "**7.0 %**" in text and "500 Pfade" in text and "unter 0.15 K" in text and "höchstens 10 %" in text
    assert "mehr als die 10 %" in ev.verdict_collapse(0.2, 0.1, 500, "Quote")[1]
    assert "die Mehrheit der Pfade" in ev.verdict_collapse(0.9, 0.1, 500, "Quote")[1]
    assert "höchstens 20 %" in ev.verdict_collapse(0.15, 0.1, 500, "Quote", cap=0.2)[1]


# ---------------------------------------------------------------------------------------------------------------- Kernfrage 2: Ziel versteckt Kollaps
def test_verdict_hidden_alle_innerhalb_der_grenze():
    rows = [_row("quote", "frei", collapse=0.10), _row("aufwand", "frei", collapse=0.0), _row("escape", "frei", collapse=0.05), _row("schwelle", "frei", collapse=0.1)]
    kind, text = ev.verdict_hidden(_cmp(rows))
    assert kind == "success" and "Alle vier" in text and "höchstens 10 %" in text and "ohne Nebenbedingung" in text and "kostet" not in text      # genau 10 % gilt als innerhalb


def test_verdict_hidden_nennt_jede_regel_ueber_der_grenze_mit_zahlen():
    rows = [_row("quote", "frei", obj=0.0839, collapse=0.986), _row("aufwand", "frei", obj=0.1303, collapse=0.1001), _row("escape", "frei", collapse=0.0), _row("schwelle", "frei", collapse=0.0),
            _row("quote", "kollaps", collapse=0.9), _row("escape", "lehrbuch", collapse=0.9)]       # nur die freien Zeilen zählen
    kind, text = ev.verdict_hidden(_cmp(rows))
    assert kind == "warning" and text.startswith("2 von 4")
    assert "Quote 98.6 % (Ziel 0.0839)" in text and "Aufwand 10.0 % (Ziel 0.1303)" in text and "Escapement" not in text


# ---------------------------------------------------------------------------------------------------------------- Kernfrage 3: Reihenfolge
def _order_rows(feas_esc=True, feas_auf=True):
    return [_row("escape", "kollaps", feasible=feas_esc), _row("aufwand", "kollaps", feasible=feas_auf)]


@pytest.mark.parametrize(
    "d, se, kind",
    [(0.0002, 0.0001, "info"), (-0.0002, 0.0001, "info"), (0.00021, 0.0001, "success"), (-0.00021, 0.0001, "warning"), (0.0, 0.0, "info"), (0.0001, 0.0, "success"), (-0.0001, 0.0, "warning")],
)
def test_verdict_order_schwelle_bei_zwei_standardfehlern(d, se, kind):
    assert ev.verdict_order(_cmp(_order_rows(), d_cap=d, se_cap=se), 0.2)[0] == kind


def test_verdict_order_nennt_differenz_standardfehler_pfade_und_nicht_erreichbares():
    kind, text = ev.verdict_order(_cmp(_order_rows(), d_free=0.0071, d_cap=-0.0017, se_cap=0.0001), 0.2)
    assert kind == "warning" and "Messrauschen 0.2" in text and "Escapement minus Aufwand -0.0017" in text and "ohne Kollapsgrenze: +0.0071" in text
    assert "Standardfehler der Differenz 0.0001" in text and "5000 Testpfade" in text and "nicht erreichbar" not in text
    _, t2 = ev.verdict_order(_cmp(_order_rows(feas_esc=False), d_cap=0.01), 0.3)
    assert "für die Escapement-Schwelle nicht erreichbar" in t2 and "Messrauschen 0.3" in t2
    _, t3 = ev.verdict_order(_cmp(_order_rows(feas_esc=False, feas_auf=False), d_cap=0.01), 0.3)
    assert "für die Escapement-Schwelle und den Aufwand nicht erreichbar" in t3
    assert "vorn" in ev.verdict_order(_cmp(_order_rows(), d_cap=0.01), 0.1)[1] and "gleichauf" in ev.verdict_order(_cmp(_order_rows(), d_cap=0.0), 0.1)[1]


# ---------------------------------------------------------------------------------------------------------------- Tab 3 und 4
def test_kipp_linear_zwischen_den_rasterpunkten_von_hand():
    assert ev.kipp([0.0, 1.0, 2.0], [2.0, 1.0, -1.0]) == pytest.approx(1.5)          # 1 + 1 * 1 / (1 + 1)
    assert ev.kipp([0.0, 0.1, 0.2], [0.3, 0.1, 0.0]) == pytest.approx(0.2)           # Differenz genau 0 zählt als Wechsel (d1 <= 0): Nullstelle bei 0,2
    assert ev.kipp([0.0, 0.1], [0.3, 0.1]) is None                                    # nie kleiner als null
    assert ev.kipp([0.0, 0.1], [-0.1, -0.2]) is None                                  # schon bei 0 hinten: kein Wechsel von + nach -
    assert ev.kipp([0.0, 0.1, 0.2, 0.3], [0.2, -0.1, 0.4, -0.2]) == pytest.approx(0.1 * 0.2 / 0.3)      # der erste Wechsel zählt


def test_verdict_kipp_nennt_die_stelle_oder_kein_kipp():
    kind, text = ev.verdict_kipp(0.1934, 0.8, "Escapement-Schwelle gegen Aufwand, mit Kollapsgrenze")
    assert kind == "info" and "**0.19**" in text and text.startswith("Escapement-Schwelle gegen Aufwand, mit Kollapsgrenze:")
    kind, text = ev.verdict_kipp(None, 0.8, "frei abgestimmt")
    assert "kein Vorzeichenwechsel" in text and "bis Messrauschen 0.8" in text


def test_verdict_dp_gitterschritt_inklusive_und_mit_rauschen():
    kind, text = ev.verdict_dp(0.0, 0.45, 0.4475, 0.0025)
    assert kind == "success" and "höchstens einen Gitterschritt (0.0025)" in text
    kind, text = ev.verdict_dp(0.0, 0.4501, 0.4475, 0.0025)
    assert kind == "warning" and "um mehr als einen Gitterschritt" in text
    kind, text = ev.verdict_dp(0.2, 0.4525, 0.4474, 0.0025)
    assert kind == "info" and "Wachstumsrauschen 0.20" in text and "+0.0051" in text and "0.4525" in text and "0.4474" in text
    kind, text = ev.verdict_dp(0.0, 0.1, None, 0.0025)
    assert kind == "info" and "gilt nur für 0 < 1/β − 1 < r" in text
