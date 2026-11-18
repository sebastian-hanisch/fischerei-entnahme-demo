"""Randfälle und Definitionen, an denen Modelle dieser Art erfahrungsgemäß brechen (DEMO-PLAYBOOK §12; Lehre der Anbaupause in der Fruchtfolge-Demo: jede Definition
mit einem Randfall prüfen). Hier: Reihenfolge im Zeitschritt (Entnahme vor Wachstum), Kollapsgrenze genau auf dem Wert, Reglerwert 0, Kapazitätsgrenze, Überschießen."""

import numpy as np
import pytest

import fis_algorithm as A
import fis_constants as C
import fis_dp as D
from fis_scenario import Scenario, noise


def _zeros(years, paths=1):
    return np.zeros((years, paths)), np.zeros((years, paths))


def test_entnahme_kommt_vor_dem_wachstum():
    """r = 1, Aufwand e = 1/2, B_0 = 1: erst Entnahme (S = 1/2), dann Wachstum 1/2 + 1/4 = 3/4. Wüchse der Bestand zuerst (bei K gar nicht) und würde danach entnommen, käme 1/2 heraus."""
    sc = Scenario(r=1.0, sg=0.0, sm=0.0, beta=0.95, hmax=1.0)
    res = A.simulate("aufwand", [0.5], sc, _zeros(2), record=True)
    assert res["H"][0, 0, 0] == pytest.approx(0.5) and res["B"][1, 0, 0] == pytest.approx(0.75)


def test_kollaps_ist_strikt_unter_der_grenze_und_zaehlt_die_jahre_1_bis_t(monkeypatch):
    sc0 = Scenario(r=0.5, sg=0.0, sm=0.0, hmax=1.0, thr=0.1)
    b1 = float(A.simulate("aufwand", [0.5], sc0, _zeros(1), record=True)["B"][1, 0, 0])     # B_1 = 0,5 + 0,5 * 0,5 * 0,5 = 0,625
    assert b1 == pytest.approx(0.625)
    exact = Scenario(r=0.5, sg=0.0, sm=0.0, hmax=1.0, thr=b1)           # Grenze genau auf B_1: nicht unter der Grenze
    above = Scenario(r=0.5, sg=0.0, sm=0.0, hmax=1.0, thr=np.nextafter(b1, 1.0))
    assert A.simulate("aufwand", [0.5], exact, _zeros(1))["collapse"][0, 0] == 0.0
    assert A.simulate("aufwand", [0.5], above, _zeros(1))["collapse"][0, 0] == 1.0
    # der Anfangsbestand B_0 zählt nicht mit: B_0 = 0,3 liegt unter der Grenze 0,35, B_1 = 0,3 + 0,5 * 0,3 * 0,7 = 0,405 darüber, also kein Kollaps
    monkeypatch.setattr(C, "B0", 0.3)
    low_start = Scenario(r=0.5, sg=0.0, sm=0.0, hmax=1.0, thr=0.35)
    assert A.simulate("quote", [0.0], low_start, _zeros(1))["collapse"][0, 0] == 0.0


def test_aussterben_grenze_ist_strikt():
    sc = Scenario(r=0.5, sg=0.0, sm=0.0, hmax=0.9, thr=0.1)
    # Quote 10 (nur durch h_max begrenzt): S = 0,1 B, der Bestand fällt jedes Jahr auf rund 0,1 f(...) und unterschreitet 1e-4 K nach einigen Jahren
    res = A.simulate("quote", [10.0], sc, _zeros(10))
    assert res["extinct"][0, 0] == 1.0 and res["collapse"][0, 0] == 1.0


@pytest.mark.parametrize("rule, par, par2", [("quote", 0.0, None), ("aufwand", 0.0, None), ("schwelle", 0.4, 0.0)])
def test_reglerwert_null_gibt_keinen_fang_und_das_reine_wachstum(grundfall, rule, par, par2):
    nz = noise(9, 40, 20)
    res = A.simulate(rule, [par], grundfall, nz, par2=None if par2 is None else [par2], record=True)
    assert np.all(res["H"] == 0.0) and np.all(res["obj"] == 0.0)
    B = np.ones(20)
    for t in range(40):                                                   # reine Wachstumsbahn: dieselben Zufallszahlen, Schritt für Schritt von Hand
        B = np.exp(grundfall.sg * nz[0][t] - 0.5 * grundfall.sg ** 2) * (B + grundfall.r * B * (1 - B))
        assert res["B"][t + 1, 0] == pytest.approx(B, rel=1e-12)


def test_kapazitaetsgrenze_haelt_den_rest_positiv():
    sc = Scenario(r=0.5, sg=0.2, sm=0.2, hmax=0.9)
    res = A.simulate("quote", [10.0], sc, noise(2, 60, 30), record=True)    # die Quote will mehr, als der Bestand hat
    assert np.all(res["H"] <= 0.9 * res["B"][:-1] + 1e-15)
    assert np.all(res["B"] >= 0.0) and res["B"][-1].max() > 0.0
    full = A.simulate("quote", [10.0], Scenario(r=0.5, sg=0.0, sm=0.0, hmax=1.0), _zeros(3), record=True)
    assert full["H"][0, 0, 0] == pytest.approx(1.0) and np.all(full["B"][1:] == 0.0)     # h_max = 1: der ganze Bestand, danach bleibt er 0


def test_ueberschiessen_setzt_den_bestand_auf_null_statt_negativ():
    """Ein Wachstumsfaktor, der den Rest über 1 + 1/r hebt: B_1 = Z * f(1) mit Z = 4 gibt B_1 = 4 (r = 0,5 hat die Nullstelle bei 3); im nächsten Jahr ohne Entnahme ist f(4) < 0."""
    sc = Scenario(r=0.5, sg=1.0, sm=0.0, hmax=1.0)
    z = np.array([[np.log(4.0) + 0.5], [0.0]])               # xi_1 so, dass Z_0 = 4; Z_1 = exp(-0,5) irrelevant, weil B_2 = 0
    res = A.simulate("quote", [0.0], sc, (z, np.zeros_like(z)), record=True)
    assert res["B"][1, 0, 0] == pytest.approx(4.0) and res["B"][2, 0, 0] == 0.0
    assert res["overshoot"][0, 0] == 1.0 and res["collapse"][0, 0] == 1.0 and res["extinct"][0, 0] == 1.0


def test_exakter_survey_und_kein_wachstumsrauschen_geben_deterministische_pfade(ohne_zufall):
    res = A.simulate("escape", [0.45], ohne_zufall, noise(1, 60, 5), record=True)
    assert np.all(res["B"][:, 0, :] == res["B"][:, 0, [0]])   # alle Pfade gleich
    assert np.array_equal(res["Y"][:, 0, :], res["B"][:-1, 0, :])     # Y = B bei sm = 0


def test_beta_eins_braucht_mehr_jahre_als_die_einschwingzeit():
    sc = Scenario(beta=1.0)
    with pytest.raises(ValueError):
        A.simulate("aufwand", [0.2], sc, _zeros(C.BURN))
    assert A.simulate("aufwand", [0.2], sc, _zeros(C.BURN + 1))["obj"].shape == (1, 1)


def test_zwei_pfade_reichen_fuer_den_standardfehler(grundfall):
    """Ein einzelner Pfad hätte keinen Standardfehler (ddof = 1); die App erlaubt mindestens 200 Trainings- und 1 000 Testpfade."""
    assert np.isfinite(A.evaluate("aufwand", 0.2, None, grundfall, noise(1, 20, 2))["se"])
    assert C.TRAIN_MIN >= 2 and C.TEST_MIN >= 2


def test_wertiteration_ohne_diskontierung_wird_abgelehnt():
    for beta in (1.0, 0.0, 1.5):
        with pytest.raises(ValueError):
            D.value_iteration(0.5, 0.2, beta, n=21)


def test_dp_ohne_rauschen_hat_einen_knoten_und_gewicht_eins():
    z, w = D.quadrature(0.0)
    assert z.tolist() == [1.0] and w.tolist() == [1.0]
    z2, w2 = D.quadrature(0.2)
    assert w2.sum() == pytest.approx(1.0) and len(z2) == C.DP_NODES
