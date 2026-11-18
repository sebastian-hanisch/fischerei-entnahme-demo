"""Orakel-Tests fischerei-entnahme-demo: der Kern (Simulation, Regeln, Wertiteration) gegen ANDERE Rechenwege (DEMO-PLAYBOOK §12, ORACLE-PLAYBOOK §2).

Orakel: (1) geschlossene Formel der Wertiteration bei σ_g = 0, (2) Gleichgewicht des Aufwands in geschlossener Form, (3) deterministische Bahn der Quote gegen exakte Brüche,
(4) Eigenschaftstests der Simulation, (5) Politik der Wertiteration, simuliert, gegen ihren Wert, (6) Escapement bei exaktem Survey gegen die Politik, (7) gemeinsame Zufallszahlen,
(8) Reihenfolge der Kippstelle im Grundfall, (9) Vollaufzählung aller 256 Vorzeichenpfade einer 4-Jahres-Instanz gegen eine Schleife in reinem Python ohne numpy, (10) Bellman-Gleichung
der Wertiteration mit anderer Quadratur und feinem Suchgitter (scipy), (11) Momente der Gauß-Hermite-Näherung. Jeder Test läuft in wenigen Sekunden; große Läufe (≥ 200 Instanzen) liefen im Scratch."""

import itertools
import math
from fractions import Fraction as Fr

import numpy as np
import pytest

import fis_algorithm as A
import fis_constants as C
import fis_dp as D
import fis_hand as H
from fis_scenario import Scenario, noise


def _zeros(years, paths=1):
    return np.zeros((years, paths)), np.zeros((years, paths))


# ------------------------------------------------------------------------------------------------------ (1) Wertiteration gegen die Formel
@pytest.mark.parametrize("r", [0.3, 0.5, 1.0])
@pytest.mark.parametrize("beta", [0.9, 0.95, 0.98])
def test_wertiteration_gleicht_der_geschlossenen_formel_ohne_wachstumsrauschen(r, beta):
    n = 1001
    g, V, Sp = D.value_iteration(r, 0.0, beta, n=n)
    step = g[1] - g[0]
    s_formula = (1 - (1 / beta - 1) / r) / 2                                    # eigene Schreibweise der Formel, nicht D.s_star_formula
    v_formula = (1 - s_formula) + beta / (1 - beta) * r * s_formula * (1 - s_formula)
    assert abs(D.threshold(Sp) - s_formula) <= step + 1e-12                     # Gitterschritt
    assert np.interp(1.0, g, V) == pytest.approx(v_formula, rel=2e-5)
    assert D.s_star_formula(r, beta) == pytest.approx(s_formula) and D.value_formula(r, beta) == pytest.approx(v_formula)


# ------------------------------------------------------------------------------------------------------ (2) Gleichgewicht des Aufwands
@pytest.mark.parametrize("r", [0.3, 0.5, 1.0])
def test_aufwand_gleichgewicht_in_geschlossener_form(r):
    """Bei e = r / (2 + r) ist ohne Zufall B* = K/2 + r K/4 der Fixpunkt und r K / 4 der Fang; der Fixpunkt folgt aus (1 - e) B + r (1 - e) B (1 - (1 - e) B) = B."""
    e = r / (2 + r)
    b_star = 0.5 + r / 4
    s = (1 - e) * b_star
    assert s + r * s * (1 - s) == pytest.approx(b_star)                                    # Fixpunkt (unabhängig vom Code der Demo)
    sc = Scenario(r=r, sg=0.0, sm=0.0, hmax=1.0)
    res = A.simulate("aufwand", [e], sc, _zeros(400), record=True)
    assert res["B"][-1, 0, 0] == pytest.approx(b_star, rel=1e-9)
    assert res["H"][-1, 0, 0] == pytest.approx(r / 4, rel=1e-9)


# ------------------------------------------------------------------------------------------------------ (3) Quote am MSY ohne Zufall
def test_quote_am_msy_hat_den_fixpunkt_fuenf_achtel_und_ist_nach_unten_instabil(monkeypatch):
    sc = Scenario(r=0.5, sg=0.0, sm=0.0, hmax=0.9)
    last = {}
    for b0 in (0.625, 0.62, 0.55):
        monkeypatch.setattr(C, "B0", b0)
        last[b0] = float(A.simulate("quote", [0.125], sc, _zeros(100), record=True)["B"][-1, 0, 0])
    assert last[0.625] == pytest.approx(0.625, rel=1e-12)             # Fixpunkt: Rest 1/2, Wachstum 1/8 gleich der Quote
    assert last[0.62] == pytest.approx(0.6183, abs=5e-4) and last[0.62] < 0.62     # langsam abwärts, nach 100 Jahren noch fast dort
    assert last[0.55] < 1e-6                                           # unter dem Fixpunkt (hier 0,55 K): Absturz
    # die ersten Jahre in exakten Brüchen (ohne die Kapazitätsgrenze 9/10, die hier nicht greift): dieselbe Bahn
    monkeypatch.setattr(C, "B0", 0.55)
    path = A.simulate("quote", [0.125], sc, _zeros(4), record=True)["B"][:, 0, 0]
    b = Fr(11, 20)
    for t in range(4):
        assert path[t] == pytest.approx(float(b), rel=1e-12)
        s = b - Fr(1, 8)
        b = s + Fr(1, 2) * s * (1 - s)


# ------------------------------------------------------------------------------------------------------ (4) Eigenschaften
@pytest.mark.parametrize("rule, par, par2", [("quote", 0.1, None), ("aufwand", 0.3, None), ("escape", 0.4, None), ("schwelle", 0.3, 0.6)])
@pytest.mark.parametrize("r, sg, sm, hmax", [(0.5, 0.2, 0.2, 0.9), (1.5, 0.4, 1.0, 0.5), (0.2, 0.0, 0.0, 1.0)])
def test_entnahme_und_bestand_bleiben_in_ihren_grenzen(rule, par, par2, r, sg, sm, hmax):
    sc = Scenario(r=r, sg=sg, sm=sm, hmax=hmax)
    res = A.simulate(rule, [par], sc, noise(13, 120, 100), par2=None if par2 is None else [par2], record=True)
    B, Hh = res["B"][:, 0, :], res["H"][:, 0, :]
    assert np.all(Hh >= 0.0) and np.all(Hh <= hmax * B[:-1] + 1e-15) and np.all(B >= 0.0)


def test_rauschfaktoren_haben_den_mittelwert_eins_und_die_lognormale_streuung():
    zg, zm = noise(101, 400, 2000)
    for sig in (0.2, 0.5):
        for xi in (zg, zm):
            f = np.exp(sig * xi - 0.5 * sig * sig)
            assert f.mean() == pytest.approx(1.0, abs=4 * f.std() / np.sqrt(f.size))
            assert np.log(f).std() == pytest.approx(sig, rel=0.01)
    assert abs(np.corrcoef(zg.ravel(), zm.ravel())[0, 1]) < 0.01      # Wachstums- und Messzufall sind unabhängig


# ------------------------------------------------------------------------------------------------------ (5) DP-Politik simuliert gegen den Wert
@pytest.mark.parametrize("sg", [0.1, 0.2, 0.4])
def test_dp_politik_simuliert_gleicht_dem_wert_der_wertiteration_innerhalb_von_drei_standardfehlern(sg):
    beta = 0.95
    g, V, Sp = D.value_iteration(0.5, sg, beta, n=401)
    sc = Scenario(r=0.5, sg=sg, sm=0.0, beta=beta, hmax=1.0)
    ev = A.evaluate("opt", 0.0, None, sc, noise(777, C.T_YEARS, 4000), policy=(g, Sp))
    assert abs(ev["obj"] - D.annual_value(g, V, beta)) <= 3 * ev["se"]


# ------------------------------------------------------------------------------------------------------ (6) Escapement bei exaktem Survey
def test_politik_der_wertiteration_ist_eine_reine_schwelle_und_escapement_gleicht_ihr():
    g, V, Sp = D.value_iteration(0.5, 0.2, 0.95, n=401)
    s_star = D.threshold(Sp)
    assert np.allclose(Sp, np.minimum(g, s_star), atol=1e-12)           # unter der Schwelle kein Fang, darüber Rest S*
    sc = Scenario(r=0.5, sg=0.2, sm=0.0, hmax=0.9)
    nz = noise(5, C.T_YEARS, 300)
    opt = A.simulate("opt", [0.0], sc, nz, policy=(g, Sp))
    esc = A.simulate("escape", [s_star], sc, nz)
    assert np.allclose(opt["obj"], esc["obj"], rtol=0, atol=1e-12) and np.array_equal(opt["collapse"], esc["collapse"])


# ------------------------------------------------------------------------------------------------------ (7) Gemeinsame Zufallszahlen
def test_alle_regeln_sehen_dieselben_wachstumsfaktoren(grundfall):
    nz = noise(21, 60, 40)
    seen = []
    for rule, par, par2 in (("quote", 0.05, None), ("aufwand", 0.2, None), ("schwelle", 0.3, 0.6)):
        res = A.simulate(rule, [par], grundfall, nz, par2=None if par2 is None else [par2], record=True)
        S = res["B"][:-1, 0, :] - res["H"][:, 0, :]
        raw = S + grundfall.r * S * (1 - S)
        mask = raw > 1e-9
        z = np.where(mask, res["B"][1:, 0, :] / np.where(mask, raw, 1.0), np.nan)
        seen.append(z)
    expected = np.exp(grundfall.sg * nz[0] - 0.5 * grundfall.sg ** 2)
    for z in seen:
        ok = ~np.isnan(z)
        assert ok.mean() > 0.9                                          # fast jedes Jahr lässt sich der Faktor zurückrechnen
        assert np.allclose(z[ok], expected[ok], rtol=1e-10, atol=0)     # und ist bei jeder Regel derselbe, den die Zufallszahlen vorgeben


# ------------------------------------------------------------------------------------------------------ (8) Reihenfolge der Kippstelle
def test_im_grundfall_wechselt_die_reihenfolge_mit_kollapsgrenze_zwischen_messrauschen_015_und_02():
    """Escapement minus Aufwand (gepaart, beide mit Kollapsgrenze abgestimmt): bei 0,15 vorn (Vormessung +0,0107), bei 0,2 hinten (−0,0017); als Band, nicht als Ziffern."""
    train = noise(C.GRID_SEED_TRAIN, C.T_YEARS, C.GRID_TRAIN)
    test = noise(C.GRID_SEED_TEST, C.T_YEARS, C.GRID_TEST)
    diffs = {}
    auf = None
    for sm in (0.15, 0.2):
        sc = Scenario(sm=sm)
        if auf is None:
            _, tu = A.tune("aufwand", sc, train)
            auf = A.evaluate("aufwand", tu.par, None, sc, test)
        _, tu = A.tune("escape", sc, train)
        diffs[sm] = A.paired_difference(A.evaluate("escape", tu.par, None, sc, test)["paths"], auf["paths"])
    assert 0.008 < diffs[0.15][0] < 0.013
    assert -0.0035 < diffs[0.2][0] < -0.0005 and diffs[0.2][0] < -5 * diffs[0.2][1]


# ------------------------------------------------------------------------------------------------------ (9) Vollaufzählung gegen reines Python
def _reference(rule, par, par2, sc, xi, eta, burn):
    """Eine Schleife in reinem Python, Zeile für Zeile aus der Modellbeschreibung (README): Messung, Entnahme, Rest, Wachstum, Ziel, Kollaps."""
    B, years, obj, bmin = 1.0, len(xi), 0.0, math.inf
    for t in range(years):
        Y = B * math.exp(sc.sm * xi[t] - sc.sm ** 2 / 2) if sc.sm > 0 else B
        if rule == "quote":
            h = min(par, B)
        elif rule == "aufwand":
            h = par * B
        elif rule == "escape":
            h = min(B, max(0.0, Y - par))
        else:
            h = min(B, par2 * max(0.0, Y - par))
        h = min(h, sc.hmax * B)
        if sc.beta < 1.0:
            obj += (1 - sc.beta) / (1 - sc.beta ** years) * sc.beta ** t * h
        elif t >= burn:
            obj += h / (years - burn)
        S = B - h
        z = math.exp(sc.sg * eta[t] - sc.sg ** 2 / 2) if sc.sg > 0 else 1.0
        B = z * max(0.0, S + sc.r * S * (1 - S))
        bmin = min(bmin, B)
    return obj, bmin < sc.thr


@pytest.mark.parametrize("sc, burn", [
    (Scenario(r=0.5, sg=0.3, sm=0.4, beta=0.95, hmax=0.9, thr=0.5), 50),
    (Scenario(r=1.2, sg=0.5, sm=0.0, beta=0.9, hmax=1.0, thr=0.2), 50),
    (Scenario(r=0.8, sg=0.0, sm=0.6, beta=1.0, hmax=0.7, thr=0.3), 2),          # beta = 1: Mittel ab der Einschwingzeit (hier 2 Jahre)
])
@pytest.mark.parametrize("rule, par, par2", [("quote", 0.2, None), ("aufwand", 0.3, None), ("escape", 0.5, None), ("schwelle", 0.4, 0.7)])
def test_alle_256_vorzeichenpfade_einer_vierjahresinstanz_gleichen_der_python_schleife(sc, burn, rule, par, par2):
    combos = list(itertools.product(itertools.product((-1.0, 1.0), (-1.0, 1.0)), repeat=4))     # 4^4 = 256 Pfade: je Jahr (xi, eta) in {±1}^2
    xi = np.array([[c[t][0] for c in combos] for t in range(4)])
    eta = np.array([[c[t][1] for c in combos] for t in range(4)])
    assert xi.shape == (4, 256)
    res = A.simulate(rule, [par], sc, (eta, xi), par2=None if par2 is None else [par2], burn=burn)
    for j in range(256):
        obj, coll = _reference(rule, par, par2, sc, xi[:, j], eta[:, j], burn)
        assert res["obj"][0, j] == pytest.approx(obj, rel=1e-12, abs=1e-15), j
        assert bool(res["collapse"][0, j]) == coll, j


# ------------------------------------------------------------------------------------------------------ (10) Bellman-Gleichung mit anderer Quadratur
@pytest.mark.parametrize("cost", [0.0, 0.1])
def test_wertiteration_erfuellt_die_bellman_gleichung_auch_mit_anderer_quadratur_und_feinem_suchgitter(cost):
    special = pytest.importorskip("scipy.special")
    r, sg, beta = 0.5, 0.2, 0.95
    g, V, Sp = D.value_iteration(r, sg, beta, n=401, cost=cost)
    q = special.ndtri((np.arange(2000) + 0.5) / 2000)                  # Mittelpunktsregel auf den Quantilen: kein Gauß-Hermite
    zk = np.exp(sg * q - 0.5 * sg ** 2)
    for b in (0.3, 0.5, 0.8, 1.0, 1.4):
        s = np.linspace(0.0, b, 4001)
        if cost > 0:
            s = s[1:]                                                    # S = 0 hieße unendlicher Aufwand
            reward = (b - s) - cost * np.log(b / s)
        else:
            reward = b - s
        nxt = np.minimum((s + r * s * (1 - s))[:, None] * zk[None, :], g[-1])
        rhs = (reward + beta * np.interp(nxt, g, V).mean(axis=1)).max()
        assert rhs == pytest.approx(float(np.interp(b, g, V)), abs=4e-4), (cost, b)


# ------------------------------------------------------------------------------------------------------ (11) Quadratur
@pytest.mark.parametrize("sg", [0.1, 0.4, 0.8])
def test_gauss_hermite_knoten_treffen_die_momente_der_lognormalverteilung(sg):
    zk, w = D.quadrature(sg)
    assert w.sum() == pytest.approx(1.0) and (zk * w).sum() == pytest.approx(1.0, rel=1e-10)
    assert (zk ** 2 * w).sum() == pytest.approx(math.exp(sg ** 2), rel=1e-8)       # E[Z^2] = exp(sg^2) für Z = exp(sg X - sg^2/2)


# ------------------------------------------------------------------------------------------------------ Hand gegen Wertiteration
def test_handbeispiel_in_brueche_gleicht_der_wertiteration():
    s_h, v_h = H.optimum_closed()
    assert s_h == Fr(17, 38) and v_h == Fr(441, 152) and H.annual_value() == Fr(441, 3040)
    g, V, Sp = D.value_iteration(0.5, 0.0, 0.95, n=2001)
    assert abs(D.threshold(Sp) - float(s_h)) <= g[1] - g[0] + 1e-12 and np.interp(1.0, g, V) == pytest.approx(float(v_h), rel=2e-5)
