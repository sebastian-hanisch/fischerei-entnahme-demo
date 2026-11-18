"""Wertiteration und Formeln (fis_dp.py) an kleinen Fällen; die Orakel gegen Formel, Simulation und Bellman-Gleichung stehen in test_oracle_*.py."""

import numpy as np
import pytest

import fis_constants as C
import fis_dp as D


def test_wert_hat_die_grundeigenschaften():
    g, V, Sp = D.value_iteration(0.5, 0.2, 0.95, n=201)
    assert V[0] == 0.0 and np.all(np.diff(V) >= -1e-12)             # V(0) = 0, nicht fallend
    assert np.all(V >= g - 1e-12)                                    # den ganzen Bestand sofort zu entnehmen ist immer möglich
    assert np.all(Sp <= g + 1e-12) and np.all(Sp >= 0.0)             # der Rest liegt zwischen 0 und dem Bestand


def test_kosten_heben_die_schwelle_und_senken_den_wert():
    s, v = [], []
    for c in C.DP_COST_OPTIONS:
        g, V, Sp = D.value_iteration(0.5, 0.2, 0.95, n=201, cost=c)
        s.append(D.threshold(Sp))
        v.append(D.annual_value(g, V, 0.95))
    assert all(b > a for a, b in zip(s, s[1:])) and all(b < a for a, b in zip(v, v[1:]))


def test_kosten_null_im_kostenzweig_gleichen_dem_zweig_ohne_kosten():
    """Der Kostenzweig (volles Maximum über alle Reste) mit c = 1e-12 muss das laufende Maximum des Zweigs ohne Kosten reproduzieren."""
    g, V0, S0 = D.value_iteration(0.5, 0.2, 0.95, n=151)
    g1, V1, S1 = D.value_iteration(0.5, 0.2, 0.95, n=151, cost=1e-12)
    assert np.allclose(V0, V1, atol=1e-8) and abs(D.threshold(S0) - D.threshold(S1)) <= g[1] - g[0] + 1e-12


def test_jahreswert_und_formel_von_hand():
    g = np.array([0.0, 1.0, 2.0])
    assert D.annual_value(g, np.array([0.0, 10.0, 20.0]), 0.9) == pytest.approx(1.0)           # (1 - 0,9) * V(1)
    assert D.annual_value(g, np.array([0.0, 10.0, 20.0]), 0.9, b=1.5) == pytest.approx(1.5)    # linear zwischen den Gitterpunkten
    assert D.s_star_formula(0.5, 0.95) == pytest.approx(0.5 * (1 - (1 / 0.95 - 1) / 0.5))      # 0,4474
    assert D.s_star_formula(0.5, 0.95) == pytest.approx(0.44737, abs=1e-5)
    assert D.value_formula(0.5, 0.95) == pytest.approx((1 - 17 / 38) + 19 * 0.5 * (17 / 38) * (21 / 38), rel=1e-4)   # β = 0,95 statt 19/20 in Gleitkomma: 441/152
    assert D.value_formula(0.5, 0.95, s_star=0.4, b=0.7) == pytest.approx(0.3 + 19 * 0.5 * 0.4 * 0.6)   # eigene Schwelle und eigener Anfangsbestand


def test_formelbereich_der_schwelle_in_den_reglergrenzen():
    for r, beta in ((0.2, 0.9), (0.5, 0.9), (1.5, 0.9)):
        assert 0.0 < D.s_star_formula(r, beta) < 0.5            # 0 < 1/beta - 1 < r: im Regler-Bereich 0,2 bis 1,5 und β ≥ 0,9 immer
    assert D.s_star_formula(0.1, 0.9) < 0.0                      # außerhalb des Bereichs wäre die Formel negativ: die App lässt r < 0,2 nicht zu
    assert C.R_MIN > 1 / min(C.BETA_OPTIONS) - 1


def test_zwischenspeicher_gibt_dieselben_nur_lesbaren_arrays():
    a = D.solve_cached(0.5, 0.2, 0.95, n=51)
    b = D.solve_cached(0.5, 0.2 + 1e-9, 0.95, n=51)             # gleiche Eingabe bis auf Rundung des Schlüssels
    assert a is b and not a[1].flags.writeable
    with pytest.raises(ValueError):
        a[1][0] = 1.0
    assert D.solve_cached(0.5, 0.2, 0.95, n=61)[0].shape == (61,)
