"""Die Zahlen der README („Befunde“) werden hier nachgerechnet, mit den echten Gittern, Pfadzahlen und Zufalls-Seeds der README (Training 1 000 Pfade, Zufalls-Seed 11; Test 5 000 Pfade,
Zufalls-Seed 22; Raster aus data/kipp.csv). Neu gerechnete Zahlen werden als Band verglichen (Plattformen können Gleichstände und die letzte Stelle verschieden auflösen,
DEMO-PLAYBOOK §5); das vorgerechnete Raster (data/kipp.csv) wird gelesen und seine Kippstellen werden aus den Zeilen neu interpoliert. Nicht hier, nur in der README:
die Wiederholung der Kippstelle mit weiteren Zufalls-Seed-Paaren und die Messung des feineren Gitters der gedämpften Schwelle (README, „Messungen ohne Test“)."""

from fractions import Fraction as Fr

import numpy as np
import pytest

import fis_algorithm as A
import fis_constants as C
import fis_dp as D
import fis_evaluation as EV
import fis_grid as G
import fis_hand as H
from fis_scenario import Scenario, noise

ZIEL = 8e-4          # Band für das Ziel (K je Jahr): Standardfehler 0,0002 bis 0,0003, dazu Gleichstände der Abstimmung
KOLL = 0.015         # Band für Kollapsanteile auf 5 000 Pfaden (Standardfehler bis 0,007)


@pytest.fixture(scope="module")
def train():
    return noise(C.DEFAULT_SEED_TRAIN, C.T_YEARS, C.DEFAULT_TRAIN)


@pytest.fixture(scope="module")
def test_paths():
    return noise(C.DEFAULT_SEED_TEST, C.T_YEARS, C.DEFAULT_TEST)


@pytest.fixture(scope="module")
def cmp_grund():
    return EV.compare_rules(Scenario())


@pytest.fixture(scope="module")
def grid():
    rows = G.load()
    assert rows
    return rows


def _capped(rule, sc, train, test):
    """Regel mit der Abstimmung mit Kollapsgrenze (Training) auf den Testpfaden bewertet."""
    tu = A.tune(rule, sc, train)[1]
    return A.evaluate(rule, tu.par, tu.par2, sc, test)


def _row(cmp, rule, mode):
    return EV.row_of(cmp, rule, mode)


# ---------------------------------------------------------------------------------------------------------------- Grundfall: Tabelle der README
@pytest.mark.parametrize(
    "rule, mode, par, par2, ziel, kollaps",
    [
        ("quote", "frei", 0.10, None, 0.0839, 0.986), ("quote", "kollaps", 0.06, None, 0.0598, 0.060),
        ("aufwand", "frei", 0.23, None, 0.1303, 0.108), ("aufwand", "kollaps", 0.22, None, 0.1302, 0.047),
        ("escape", "frei", 0.47, None, 0.1374, 0.368), ("escape", "kollaps", 0.63, None, 0.1285, 0.090),
        ("schwelle", "frei", 0.38, 0.7, 0.1400, 0.056), ("schwelle", "kollaps", 0.38, 0.7, 0.1400, 0.056),
        ("quote", "lehrbuch", 0.125, None, 0.0821, 1.000), ("aufwand", "lehrbuch", 0.20, None, 0.1281, 0.009), ("escape", "lehrbuch", 0.50, None, 0.1370, 0.293),
        ("opt", "dp", 0.0, None, 0.1373, 0.418),
    ],
)
def test_grundfall_tabelle(cmp_grund, rule, mode, par, par2, ziel, kollaps):
    row = _row(cmp_grund, rule, mode)
    assert row["obj"] == pytest.approx(ziel, abs=ZIEL) and row["collapse"] == pytest.approx(kollaps, abs=KOLL)
    if rule != "opt":
        assert row["par"] == pytest.approx(par, abs=0.021 if mode in ("frei", "kollaps") else 1e-9)        # abgestimmter Wert: höchstens zwei Gitterschritte daneben
    if par2 is not None:
        assert row["par2"] == pytest.approx(par2, abs=0.11)
    assert row["feasible"]


def test_grundfall_referenz_und_anteile(cmp_grund):
    ref = cmp_grund["reference"]
    assert ref == pytest.approx(0.1448, abs=5e-4) and cmp_grund["dp_threshold"] == pytest.approx(0.4525, abs=1e-9)
    share = {(r["rule"], r["mode"]): 100 * r["obj"] / ref for r in cmp_grund["rows"]}
    assert share[("schwelle", "kollaps")] == pytest.approx(96.6, abs=0.8) and share[("aufwand", "kollaps")] == pytest.approx(89.9, abs=0.8)
    assert share[("quote", "kollaps")] == pytest.approx(41.3, abs=0.8) and share[("escape", "kollaps")] == pytest.approx(88.7, abs=0.8)
    assert share[("escape", "frei")] == pytest.approx(94.9, abs=0.8) and share[("quote", "frei")] == pytest.approx(57.9, abs=0.8)


def test_grundfall_gepaarte_differenzen(cmp_grund):
    d = cmp_grund["diffs"]
    assert d[("escape", "frei")][0] == pytest.approx(0.0071, abs=4e-4) and d[("escape", "kollaps")][0] == pytest.approx(-0.0017, abs=4e-4)
    assert d[("schwelle", "frei")][0] == pytest.approx(0.0097, abs=4e-4) and d[("schwelle", "kollaps")][0] == pytest.approx(0.0098, abs=4e-4)
    assert all(se <= 1.2e-4 for _, se in d.values())          # README: gepaarte Differenzen im Grundfall ±0,0001
    assert all(r["se"] <= 3.5e-4 for r in cmp_grund["rows"])  # README: Standardfehler des Ziels im Grundfall 0,0000 bis 0,0003
    assert d[("escape", "kollaps")][0] < -5 * d[("escape", "kollaps")][1]          # das Vorzeichen ist mehr als fünf Standardfehler vom Nullwert entfernt


def test_grundfall_kernfragen_der_app(cmp_grund):
    kind, text = EV.verdict_hidden(cmp_grund)
    assert kind == "warning" and text.startswith("3 von 4") and "Quote 98.6 %" in text
    kind, text = EV.verdict_order(cmp_grund, 0.2)
    assert kind == "warning" and "Escapement minus Aufwand -0.0017" in text


def test_aufwand_und_quote_sind_ueber_das_messrauschen_konstant_per_konstruktion(train, test_paths):
    ev = [A.evaluate(rule, par, None, Scenario(sm=sm), test_paths) for rule, par in (("quote", 0.06), ("aufwand", 0.22)) for sm in (0.0, 0.5)]
    assert ev[0]["obj"] == ev[1]["obj"] and ev[2]["obj"] == ev[3]["obj"] and ev[0]["collapse"] == ev[1]["collapse"]


# ---------------------------------------------------------------------------------------------------------------- exakter Survey, ohne Rauschen
@pytest.mark.parametrize(
    "scen, expect",
    [
        (Scenario(sm=0.0), {("escape", "frei"): (0.1453, 0.0), ("schwelle", "frei"): (0.1453, 0.0), ("opt", "dp"): (0.1453, 0.0), ("aufwand", "frei"): (0.1303, 0.108), ("quote", "kollaps"): (0.0598, 0.060)}),
        (Scenario(sg=0.0, sm=0.0), {("quote", "lehrbuch"): (0.1250, 0.0), ("aufwand", "frei"): (0.1403, 0.0), ("escape", "frei"): (0.1451, 0.0), ("opt", "dp"): (0.1451, 0.0)}),
    ],
)
def test_exakter_survey_und_ohne_rauschen(scen, expect):
    cmp = EV.compare_rules(scen)
    for (rule, mode), (ziel, koll) in expect.items():
        row = _row(cmp, rule, mode)
        assert row["obj"] == pytest.approx(ziel, abs=ZIEL) and row["collapse"] == pytest.approx(koll, abs=KOLL), (rule, mode)
    if scen.sg > 0:
        assert _row(cmp, "schwelle", "frei")["par2"] == pytest.approx(1.0)       # bei exaktem Survey wird die Dämpfung 1: die Schwelle ist die Escapement-Regel


# ---------------------------------------------------------------------------------------------------------------- Befund 1 und 2: Quote gegen Aufwand
def test_quote_am_msy_kollabiert_schon_bei_winzigem_rauschen(test_paths):
    expected = {0.0: 0.000, 0.01: 0.402, 0.02: 0.736, 0.05: 0.967, 0.2: 1.000}
    for sg, koll in expected.items():
        assert A.evaluate("quote", 0.125, None, Scenario(sg=sg, sm=0.0), test_paths)["collapse"] == pytest.approx(koll, abs=KOLL), sg


def test_aufwand_am_msy_ist_stabil_aber_die_kollapsgrenze_entscheidet(test_paths):
    for sg, k10 in ((0.0, 0.0), (0.05, 0.0), (0.1, 0.0), (0.2, 0.009)):
        assert A.evaluate("aufwand", 0.2, None, Scenario(sg=sg, sm=0.0), test_paths)["collapse"] == pytest.approx(k10, abs=KOLL), sg
    assert A.evaluate("aufwand", 0.2, None, Scenario(sg=0.2, sm=0.0, thr=0.2), test_paths)["collapse"] == pytest.approx(0.43, abs=0.02)


def test_deterministische_bahnen_quote_instabil_aufwand_kehrt_zurueck(monkeypatch):
    sc = Scenario(sg=0.0, sm=0.0)
    zeros = (np.zeros((100, 1)), np.zeros((100, 1)))
    out = {}
    for b0 in (0.2, 0.45, 0.55, 0.62):
        monkeypatch.setattr(C, "B0", b0)
        out[b0] = (A.simulate("quote", [0.125], sc, zeros, record=True)["B"][-1, 0, 0], A.simulate("aufwand", [0.2], sc, zeros, record=True)["B"][-1, 0, 0])
    assert out[0.62][0] == pytest.approx(0.6183, abs=5e-4) and all(out[b][0] < 1e-6 for b in (0.2, 0.45, 0.55))
    assert all(out[b][1] == pytest.approx(0.625, abs=1e-6) for b in out)


# ---------------------------------------------------------------------------------------------------------------- Befund 3: Escapement erreicht bei exaktem Survey das Ziel der DP-Politik
def test_escapement_gleicht_bei_exaktem_survey_der_dp_politik_ueber_das_wachstumsrauschen(train, test_paths):
    expected = {0.0: 0.1451, 0.05: 0.1452, 0.1: 0.1453, 0.2: 0.1453, 0.3: 0.1443, 0.4: 0.1414}
    for sg, ziel in expected.items():
        sc = Scenario(sg=sg, sm=0.0)
        free, _ = A.tune("escape", sc, train)
        esc = A.evaluate("escape", free.par, None, sc, test_paths)
        g, V, Sp = D.solve_cached(0.5, sg, 0.95)
        opt = A.evaluate("opt", 0.0, None, sc, test_paths, policy=(g, Sp))
        assert esc["obj"] == pytest.approx(ziel, abs=ZIEL) and abs(esc["obj"] - opt["obj"]) <= 3e-4, sg       # gleich bis auf die vierte Stelle


# ---------------------------------------------------------------------------------------------------------------- Befund 5, 6, 8: Diskontierung, Kapazitätsgrenze, Kollapsgrenze
def test_ziel_allein_versteckt_den_kollaps_und_beta_eins_macht_die_quote_vorsichtiger(train, test_paths):
    q95, _ = A.tune("quote", Scenario(), train)
    e95 = A.evaluate("quote", q95.par, None, Scenario(), test_paths)
    assert q95.par == pytest.approx(0.10) and e95["obj"] == pytest.approx(0.0839, abs=ZIEL) and e95["collapse"] == pytest.approx(0.986, abs=KOLL)
    sc1 = Scenario(beta=1.0)
    q1, _ = A.tune("quote", sc1, train)
    e1 = A.evaluate("quote", q1.par, None, sc1, test_paths)
    assert q1.par == pytest.approx(0.065, abs=0.011) and e1["collapse"] == pytest.approx(0.134, abs=0.03) and e1["obj"] == pytest.approx(0.0599, abs=ZIEL)


def test_ohne_kapazitaetsgrenze_kann_escapement_den_bestand_aussterben_lassen(train, test_paths):
    out = {}
    for hmax in (0.9, 1.0):
        sc = Scenario(hmax=hmax)
        free, _ = A.tune("escape", sc, train)
        out[hmax] = A.evaluate("escape", free.par, None, sc, test_paths)
    assert out[0.9]["extinct"] == pytest.approx(0.0, abs=0.005) and out[0.9]["collapse"] == pytest.approx(0.368, abs=KOLL) and out[0.9]["obj"] == pytest.approx(0.1374, abs=ZIEL)
    assert out[1.0]["extinct"] == pytest.approx(0.325, abs=KOLL) and out[1.0]["collapse"] == pytest.approx(0.574, abs=KOLL) and out[1.0]["obj"] == pytest.approx(0.1322, abs=ZIEL)


def test_kollapsgrenze_aendert_die_anteile_stark(test_paths):
    table = {
        ("aufwand", 0.22, None, 0.2): (0.001, 0.047, 0.740), ("escape", 0.63, None, 0.2): (0.000, 0.090, 0.652), ("schwelle", 0.38, 0.7, 0.2): (0.001, 0.056, 0.448),
        ("quote", 0.06, None, 0.2): (0.060, 0.060, 0.077), ("escape", 0.45, None, 0.0): (0.000, 0.000, 0.001),
    }
    for (rule, par, par2, sm), expected in table.items():
        got = [A.evaluate(rule, par, par2, Scenario(sm=sm, thr=t), test_paths)["collapse"] for t in (0.05, 0.1, 0.2)]
        assert got == pytest.approx(list(expected), abs=KOLL), (rule, par)


# ---------------------------------------------------------------------------------------------------------------- Befund 9 und Gegenproben der Wertiteration
def test_kosten_heben_die_schwelle_der_wertiteration():
    expected = {0.0: (0.4525, 0.1448), 0.02: (0.4625, 0.1398), 0.05: (0.4775, 0.1324), 0.1: (0.5025, 0.1206), 0.2: (0.555, 0.0988)}
    for c, (s, v) in expected.items():
        g, V, Sp = D.value_iteration(0.5, 0.2, 0.95, cost=c)
        assert D.threshold(Sp) == pytest.approx(s, abs=0.0051) and D.annual_value(g, V, 0.95) == pytest.approx(v, abs=3e-4), c


def test_gitterfeinheit_der_wertiteration():
    for n, nodes, s in ((401, 21, 0.4500), (801, 41, 0.4525), (1601, 41, 0.4512)):
        g, V, Sp = D.value_iteration(0.5, 0.2, 0.95, n=n, nodes=nodes)
        assert D.threshold(Sp) == pytest.approx(s, abs=0.0006) and D.annual_value(g, V, 0.95) == pytest.approx(0.14485, abs=2e-5), n


def test_wertiteration_gegen_formel_in_neun_faellen():
    worst_s, worst_v = 0.0, 0.0
    for r in (0.3, 0.5, 1.0):
        for beta in (0.9, 0.95, 0.98):
            g, V, Sp = D.value_iteration(r, 0.0, beta, n=2001)
            worst_s = max(worst_s, abs(D.threshold(Sp) - D.s_star_formula(r, beta)))
            worst_v = max(worst_v, abs(np.interp(1.0, g, V) - D.value_formula(r, beta)))
    assert worst_s <= 0.0005 and worst_v <= 2e-5


def test_wertiteration_gegen_simulation_der_politik():
    expected = {0.1: (0.14506, 0.4475), 0.2: (0.14485, 0.4525), 0.4: (0.14031, 0.4775)}
    for sg, (v, s) in expected.items():
        g, V, Sp = D.value_iteration(0.5, sg, 0.95)
        ev = A.evaluate("opt", 0.0, None, Scenario(sg=sg, sm=0.0, hmax=1.0), noise(777, C.T_YEARS, 4000), policy=(g, Sp))
        assert D.annual_value(g, V, 0.95) == pytest.approx(v, abs=1e-4) and D.threshold(Sp) == pytest.approx(s, abs=1e-9)
        assert abs(ev["obj"] - v) <= 3 * ev["se"], sg
    assert D.threshold(D.value_iteration(0.5, 0.4, 0.95, n=201)[2]) > D.s_star_formula(0.5, 0.95)      # mit Rauschen liegt die Schwelle über der Formel


def test_schwelle_der_wertiteration_gegen_die_formel_ist_bei_kleinem_rauschen_nicht_aufgeloest():
    """README: ab σ_g = 0,2 liegt die Schwelle über der Formel (0,4474); bei σ_g = 0,1 liegt sie weniger als einen Gitterschritt (0,0025) darüber."""
    formula = D.s_star_formula(0.5, 0.95)
    assert formula == pytest.approx(0.4474, abs=5e-5)
    step = 2.0 / 800
    s = {sg: D.threshold(D.value_iteration(0.5, sg, 0.95)[2]) for sg in (0.1, 0.2, 0.4)}
    assert s[0.1] == pytest.approx(0.4475, abs=1e-9) and 0 <= s[0.1] - formula < step
    assert s[0.2] - formula > step and s[0.4] - formula > step


# ---------------------------------------------------------------------------------------------------------------- Diskontfaktor und r = 1,5 (live gerechnet)
def test_diskontfaktor_aendert_die_reihenfolge_von_escapement_und_aufwand_nur_knapp(train, test_paths):
    expected = {0.9: -0.0019, 0.95: -0.0017, 0.98: 0.0003, 1.0: 0.0011}
    for beta, delta in expected.items():
        sc = Scenario(beta=beta)
        ev = {rule: _capped(rule, sc, train, test_paths) for rule in ("aufwand", "escape", "schwelle")}
        d, se = A.paired_difference(ev["escape"]["paths"], ev["aufwand"]["paths"])
        assert d == pytest.approx(delta, abs=4e-4), beta
        assert A.paired_difference(ev["schwelle"]["paths"], ev["aufwand"]["paths"])[0] > 0.007        # die gedämpfte Schwelle liegt bei allen vier Diskontfaktoren vorn


def test_bei_r_eins_komma_fuenf_liegt_die_gedaempfte_schwelle_hinter_dem_aufwand(train, test_paths):
    sc = Scenario(r=1.5)
    ev = {rule: _capped(rule, sc, train, test_paths) for rule in ("aufwand", "schwelle")}
    assert ev["schwelle"]["obj"] / ev["aufwand"]["obj"] - 1 == pytest.approx(-0.019, abs=0.006)
    assert ev["aufwand"]["obj"] == pytest.approx(0.3628, abs=ZIEL)


def test_ueberschiessen_ohne_entnahme_bei_grossem_wachstumsrauschen(test_paths):
    """Ohne Entnahme stürzen bei σ_g = 0,4 und r = 0,5 rund 17 % der Testpfade nach Überschießen ab (die Voreinstellung σ_g = 0,2 hat keinen solchen Pfad)."""
    assert A.evaluate("quote", 0.0, None, Scenario(sg=0.4, sm=0.0, hmax=1.0), test_paths)["overshoot"] == pytest.approx(0.169, abs=0.02)
    assert A.evaluate("quote", 0.0, None, Scenario(sg=0.2, sm=0.0, hmax=1.0), test_paths)["overshoot"] == 0.0


# ---------------------------------------------------------------------------------------------------------------- Presets
def test_zahlen_der_presets(test_paths):
    e = A.evaluate("escape", 0.45, None, Scenario(sm=0.0), test_paths)
    assert e["collapse"] == pytest.approx(0.0, abs=0.002)                                                    # „Escapement bei exaktem Survey“
    e = A.evaluate("escape", 0.47, None, Scenario(sm=0.3), test_paths)
    assert e["collapse"] == pytest.approx(0.946, abs=KOLL) and e["obj"] == pytest.approx(0.1297, abs=ZIEL)   # „Verrauschter Survey“
    e = A.evaluate("aufwand", 0.35, None, Scenario(r=1.0), test_paths)
    assert e["collapse"] == pytest.approx(0.0, abs=0.002) and e["obj"] == pytest.approx(0.2466, abs=ZIEL)    # „Schneller Bestand“
    e = A.evaluate("quote", 0.125, None, Scenario(sg=0.01), test_paths)
    assert e["collapse"] == pytest.approx(0.402, abs=KOLL)                                                   # „Lehrbuch: Quote am MSY“


# ---------------------------------------------------------------------------------------------------------------- Raster: Kippstellen
def _kipp(grid, sid, rule, mode):
    rows = G.series(grid, sid, rule, mode)
    return EV.kipp([r["sm"] for r in rows], [r["d_vs_aufwand"] for r in rows])


def test_kippstellen_des_rasters(grid):
    expected = {   # Kennung: (Escapement frei, Escapement mit Kollapsgrenze, Schwelle mit Kollapsgrenze); None = kein Vorzeichenwechsel im Raster bis 0,8
        "grund": (0.292, 0.193, 0.379), "thr05": (0.292, 0.292, 0.419), "thr20": (0.292, 0.153, 0.401), "hmax70": (0.424, 0.415, 0.452), "hmax100": (0.208, 0.156, 0.377),
        "sg05": (0.171, 0.158, 0.255), "sg10": (0.199, 0.167, 0.301), "sg30": (0.453, 0.247, 0.526), "sg40": (0.703, None, None), "r02": (0.746, None, 0.688),
        "r03": (0.466, 0.222, 0.534), "r08": (0.206, 0.195, 0.279), "r10": (0.176, 0.176, 0.233),
    }
    for sid, (a, b, c) in expected.items():
        for got, want in zip((_kipp(grid, sid, "escape", "frei"), _kipp(grid, sid, "escape", "kollaps"), _kipp(grid, sid, "schwelle", "kollaps")), (a, b, c)):
            if want is None:
                assert got is None, sid
            else:
                assert got == pytest.approx(want, abs=0.006), (sid, got, want)


def test_mit_kollapsgrenze_liegt_die_kippstelle_in_sieben_einstellungen_zwischen_016_und_025(grid):
    seven = ["grund", "sg05", "sg10", "sg30", "r03", "r08", "r10"]
    ks = [_kipp(grid, sid, "escape", "kollaps") for sid in seven]
    assert all(k is not None for k in ks) and 0.158 <= min(ks) + 0.006 and max(ks) <= 0.253
    assert all(all(r["feasible"] for r in G.series(grid, sid, "aufwand", "kollaps")) for sid in seven)       # in diesen sieben ist der Aufwand zulässig
    free = [_kipp(grid, sid, "escape", "frei") for sid in ("sg05", "sg10", "grund", "sg30", "sg40", "r02", "r03", "r08", "r10")]
    assert min(free) == pytest.approx(0.171, abs=0.006) and max(free) == pytest.approx(0.746, abs=0.006)          # ohne Kollapsgrenze von 0,17 bis 0,75


def test_kippstellen_ueber_alle_dreizehn_einstellungen_des_rasters(grid):
    """README, Raster-Befund 1: ohne Kollapsgrenze hat jede der 13 Einstellungen einen Vorzeichenwechsel (0,171 bis 0,746); mit Kollapsgrenze 11 von 13 (0,153 bis 0,415),
    ohne Wechsel sind σ_g = 0,4 und r = 0,2; der Aufwand ist mit Kollapsgrenze in 12 von 13 Einstellungen zulässig (nur σ_g = 0,4 nicht)."""
    ids = sorted({r["scen"] for r in grid})
    assert len(ids) == 13
    free = {sid: _kipp(grid, sid, "escape", "frei") for sid in ids}
    assert all(k is not None for k in free.values()) and min(free.values()) == pytest.approx(0.171, abs=0.006) and max(free.values()) == pytest.approx(0.746, abs=0.006)
    capped = {sid: _kipp(grid, sid, "escape", "kollaps") for sid in ids}
    assert sorted(s for s, k in capped.items() if k is None) == ["r02", "sg40"]
    vals = [k for k in capped.values() if k is not None]
    assert len(vals) == 11 and min(vals) == pytest.approx(0.153, abs=0.006) and max(vals) == pytest.approx(0.415, abs=0.006)
    feasible = {sid: all(r["feasible"] for r in G.series(grid, sid, "aufwand", "kollaps")) for sid in ids}
    assert sum(feasible.values()) == 12 and [s for s, ok in feasible.items() if not ok] == ["sg40"]


def test_kippstelle_im_grundfall_liegt_zwischen_den_rasterpunkten_0_15_und_0_2(grid):
    """App-Einleitung: ab einem Messrauschen zwischen 0,15 und 0,2 (linear interpoliert etwa 0,19) liegt Escapement mit Kollapsgrenze hinter dem Aufwand."""
    rows = G.series(grid, "grund", "escape", "kollaps")
    d = {r["sm"]: r["d_vs_aufwand"] for r in rows}
    assert d[0.15] > 0 >= d[0.2] and 0.15 < _kipp(grid, "grund", "escape", "kollaps") < 0.2
    assert _kipp(grid, "grund", "escape", "kollaps") == pytest.approx(0.19, abs=0.005)


def test_standardfehler_im_raster(grid):
    """README, Befunde (Einleitung): im Raster Standardfehler des Ziels und der gepaarten Differenzen bis 0,0004, bei σ_g = 0,4 bis 0,0006."""
    rest = [r for r in grid if r["scen"] != "sg40"]
    assert 0.0003 < max(r["se"] for r in rest) <= 0.0005 and 0.0003 < max(r["d_se"] for r in rest if r["d_se"] is not None) <= 0.0005
    sg40 = [r for r in grid if r["scen"] == "sg40"]
    assert 0.0005 < max(r["se"] for r in sg40) <= 0.0007 and 0.0005 < max(r["d_se"] for r in sg40 if r["d_se"] is not None) <= 0.0007


def test_escapement_im_grundfall_des_rasters_hat_bei_exaktem_survey_null_und_beim_groessten_rauschen_ganz_kollaps(grid):
    """README, Methodik: im Grundfall des Rasters hat Escapement (frei) bei σ_m = 0 den Kollaps 0 % und bei σ_m = 0,8 100 %; nicht in allen Einstellungen (σ_g = 0,4: 32 % schon bei σ_m = 0)."""
    frei = G.series(grid, "grund", "escape", "frei")
    assert frei[0]["sm"] == 0.0 and frei[0]["collapse"] == pytest.approx(0.0, abs=0.002) and frei[-1]["sm"] == 0.8 and frei[-1]["collapse"] == pytest.approx(1.0, abs=0.002)
    assert G.series(grid, "sg40", "escape", "frei")[0]["collapse"] == pytest.approx(0.32, abs=0.03)
    assert G.series(grid, "hmax70", "escape", "frei")[-1]["collapse"] < 0.9


def test_aufwand_ist_bei_sigma_g_vier_zehntel_nicht_zulaessig_und_bei_r_zwei_zehntel_ist_die_schwelle_vorn(grid):
    assert not all(r["feasible"] for r in G.series(grid, "sg40", "aufwand", "kollaps")) and G.series(grid, "sg40", "aufwand", "kollaps")[0]["collapse"] > 0.1
    esc = G.series(grid, "r02", "escape", "kollaps")
    assert all(r["d_vs_aufwand"] > 0 for r in esc) and G.series(grid, "r02", "aufwand", "kollaps")[0]["par"] == pytest.approx(0.06)
    assert _kipp(grid, "hmax70", "escape", "kollaps") is not None
    h70 = G.series(grid, "hmax70", "escape", "kollaps")
    assert all(r["d_vs_aufwand"] > 0 for r in h70 if r["sm"] <= 0.4)                                               # bis 0,4 kein Kipp


def test_tabelle_der_differenzen_im_grundfall(grid):
    esc_f, esc_k = G.series(grid, "grund", "escape", "frei"), G.series(grid, "grund", "escape", "kollaps")
    at = lambda rows, sm: next(r for r in rows if r["sm"] == sm)
    assert at(esc_f, 0.0)["d_vs_aufwand"] == pytest.approx(0.0151, abs=4e-4) and at(esc_f, 0.2)["d_vs_aufwand"] == pytest.approx(0.0071, abs=4e-4)
    assert at(esc_f, 0.3)["d_vs_aufwand"] == pytest.approx(-0.0006, abs=4e-4) and at(esc_f, 0.3)["collapse"] == pytest.approx(0.95, abs=0.02)
    assert at(esc_k, 0.2)["d_vs_aufwand"] == pytest.approx(-0.0017, abs=4e-4) and at(esc_k, 0.3)["d_vs_aufwand"] == pytest.approx(-0.0534, abs=2e-3)
    assert not at(esc_k, 0.3)["feasible"] and not at(esc_k, 0.4)["feasible"] and at(esc_k, 0.25)["feasible"]
    sch = G.series(grid, "grund", "schwelle", "kollaps")
    assert at(sch, 0.2)["obj"] == pytest.approx(0.1400, abs=ZIEL) and at(sch, 0.3)["obj"] == pytest.approx(0.1354, abs=ZIEL) and at(sch, 0.3)["d_vs_aufwand"] == pytest.approx(0.0052, abs=4e-4)
    assert at(sch, 0.375)["d_vs_aufwand"] == pytest.approx(0.0010, abs=4e-4) and at(sch, 0.4)["d_vs_aufwand"] == pytest.approx(-0.0047, abs=4e-4)
    assert all(r["collapse"] <= 0.11 for r in sch)                                                                  # die gedämpfte Schwelle bleibt bis 0,8 bei höchstens 11 % Kollaps


def test_raster_rechnet_eine_zelle_gleich_neu(grid, train, test_paths):
    sc = Scenario(sm=0.2)
    _, k = A.tune("escape", sc, train)
    ev = A.evaluate("escape", k.par, None, sc, test_paths)
    row = next(r for r in G.series(grid, "grund", "escape", "kollaps") if r["sm"] == 0.2)
    assert row["par"] == pytest.approx(k.par, abs=0.02) and row["obj"] == pytest.approx(ev["obj"], abs=ZIEL) and row["collapse"] == pytest.approx(ev["collapse"], abs=KOLL)


def test_raster_ueberschiessen_nur_bei_sigma_g_vier_zehntel_nennenswert(grid):
    """Außer bei σ_g = 0,4 stürzen in den abgestimmten Regeln höchstens 0,5 % der Testpfade durch Überschießen ab; bei σ_g = 0,4 sind es bis 17 % (der Aufwand ohne Fang) und
    bis 10 % (gedämpfte Schwelle bei Messrauschen 0,8). Dort ist der Kollaps zum Teil ein Merkmal der Wachstumsfunktion, kein Ergebnis der Regel."""
    others = [r["overshoot"] for r in grid if r["overshoot"] is not None and r["scen"] != "sg40"]
    assert max(others) <= 0.005
    sg40 = [r for r in grid if r["scen"] == "sg40" and r["overshoot"] is not None]
    assert max(r["overshoot"] for r in sg40) == pytest.approx(0.169, abs=0.02)
    assert max(r["overshoot"] for r in sg40 if r["rule"] == "schwelle") == pytest.approx(0.10, abs=0.02)
    auf = G.series(grid, "sg40", "aufwand", "kollaps")[0]
    assert auf["par"] == 0.0 and not auf["feasible"] and auf["collapse"] > 0.1          # kein Aufwand ist zulässig: gewählt ist e = 0


def test_vorsprung_der_gedaempften_schwelle_gegen_den_aufwand_faellt_mit_der_wachstumsrate(grid):
    expected = {"r02": 70, "r03": 23, "grund": 7.5, "r08": 3.2, "r10": 1.4}          # Prozent bei Messrauschen 0,2, beide mit Kollapsgrenze
    got = {}
    for sid, pct in expected.items():
        sch = next(r for r in G.series(grid, sid, "schwelle", "kollaps") if r["sm"] == 0.2)["obj"]
        auf = next(r for r in G.series(grid, sid, "aufwand", "kollaps") if r["sm"] == 0.2)["obj"]
        got[sid] = 100 * (sch / auf - 1)
        assert got[sid] == pytest.approx(pct, abs=max(1.0, 0.04 * pct)), sid
    assert list(got.values()) == sorted(got.values(), reverse=True)


def test_wachstumsrauschen_vergroessert_den_vorsprung_der_schwelle_vor_dem_aufwand(grid):
    expected = {"sg05": (0.1398, 0.1321, 0.1413), "sg10": (0.1379, 0.1315, 0.1410), "grund": (0.1302, 0.1285, 0.1400), "sg30": (0.0946, 0.1233, 0.1381)}
    for sid, (auf, esc, sch) in expected.items():
        got = [next(r for r in G.series(grid, sid, rule, "kollaps") if r["sm"] == 0.2)["obj"] for rule in ("aufwand", "escape", "schwelle")]
        assert got == pytest.approx([auf, esc, sch], abs=ZIEL), sid
    assert G.series(grid, "sg30", "aufwand", "kollaps")[0]["par"] == pytest.approx(0.13, abs=0.011)


# ---------------------------------------------------------------------------------------------------------------- Texte der App: Schwellenregeln lesen den Messwert, Verdikt zur Kollapsgrenze
def test_schwellenregel_entnimmt_nach_dem_messwert_nicht_nach_dem_bestand():
    """MATH_NOTE: bei den Schwellenregeln entscheidet der Messwert Y, nicht der Bestand B (Beispiel B = 0,30, S* = 0,38, f = 0,7, Y = 0,439 gibt H = 0,041)."""
    h = A.harvest("schwelle", 0.38, 0.7, np.array([0.30]), np.array([0.439]), 0.9)
    assert h[0] == pytest.approx(0.7 * (0.439 - 0.38), abs=1e-12) and h[0] > 0
    assert A.harvest("schwelle", 0.38, 0.7, np.array([0.50]), np.array([0.37]), 0.9)[0] == 0.0       # Messwert unter der Schwelle: nichts, auch wenn B darüber liegt
    from fis_texts import MATH_NOTE
    assert "Messwert $Y_t$ unter der Schwelle liegt" in MATH_NOTE and "Bestand unter die Schwelle fällt" not in MATH_NOTE


def test_verdict_hidden_erfolgsfall_behauptet_nur_den_test_kollaps():
    """Ohne Rauschen (σ_g = σ_m = 0) halten alle vier frei abgestimmten Regeln den Kollaps ein; der Text sagt nur, dass die Kollapsgrenze auf den Testpfaden schon ohne
    Nebenbedingung eingehalten ist, nicht dass sie nichts kostet (gemessen: hier wählen beide Abstimmungen dieselben Reglerwerte)."""
    cmp = EV.compare_rules(Scenario(sg=0.0, sm=0.0))
    kind, text = EV.verdict_hidden(cmp)
    assert kind == "success" and "schon ohne Nebenbedingung eingehalten" in text and "kostet" not in text
    for rule in ("quote", "aufwand", "escape", "schwelle"):
        assert _row(cmp, rule, "frei")["par"] == _row(cmp, rule, "kollaps")["par"], rule


# ---------------------------------------------------------------------------------------------------------------- Handbeispiel
def test_handbeispiel_zahlen_der_readme():
    assert H.next_stock(Fr(1), "quote", Fr(1, 8))[2] == Fr(119, 128)
    assert H.next_stock(Fr(11, 20), "quote", Fr(1, 8))[2] == Fr(1751, 3200)
    assert H.next_stock(Fr(11, 20), "aufwand", Fr(1, 5))[2] == Fr(352, 625)
    assert H.optimum_closed() == (Fr(17, 38), Fr(441, 152)) and H.effort_equilibrium() == (Fr(5, 8), Fr(1, 8))
    path = H.orbit("quote", 0.125, 11 / 20, 101)
    assert path[100][0] < 1e-6 and H.orbit("aufwand", 0.2, 11 / 20, 101)[100][0] == pytest.approx(0.625, abs=1e-6)



# ---------------------------------------------------------------------------------------------------------------- Zustandsraum und Kosten der Wertiteration (README, Methodik)
def test_zustandsraum_und_kosten_der_wertiteration_stehen_in_der_readme_und_im_code():
    """README, Methodik: Gitter in [0, 2] mit Schritt 0,0025, Bestand nach dem Wachstum über 2 wird auf 2 gesetzt, Kosten c ln(B/S) je Jahr bei Preis 1, S = 0 bei B > 0 ausgeschlossen.
    Gerechnet wird die Bellman-Gleichung der Ergebnisse Zeile für Zeile in einer Schleife (kleines Gitter mit bmax = 0,5, damit das Abschneiden wirklich greift)."""
    from pathlib import Path
    import fis_dp as D2  # noqa: F401
    readme = (Path(__file__).resolve().parent.parent / "README.md").read_text(encoding="utf-8")
    assert "gleichmäßigen Gitter in $[0, 2]$ (Schritt 0,0025)" in readme and r"Kosten $c\ln(B/S)$ je Jahr bei Preis 1" in readme
    assert C.DP_BMAX == 2.0 and C.DP_POINTS == 801
    g, _V, _S = D2.value_iteration(0.5, 0.2, 0.95)
    assert g[0] == 0.0 and g[-1] == 2.0 and g[1] - g[0] == pytest.approx(0.0025)
    r, beta, bmax, n = 0.5, 0.9, 0.5, 51
    for cost in (0.0, 0.05):
        g, V, Sp = D2.value_iteration(r, 0.0, beta, n=n, bmax=bmax, cost=cost, tol=1e-13)
        f = g + r * g * (1.0 - g)
        assert f.max() > bmax                                           # der Fall „über bmax“ kommt vor und wird abgeschnitten
        for i in range(n):
            best = -np.inf
            for j in range(i + 1):
                if cost > 0.0 and j == 0 and i > 0:
                    continue                                             # S = 0 bei B > 0 ausgeschlossen
                lr = np.log(g[i] / g[j]) if (cost > 0.0 and i > 0 and j > 0) else 0.0
                best = max(best, (g[i] - g[j]) - cost * lr + beta * np.interp(min(f[j], bmax), g, V))
            assert V[i] == pytest.approx(best, abs=1e-9), (cost, i)


def test_modelldetails_der_readme_zweite_runde_stehen_im_code():
    """README: negatives Wachstum der Wertiteration wirkt wie Bestand 0 (r = 1,5), Abbruch 1e-11 und höchstens 20 000 Durchläufe, Gleichstand der Abstimmung gibt den kleinsten Index,
    Rasterpunkte und erster Vorzeichenwechsel, Standardfehler eines Kollapsanteils, Überschießen bei r = 1,5 (ohne Entnahme 52 %, abgestimmte Regeln 0)."""
    import inspect
    import math
    from pathlib import Path
    import fis_dp as D2
    readme = (Path(__file__).resolve().parent.parent / "README.md").read_text(encoding="utf-8")
    for phrase in ("wie Bestand 0 ($V(0)=0$", "bei Gleichstand gilt der kleinste Index im Gitter", "gezählt wird der erste Wechsel von positiv zu höchstens null",
                   "0 / 0,05 / 0,1 / 0,15 / 0,2 / 0,25 / 0,3 / 0,35 / 0,375 / 0,4 / 0,5 / 0,6 / 0,8", "bei 9,0 % etwa 0,4 Prozentpunkte", "Rest $S_t$ nach der Entnahme negativ"):
        assert phrase in readme, phrase
    sig = inspect.signature(D2.value_iteration).parameters
    assert sig["tol"].default == 1e-11 and sig["maxit"].default == 20000
    # negatives Wachstum (r = 1,5, Rest über 1 + 1/r) wirkt wie Bestand 0: Bellman-Gleichung mit max{0, ·}
    r, beta, n = 1.5, 0.9, 41
    g, V, _S = D2.value_iteration(r, 0.0, beta, n=n, bmax=2.0, tol=1e-13)
    f = g + r * g * (1.0 - g)
    assert f.min() < 0.0 and g[-1] > 1 + 1 / r
    for i in range(n):
        best = max((g[i] - g[j]) + beta * np.interp(max(0.0, min(f[j], 2.0)), g, V) for j in range(i + 1))
        assert V[i] == pytest.approx(best, abs=1e-9), i
    # Gleichstand: der kleinste Index; gedämpfte Schwelle zuerst nach S*, dann nach f
    assert A.pick(np.array([1.0, 1.0, 0.5]), np.array([0.0, 0.0, 0.0]), None)[0] == 0
    ss, ff = A.tuning_grid("schwelle")
    assert ss[0] == ss[1] and ff[0] < ff[1] and ss[0] < ss[-1]
    # Raster: die 13 Rasterpunkte, erster Vorzeichenwechsel
    assert C.GRID_SM == (0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.375, 0.4, 0.5, 0.6, 0.8) and len(C.GRID_SM) == 13
    assert EV.kipp([0.0, 1.0, 2.0, 3.0], [0.5, -0.5, 0.5, -0.5]) == pytest.approx(0.5)
    assert math.sqrt(0.09 * 0.91 / 5000) == pytest.approx(0.004, abs=5e-5)


def test_ueberschiessen_bei_r_eins_komma_fuenf_ohne_entnahme_und_mit_abgestimmten_regeln(train, test_paths):
    """README: bei r = 1,5 stürzen ohne Entnahme 52 % der Testpfade durch Überschießen auf 0, mit dem abgestimmten Aufwand und der abgestimmten gedämpften Schwelle keiner."""
    sc = Scenario(r=1.5)
    assert A.evaluate("quote", 0.0, None, sc, test_paths)["overshoot"] == pytest.approx(0.52, abs=0.01)
    for rule in ("aufwand", "schwelle"):
        assert _capped(rule, sc, train, test_paths)["overshoot"] <= 0.001, rule
