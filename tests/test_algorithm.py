"""Kleine Einheiten des Kerns gegen Handrechnung (DEMO-PLAYBOOK §1b): Messung, Wachstum, jede Entnahmeregel, Gewichte des Ziels, Simulation einer Mini-Instanz
ohne Zufall gegen exakte Brüche, Abstimmung (Auswahl des besten Reglers) und gepaarte Differenz. Zufallsabhängiges gegen unabhängige Rechenwege steht in test_oracle_*.py."""

from fractions import Fraction as Fr

import numpy as np
import pytest

import fis_algorithm as A
import fis_constants as C
from fis_scenario import Scenario, noise


def _zeros(years, paths=1):
    return np.zeros((years, paths)), np.zeros((years, paths))


# ----------------------------------------------------------------------------------------------------------------- Messung und Wachstum
def test_measure_factor_ist_eins_ohne_messrauschen_und_hat_mittel_eins():
    assert np.all(A.measure_factor(np.array([-2.0, 0.0, 3.0]), 0.0) == 1.0)
    xi = np.random.default_rng(1).standard_normal(400_000)
    f = A.measure_factor(xi, 0.3)
    assert f.mean() == pytest.approx(1.0, abs=4 * f.std() / np.sqrt(xi.size))
    assert A.measure_factor(np.array([0.0]), 0.5)[0] == pytest.approx(np.exp(-0.125))  # xi = 0: exp(-sm^2/2)


def test_growth_noise_hat_mittel_eins_und_ohne_rauschen_genau_eins():
    assert np.all(A.growth_noise(np.array([1.5, -1.5]), 0.0) == 1.0)
    z = A.growth_noise(np.random.default_rng(2).standard_normal(400_000), 0.4)
    assert z.mean() == pytest.approx(1.0, abs=4 * z.std() / np.sqrt(z.size))


def test_growth_von_hand():
    assert A.growth(0.5, 0.5) == pytest.approx(0.625)            # 0,5 + 0,5 * 0,5 * 0,5
    assert A.growth(1.0, 0.5) == pytest.approx(1.0)              # bei K gibt es kein Wachstum
    assert A.growth(0.0, 0.5) == 0.0
    assert A.growth(0.4, 1.0, z=2.0) == pytest.approx(2.0 * (0.4 + 0.4 * 0.6))
    assert A.growth_raw(0.5, 0.5) == pytest.approx(0.625)


def test_growth_wird_nie_negativ_und_ist_an_der_nullstelle_null():
    # f(S) = S + r S (1 - S) hat die Nullstelle S = 1 + 1/r: für r = 1/2 bei S = 3
    assert A.growth_raw(3.0, 0.5) == pytest.approx(0.0, abs=1e-15)
    assert A.growth_raw(3.5, 0.5) < 0.0 and A.growth(3.5, 0.5, z=2.0) == 0.0     # Überschießen: Absturz auf 0, nicht negativ
    assert A.growth(2.9, 0.5) == pytest.approx(2.9 + 0.5 * 2.9 * (1 - 2.9)) and A.growth(2.9, 0.5) > 0


# ----------------------------------------------------------------------------------------------------------------- Entnahmeregeln
@pytest.mark.parametrize(
    "rule, par, par2, B, Y, hmax, expected",
    [
        ("quote", 0.1, None, 0.8, 0.8, 1.0, 0.1),
        ("quote", 0.1, None, 0.05, 0.05, 1.0, 0.05),        # mehr als der Bestand geht nicht
        ("quote", 0.1, None, 0.05, 0.05, 0.9, 0.045),       # Kapazitätsgrenze 0,9 B
        ("quote", 0.1, None, 0.8, 5.0, 1.0, 0.1),           # der Survey wird nicht gelesen
        ("aufwand", 0.25, None, 0.8, 0.8, 1.0, 0.2),
        ("aufwand", 0.25, None, 0.8, 9.0, 1.0, 0.2),
        ("aufwand", 0.95, None, 0.8, 0.8, 0.9, 0.72),       # Kapazitätsgrenze schneidet ab
        ("escape", 0.45, None, 0.8, 0.9, 1.0, 0.45),        # Y - S* = 0,45
        ("escape", 0.45, None, 0.8, 1.5, 1.0, 0.8),         # Y - S* = 1,05 > B: höchstens der ganze Bestand
        ("escape", 0.45, None, 0.8, 1.5, 0.9, 0.72),        # ... und höchstens h_max B
        ("escape", 0.45, None, 0.8, 0.3, 1.0, 0.0),         # Messwert unter der Schwelle: keine Entnahme
        ("escape", 0.45, None, 0.8, 0.45, 1.0, 0.0),        # genau auf der Schwelle: keine Entnahme
        ("schwelle", 0.4, 0.5, 0.8, 1.0, 1.0, 0.3),         # 0,5 * (1,0 - 0,4)
        ("schwelle", 0.4, 0.5, 0.8, 0.3, 1.0, 0.0),
        ("schwelle", 0.0, 1.0, 0.5, 0.5, 1.0, 0.5),         # S* = 0, f = 1: alles
    ],
)
def test_entnahmeregeln_von_hand(rule, par, par2, B, Y, hmax, expected):
    assert A.harvest(rule, par, par2, np.array(B), np.array(Y), hmax) == pytest.approx(expected)


def test_regel_opt_nutzt_die_politik_auf_dem_messwert():
    grid = np.array([0.0, 0.5, 1.0, 2.0])
    esc = np.array([0.0, 0.4, 0.4, 0.4])          # Rest S(Y): bis 0,4 gleich Y (hier 0 und 0,4), darüber 0,4
    # Y = 1,0: Rest 0,4, Entnahme 0,6; bei B = 0,5 höchstens der Bestand (0,5), mit h_max 0,9 höchstens 0,45
    assert A.harvest("opt", 0.0, None, np.array(2.0), np.array(1.0), 1.0, (grid, esc)) == pytest.approx(0.6)
    assert A.harvest("opt", 0.0, None, np.array(0.5), np.array(1.0), 1.0, (grid, esc)) == pytest.approx(0.5)
    assert A.harvest("opt", 0.0, None, np.array(0.5), np.array(1.0), 0.9, (grid, esc)) == pytest.approx(0.45)
    # Y = 0,25: der Rest wird zwischen den Gitterpunkten linear gelesen: S = 0,2 -> Entnahme 0,05
    assert A.harvest("opt", 0.0, None, np.array(1.0), np.array(0.25), 1.0, (grid, esc)) == pytest.approx(0.05)


def test_unbekannte_regel_und_ungueltige_kapazitaetsgrenze_werden_abgelehnt():
    with pytest.raises(ValueError):
        A.harvest("zufall", 0.1, None, np.array(1.0), np.array(1.0), 1.0)
    for bad in (0.0, 1.01):
        with pytest.raises(ValueError):
            A.simulate("aufwand", [0.1], Scenario(hmax=bad), _zeros(3))


# ----------------------------------------------------------------------------------------------------------------- Gewichte des Ziels
def test_gewichte_des_ziels_summieren_sich_zu_eins_und_fallen_geometrisch():
    w = A.discount_weights(0.95, 3)
    assert w.sum() == pytest.approx(1.0)
    assert w[0] == pytest.approx(0.05 / (1 - 0.95 ** 3)) and w[1] / w[0] == pytest.approx(0.95) and w[2] / w[1] == pytest.approx(0.95)
    assert A.discount_weights(0.9, 200).sum() == pytest.approx(1.0)


# ----------------------------------------------------------------------------------------------------------------- Mini-Instanz gegen Brüche
def _fractions_orbit(rule, par, years):
    """Unabhängige Handrechnung in Brüchen: r = 1/2, K = 1, B_0 = 1, ohne Zufall und ohne Kapazitätsgrenze. Rückgabe: ([B_0 ... B_T], [H_0 ... H_{T-1}])."""
    B, bs, hs = Fr(1), [Fr(1)], []
    for _ in range(years):
        if rule == "quote":
            H = min(par, B)
        elif rule == "aufwand":
            H = par * B
        else:
            H = max(Fr(0), B - par)
        S = B - H
        B = S + Fr(1, 2) * S * (1 - S)
        hs.append(H)
        bs.append(B)
    return bs, hs


@pytest.mark.parametrize("rule, par", [("quote", Fr(1, 8)), ("aufwand", Fr(1, 5)), ("escape", Fr(9, 20))])
def test_simulation_stimmt_mit_exakten_bruechen_ueberein(ohne_zufall, rule, par):
    years = 3
    bs, hs = _fractions_orbit(rule, par, years)
    res = A.simulate(rule, [float(par)], ohne_zufall, _zeros(years), record=True)
    assert [float(b) for b in bs] == pytest.approx(list(res["B"][:, 0, 0]), rel=1e-12)
    assert [float(h) for h in hs] == pytest.approx(list(res["H"][:, 0, 0]), rel=1e-12)
    beta = Fr(19, 20)
    expected = (1 - beta) / (1 - beta ** years) * sum(beta ** t * h for t, h in enumerate(hs))   # Ziel nach der Formel aus der README
    assert res["obj"][0, 0] == pytest.approx(float(expected), rel=1e-12)


def test_mini_instanz_zahlen_von_hand():
    """Jahr 0 bei B_0 = 1: Quote 1/8 gibt B_1 = 7/8 + 7/128 = 119/128, Aufwand 1/5 gibt 0,88, Escapement 9/20 gibt 0,45 + 0,12375 = 0,57375."""
    assert _fractions_orbit("quote", Fr(1, 8), 1)[0][1] == Fr(119, 128)
    assert _fractions_orbit("aufwand", Fr(1, 5), 1)[0][1] == Fr(22, 25)
    assert _fractions_orbit("escape", Fr(9, 20), 1)[0][1] == Fr(57375, 100000)


def test_ziel_bei_beta_eins_ist_das_mittel_der_jahre_ab_der_einschwingzeit():
    sc = Scenario(r=0.5, sg=0.0, sm=0.0, beta=1.0, hmax=1.0)
    years = C.BURN + 4
    res = A.simulate("escape", [0.45], sc, _zeros(years), record=True)
    assert res["obj"][0, 0] == pytest.approx(res["H"][C.BURN:, 0, 0].mean())
    assert res["avgcatch"][0, 0] == pytest.approx(res["H"][C.BURN:, 0, 0].mean())      # dasselbe Fenster: Jahre BURN bis T - 1
    assert res["meanB"][0, 0] == pytest.approx(res["B"][C.BURN:years, 0, 0].mean())    # B_t vor der Entnahme, Jahre BURN bis T - 1


# ----------------------------------------------------------------------------------------------------------------- Simulation: Aufbau
def test_leichter_und_voller_lauf_geben_dasselbe_ziel_und_denselben_kollaps(grundfall):
    nz = noise(3, C.T_YEARS, 50)
    full = A.simulate("escape", [0.4, 0.6], grundfall, nz)
    light = A.simulate("escape", [0.4, 0.6], grundfall, nz, light=True)
    assert np.array_equal(full["obj"], light["obj"]) and np.array_equal(full["collapse"], light["collapse"])
    assert "meanB" not in light and "meanB" in full


def test_mehrere_reglerwerte_auf_einmal_gleichen_einzelnen_laeufen(grundfall):
    nz = noise(4, C.T_YEARS, 40)
    many = A.simulate("schwelle", [0.3, 0.5], grundfall, nz, par2=[0.5, 0.9])
    for i, (p, p2) in enumerate(((0.3, 0.5), (0.5, 0.9))):
        one = A.simulate("schwelle", [p], grundfall, nz, par2=[p2])
        assert np.array_equal(many["obj"][i], one["obj"][0]) and np.array_equal(many["collapse"][i], one["collapse"][0])


def test_aufzeichnung_hat_die_erwarteten_formen(grundfall):
    nz = noise(5, 30, 7)
    res = A.simulate("aufwand", [0.2], grundfall, nz, record=True)
    assert res["B"].shape == (31, 1, 7) and res["H"].shape == (30, 1, 7) and res["Y"].shape == (30, 1, 7)
    assert np.all(res["B"][0] == C.B0)
    # B_{t+1} folgt aus B_t, H_t und dem Wachstumsrauschen: Nachrechnen für einen Pfad
    zg = A.growth_noise(nz[0][4, 3], grundfall.sg)
    b4, h4 = res["B"][4, 0, 3], res["H"][4, 0, 3]
    assert res["B"][5, 0, 3] == pytest.approx(zg * ((b4 - h4) + 0.5 * (b4 - h4) * (1 - (b4 - h4))))


# ----------------------------------------------------------------------------------------------------------------- Abstimmung
def test_pick_ohne_nebenbedingung_nimmt_das_groesste_ziel_und_bei_gleichstand_den_ersten():
    assert A.pick(np.array([0.1, 0.3, 0.3, 0.2]), np.array([0.0, 0.5, 0.0, 0.0]), None) == (1, True)


def test_pick_mit_kollapsgrenze_waehlt_unter_den_zulaessigen_inklusive_der_grenze():
    obj = np.array([0.9, 0.5, 0.4, 0.3])
    col = np.array([0.5, 0.10, 0.04, 0.0])
    assert A.pick(obj, col, 0.10) == (1, True)             # genau 0,10 ist noch zulässig (<=)
    assert A.pick(obj, col, 0.04) == (2, True)
    assert A.pick(obj, col, 0.0) == (3, True)


def test_pick_ohne_zulaessigen_wert_nimmt_das_kleinste_kollapsrisiko_und_meldet_es():
    obj = np.array([0.9, 0.5, 0.4])
    col = np.array([0.5, 0.2, 0.2])
    assert A.pick(obj, col, 0.10) == (1, False)            # kleinstes Risiko 0,2, bei Gleichstand der erste Index


def test_abstimmungsgitter_haben_die_dokumentierte_groesse():
    assert len(A.tuning_grid("quote")[0]) == 81 and len(A.tuning_grid("aufwand")[0]) == 91 and len(A.tuning_grid("escape")[0]) == 101
    s, f = A.tuning_grid("schwelle")
    assert len(s) == len(f) == 51 * 10 and f.min() == pytest.approx(0.1) and f.max() == pytest.approx(1.0) and s.max() == pytest.approx(1.0)
    assert A.tuning_grid("quote")[0][-1] == pytest.approx(0.4) and A.tuning_grid("aufwand")[0][-1] == pytest.approx(0.9)
    with pytest.raises(ValueError):
        A.tuning_grid("opt")


def test_tune_gibt_zwei_abstimmungen_aus_einem_lauf(grundfall):
    train = noise(11, C.T_YEARS, 100)
    free, capped = A.tune("aufwand", grundfall, train)
    assert free.rule == "aufwand" and free.par2 is None
    assert free.train_obj >= capped.train_obj - 1e-12     # die Nebenbedingung kann das Ziel nur senken
    assert capped.train_collapse <= C.COLLAPSE_CAP or not capped.feasible


# ----------------------------------------------------------------------------------------------------------------- Bewertung
def test_paired_difference_von_hand():
    d, se = A.paired_difference([1.0, 2.0, 3.0], [0.0, 2.0, 5.0])       # Differenzen 1, 0, -2
    assert d == pytest.approx(-1 / 3)
    sd = np.std([1.0, 0.0, -2.0], ddof=1)
    assert se == pytest.approx(sd / np.sqrt(3))


def test_evaluate_stimmt_mit_dem_pfadweisen_ziel_ueberein(grundfall):
    nz = noise(8, C.T_YEARS, 60)
    ev = A.evaluate("aufwand", 0.2, None, grundfall, nz)
    res = A.simulate("aufwand", [0.2], grundfall, nz)
    assert ev["obj"] == pytest.approx(res["obj"][0].mean()) and ev["se"] == pytest.approx(res["obj"][0].std(ddof=1) / np.sqrt(60))
    assert ev["collapse"] == pytest.approx(res["collapse"][0].mean()) and np.array_equal(ev["paths"], res["obj"][0])
