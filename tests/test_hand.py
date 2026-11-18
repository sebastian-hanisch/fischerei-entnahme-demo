"""Handbeispiel in exakten Brüchen (fis_hand.py): jede Zahl von Hand nachgerechnet, nicht aus dem Code abgeschrieben."""

from fractions import Fraction as Fr

import pytest

import fis_hand as H


def test_quote_von_elf_zwanzigstel_von_hand():
    """B_0 = 11/20, Q = 1/8: H = 1/8 (9/10 B = 99/200 ist größer), S = 11/20 - 1/8 = 17/40, B_1 = 17/40 + 1/2 * 17/40 * 23/40 = 17/40 + 391/3200 = 1751/3200."""
    h, s, b1 = H.next_stock(Fr(11, 20), "quote", Fr(1, 8))
    assert h == Fr(1, 8) and s == Fr(17, 40) and b1 == Fr(17, 40) + Fr(391, 3200) == Fr(1751, 3200)
    assert float(b1) == pytest.approx(0.54719, abs=1e-5)        # unter 0,55: der Bestand fällt


def test_aufwand_von_start_eins_von_hand():
    """B_0 = 1, e = 1/5: H = 1/5, S = 4/5, B_1 = 4/5 + 1/2 * 4/5 * 1/5 = 4/5 + 2/25 = 22/25."""
    h, s, b1 = H.next_stock(Fr(1), "aufwand", Fr(1, 5))
    assert (h, s, b1) == (Fr(1, 5), Fr(4, 5), Fr(22, 25))


def test_escapement_von_start_eins_von_hand():
    """B_0 = 1, S* = 9/20: H = 11/20 (unter 9/10), S = 9/20, B_1 = 9/20 + 1/2 * 9/20 * 11/20 = 9/20 + 99/800 = 459/800; im nächsten Jahr H = 459/800 - 9/20 = 99/800."""
    h, s, b1 = H.next_stock(Fr(1), "escape", Fr(9, 20))
    assert (h, s, b1) == (Fr(11, 20), Fr(9, 20), Fr(459, 800))
    assert H.next_stock(b1, "escape", Fr(9, 20))[0] == Fr(99, 800)


def test_kapazitaetsgrenze_greift_bei_kleinem_bestand():
    """B = 1/10 und Q = 1/8: der Bestand reicht nicht, H = min(1/8, 1/10) = 1/10, danach höchstens 9/10 * 1/10 = 9/100."""
    assert H.harvest("quote", Fr(1, 8), Fr(1, 10)) == Fr(9, 100)
    assert H.harvest("escape", Fr(9, 20), Fr(2, 5)) == Fr(0)        # unter der Schwelle: nichts


def test_gleichgewicht_und_fixpunkte():
    b_star, h_star = H.effort_equilibrium()
    assert (b_star, h_star) == (Fr(5, 8), Fr(1, 8))
    # Fixpunkt des Aufwands: derselbe Bestand nach einem Jahr
    assert H.next_stock(b_star, "aufwand", Fr(1, 5))[2] == b_star
    # Fixpunkt der Quote: derselbe Bestand nach einem Jahr, Fang 1/8
    h, _, b_next = H.next_stock(b_star, "quote", Fr(1, 8))
    assert h == Fr(1, 8) and b_next == b_star


def test_umlaufbahn_in_bruechen_wird_exakt_und_gleitend_gleich_gerechnet():
    exact = H.orbit("aufwand", Fr(1, 5), Fr(1), 5)
    approx = H.orbit("aufwand", 0.2, 1.0, 5)
    for (be, he), (bf, hf) in zip(exact, approx):
        assert float(be) == pytest.approx(bf, rel=1e-12) and float(he) == pytest.approx(hf, rel=1e-12)
    assert exact[1][0] == Fr(22, 25)


def test_optimum_in_bruechen():
    """r = 1/2, β = 19/20: 1/β - 1 = 1/19, durch r = 2/19, S* = (1 - 2/19)/2 = 17/38; V(1) = 21/38 + 19 * (1/2) * (17/38) * (21/38) = 21/38 + 357/152 = 441/152."""
    s, v = H.optimum_closed()
    assert s == Fr(17, 38)
    assert v == Fr(21, 38) + 19 * Fr(1, 2) * Fr(17, 38) * Fr(21, 38) == Fr(441, 152)
    assert H.annual_value() == Fr(1, 20) * Fr(441, 152) == Fr(441, 3040)
    assert float(H.annual_value()) == pytest.approx(0.14507, abs=1e-5)


def test_unbekannte_regel_wird_abgelehnt():
    with pytest.raises(ValueError):
        H.harvest("zufall", Fr(1), Fr(1))
