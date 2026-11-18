"""Gemeinsame Fixtures: Szenarien und die Mini-Instanz, an der jede Einheit von Hand geprüft wird (DEMO-PLAYBOOK §1b)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import fis_constants as C  # noqa: E402
import fis_dp as D  # noqa: E402
from fis_scenario import Scenario  # noqa: E402


@pytest.fixture(scope="session")
def grundfall():
    """Der Grundfall der App: r = 0,5, σ_g = 0,2, σ_m = 0,2, β = 0,95, Kapazitätsgrenze 0,9, Kollapsgrenze 0,1 K."""
    return Scenario()


@pytest.fixture(scope="session")
def ohne_zufall():
    """Mini-Szenario ohne Zufall und ohne Kapazitätsgrenze: r = 1/2, exakter Survey, kein Wachstumsrauschen (Handrechnung in tests/test_hand.py)."""
    return Scenario(r=0.5, sg=0.0, sm=0.0, beta=0.95, hmax=1.0, thr=0.1)


@pytest.fixture(scope="module")
def fast_limits():
    """Für App-Tests: gröbere Abstimmungsgitter und ein kleineres Gitter der Wertiteration, damit ein Lauf der App Sekunden dauert. Modul-Umfang:
    test_claims.py und die Orakeltests rechnen mit den echten Gittern. Der Zwischenspeicher von Streamlit und der Wertiteration wird davor und danach geleert,
    damit grobe Ergebnisse nie in andere Tests geraten."""
    import streamlit as st

    mp = pytest.MonkeyPatch()
    mp.setattr(C, "TUNE_STRIDE", 4)
    mp.setattr(C, "DP_POINTS", 201)
    st.cache_data.clear()
    D._CACHE.clear()
    yield
    mp.undo()
    st.cache_data.clear()
    D._CACHE.clear()
