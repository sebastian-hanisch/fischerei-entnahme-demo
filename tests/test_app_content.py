"""Inhalt der App: Tabs, Kernfragen, Expander (per Titel, nie per Index), Handbeispiel, PDF-Knopf, Formelblöcke im Quelltext, Gleichheit von App- und README-Text,
je Preset die erwartete Verdikt-Zeile der ersten Kernfrage (DEMO-PLAYBOOK §11a Punkt 7)."""

import ast
import re
from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

import fis_constants as C
import fis_dp as D
import fis_texts as TX

pytestmark = pytest.mark.usefixtures("fast_limits")
ROOT = Path(__file__).resolve().parent.parent
APP = str(ROOT / "app.py")


@pytest.fixture(scope="module")
def at():
    a = AppTest.from_file(APP, default_timeout=300)
    a.run()
    assert not a.exception, [str(e) for e in a.exception]
    return a


def _texts(a):
    return [e.value for e in list(a.success) + list(a.info) + list(a.warning)]


def test_sechs_tabs_in_fester_reihenfolge(at):
    assert [t.label for t in at.tabs] == [
        "🐟 Bestand und Entnahme", "⚖️ Regeln im fairen Vergleich", "🔀 Wo kippt es?", "🎯 Das Optimum (Wertiteration)", "✋ Von Hand", "⚠️ Annahmen und Grenzen",
    ]


def test_expander_stehen_unter_ihren_titeln(at):
    labels = {e.label for e in at.expander}
    assert {
        "💬 In einfachen Worten",
        "🔧 Wie wir das erreichen – vollständiger Methodenvergleich",
        "📐 Mathematische Formulierung",
        "⚠️ Annahmen, Grenzen und bekannte Probleme",
    } <= labels


def test_drei_kernfragen_haben_ein_verdikt(at):
    texts = _texts(at)
    assert any("fällt der Bestand in" in t and "Pfade irgendwann" in t for t in texts)
    assert any("frei abgestimmten Regeln" in t for t in texts)
    assert any("Bei Messrauschen" in t and "Escapement" in t for t in texts)
    heads = [m.value for m in at.markdown]
    for q in ("**Hält die gewählte Regel den Bestand?**", "**Versteckt das Ziel allein den Kollaps?**", "**Escapement oder fester Aufwand: wer liegt bei diesem Messrauschen vorn?**"):
        assert q in heads


def test_tab_eins_zeigt_vier_kennzahlen_und_beide_abbildungen(at):
    labels = [m.label for m in at.metric]
    for want in ("Ziel: gleichwertiger Jahresfang", "Kollaps", "Fang-Streuung", "Mittlerer Bestand"):
        assert want in labels


def test_tab_zwei_hat_referenzzeile_und_alle_regeln(at):
    table = next(d.value for d in at.dataframe if "Anteil der Referenz (%)" in d.value.columns)
    assert table["Regel"].iloc[0].startswith("Referenz")
    rules = list(table["Regel"].iloc[1:])
    assert {"Quote", "Aufwand", "Escapement-Schwelle", "Gedämpfte Schwelle", "DP-Politik auf dem Messwert"} <= set(rules)
    assert len(table) == 1 + 4 * 2 + 3 + 1                  # Referenz, vier Regeln je frei und mit Kollapsgrenze, drei Lehrbuch-Werte, DP-Politik


def test_standardzahlen_der_app_mit_den_echten_gittern(monkeypatch):
    """Eigener Lauf mit den echten Abstimmungsgittern und dem vollen Gitter der Wertiteration (die übrigen Tests der Datei rechnen grob): die Zeilen des Vergleichs im Grundfall
    gegen die README (Band; Zufalls-Seed 11 und 22, 1 000 und 5 000 Pfade)."""
    monkeypatch.setattr(C, "TUNE_STRIDE", 1)
    monkeypatch.setattr(C, "DP_POINTS", 801)
    st.cache_data.clear()
    D._CACHE.clear()
    try:
        a = AppTest.from_file(APP, default_timeout=300)
        a.run()
        assert not a.exception, [str(e) for e in a.exception]
        table = next(d.value for d in a.dataframe if "Anteil der Referenz (%)" in d.value.columns)
        ref = table["Ziel (K je Jahr)"].iloc[0]
        by = {(r["Regel"], r["Abstimmung"].split(" ")[0]): r for _, r in table.iloc[1:].iterrows()}
    finally:
        st.cache_data.clear()
        D._CACHE.clear()
    assert ref == pytest.approx(0.1448, abs=5e-4)
    assert by[("Escapement-Schwelle", "frei")]["Ziel (K je Jahr)"] == pytest.approx(0.1374, abs=1e-3)
    assert by[("Gedämpfte Schwelle", "frei")]["Ziel (K je Jahr)"] == pytest.approx(0.1400, abs=1e-3)
    assert by[("Quote", "frei")]["Kollaps (%)"] > 95.0


def test_handbeispiel_zeigt_den_exakten_bruch_und_reagiert():
    a = AppTest.from_file(APP, default_timeout=300)
    a.run()
    steps = next(d.value for d in a.dataframe if "Bestand B_t+1" in d.value.columns)
    assert list(steps["Bestand B_t+1"])[0] == "119/128"                  # Quote 1/8 von B_0 = 1
    a.selectbox(key="hand_start").set_value("11/20 (= 0,55)").run()
    steps = next(d.value for d in a.dataframe if "Bestand B_t+1" in d.value.columns)
    assert list(steps["Bestand B_t+1"])[0] == "1751/3200"
    a.selectbox(key="hand_rule").set_value("Aufwand e = 1/5").run()
    steps = next(d.value for d in a.dataframe if "Bestand B_t+1" in d.value.columns)
    assert list(steps["Bestand B_t+1"])[0] == "352/625"                  # B_0 = 11/20, H = 11/100, S = 11/25, B_1 = 11/25 + (1/2) * (11/25) * (14/25)
    texts = " ".join(m.value for m in a.markdown)
    assert "441/152" in texts and "17/38" in texts


def test_ohne_diskontierung_zeigt_das_optimum_einen_hinweis():
    a = AppTest.from_file(APP, default_timeout=300)
    a.run()
    a.select_slider(key="beta_select").set_value(1.0).run()
    assert not a.exception
    assert any("braucht β unter 1" in e.value for e in a.info)
    assert not any("Schwelle S* der Wertiteration" == m.label for m in a.metric)


def test_kostenregler_der_wertiteration_aendert_die_schwelle():
    a = AppTest.from_file(APP, default_timeout=300)
    a.run()
    s0 = next(m.value for m in a.metric if m.label == "Schwelle S* der Wertiteration")
    a.select_slider(key="dp_cost").set_value(0.1).run()
    assert not a.exception
    s1 = next(m.value for m in a.metric if m.label == "Schwelle S* der Wertiteration")
    assert float(s1.split()[0]) > float(s0.split()[0])
    assert any("gilt die geschlossene Formel nicht" in e.value for e in a.info)


def test_raster_tab_wechselt_die_einstellung_ohne_fehler():
    a = AppTest.from_file(APP, default_timeout=300)
    a.run()
    for sid, *_ in C.GRID_SCENARIOS:
        a.selectbox(key="grid_scen").set_value(sid).run()
        assert not a.exception, (sid, [str(e) for e in a.exception])
    a.radio(key="grid_mode").set_value("frei").run()
    assert not a.exception


@pytest.mark.parametrize("name, kind, fragment", [
    ("Grundfall (Voreinstellung)", "success", "höchstens 10 %"),
    ("Lehrbuch: Quote am MSY", "info", "mehr als die 10 %"),
    ("Aufwand stabil", "success", "höchstens 10 %"),
    ("Escapement bei exaktem Survey", "success", "höchstens 10 %"),
    ("Verrauschter Survey", "warning", "die Mehrheit der Pfade"),
    ("Schneller Bestand (r = 1)", "success", "höchstens 10 %"),
])
def test_jedes_preset_zeigt_die_erwartete_zeile_der_ersten_kernfrage(name, kind, fragment):
    """Messung mit 500 Pfaden (Zufalls-Seed 5): Grundfall 7,0 %, Lehrbuch-Quote 39,8 %, Aufwand 1,0 %, Escapement bei exaktem Survey 0 %, verrauschter Survey 92,8 %, r = 1: 0 %."""
    a = AppTest.from_file(APP, default_timeout=300)
    a.run()
    a.sidebar.button(key=f"preset_{name}").click().run()
    assert not a.exception
    hits = [e.value for e in getattr(a, kind) if "fällt der Bestand in" in e.value]
    assert hits and fragment in hits[0], (name, [e.value for e in list(a.success) + list(a.info) + list(a.warning)][:6])


def test_pdf_knopf_ist_da(at):
    assert any("download" in str(type(el)).lower() for el in at.main)


def test_jede_abgesetzte_formel_steht_in_einem_eigenen_markdown_aufruf_im_quelltext():
    """Ein `$$ … $$`-Block je st.markdown-Aufruf, wörtlich in app.py (nicht aus einer Variablen), und KaTeX bekommt kein ungeschütztes Prozentzeichen."""
    tree = ast.parse((ROOT / "app.py").read_text(encoding="utf-8"))
    blocks = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "markdown" and node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
            s = node.args[0].value
            if "$$" in s:
                blocks += 1
                assert s.count("$$") == 2 and s.strip().startswith("$$") and s.strip().endswith("$$"), s[:60]
                assert not re.search(r"(?<!\\)%", s), s[:60]
    assert blocks == 15                                    # die 15 Formeln des 📐-Expanders; Zählung aus dem Quelltext


def test_annahmentext_der_app_steht_in_der_readme():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert TX.ASSUMPTIONS_MARKDOWN.strip() in readme
