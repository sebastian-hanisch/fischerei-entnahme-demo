"""Vorgerechnetes Raster (data/kipp.csv, fis_grid.py): Form, Lesen, Eigenschaften, die der Aufbau erzwingt. Die Zahlen des Rasters prüft test_claims.py."""

import pytest

import fis_constants as C
import fis_grid as G


@pytest.fixture(scope="module")
def grid():
    rows = G.load()
    assert rows, "data/kipp.csv fehlt: python fis_grid.py"
    return rows


def test_raster_hat_jede_einstellung_und_jedes_messrauschen_mit_allen_regeln(grid):
    per = {(r["scen"], r["rule"], r["mode"]) for r in grid}
    for sid, *_ in C.GRID_SCENARIOS:
        for rule, mode in (("aufwand", "frei"), ("aufwand", "kollaps"), ("escape", "frei"), ("escape", "kollaps"), ("schwelle", "frei"), ("schwelle", "kollaps"), ("opt", "dp"), ("ref", "dp")):
            assert (sid, rule, mode) in per, (sid, rule, mode)
            assert [r["sm"] for r in G.series(grid, sid, rule, mode)] == list(C.GRID_SM)
    assert len(grid) == len(C.GRID_SCENARIOS) * len(C.GRID_SM) * 8


def test_der_aufwand_liest_den_survey_nicht_seine_zeilen_sind_ueber_das_messrauschen_gleich(grid):
    """Per Konstruktion (Nullspalte kein Fehler): Aufwand und Quote lesen den Survey nicht; die Zeilen mit Aufwand sind über σ_m identisch, ihre Differenz zum Aufwand ist 0."""
    for sid, *_ in C.GRID_SCENARIOS:
        for mode in ("frei", "kollaps"):
            rows = G.series(grid, sid, "aufwand", mode)
            assert len({(r["par"], r["obj"], r["collapse"], r["feasible"]) for r in rows}) == 1
            assert all(r["d_vs_aufwand"] == 0.0 for r in rows)


def test_nullspalten_im_raster_sind_echt_der_kollaps_hat_auch_von_null_verschiedene_werte(grid):
    """Gegenprobe zur Nullspalte: Escapement bei exaktem Survey kollabiert nie (echte Null), beim größten Messrauschen fast immer; kein Zweig ist tot."""
    first, last = G.series(grid, "grund", "escape", "frei")[0], G.series(grid, "grund", "escape", "frei")[-1]
    assert first["sm"] == 0.0 and first["collapse"] == 0.0 and last["collapse"] > 0.9


def test_mit_kollapsgrenze_ist_das_ziel_nie_groesser_als_frei_auf_den_trainingspfaden_und_die_zeilen_sind_konsistent(grid):
    for sid, *_ in C.GRID_SCENARIOS:
        for rule in ("escape", "schwelle"):
            free, capped = G.series(grid, sid, rule, "frei"), G.series(grid, sid, rule, "kollaps")
            for f, c in zip(free, capped):
                assert f["par"] is not None and 0.0 <= f["par"] <= 1.0 and 0.0 <= c["par"] <= 1.0
                assert 0.0 <= f["collapse"] <= 1.0 and 0.0 <= c["collapse"] <= 1.0 and f["extinct"] <= f["collapse"] + 1e-12
                if f["par"] == c["par"] and f["par2"] == c["par2"]:
                    assert f["obj"] == c["obj"] and f["collapse"] == c["collapse"]        # gleicher Reglerwert, gleiche Pfade, gleiche Zahlen


def test_referenz_der_wertiteration_ist_je_einstellung_konstant_und_positiv(grid):
    for sid, *_ in C.GRID_SCENARIOS:
        refs = {r["obj"] for r in G.series(grid, sid, "ref", "dp")}
        assert len(refs) == 1 and min(refs) > 0.0


def test_series_sortiert_nach_messrauschen_und_fehlende_datei_gibt_leere_liste(grid, tmp_path):
    xs = [r["sm"] for r in G.series(list(reversed(grid)), "grund", "escape", "frei")]
    assert xs == sorted(xs)
    assert G.load(str(tmp_path / "gibt_es_nicht.csv")) == []


def test_szenarien_haben_eindeutige_kennungen_und_der_grundfall_steht_zuerst():
    ids = [c[0] for c in C.GRID_SCENARIOS]
    assert len(set(ids)) == len(ids) and ids[0] == "grund"
    grund = C.GRID_SCENARIOS[0]
    assert (grund[2], grund[3], grund[4], grund[5]) == (C.DEFAULT_R, C.DEFAULT_SG, C.DEFAULT_THR, C.DEFAULT_HMAX)
    assert list(C.GRID_SM) == sorted(C.GRID_SM) and C.GRID_SM[0] == 0.0
