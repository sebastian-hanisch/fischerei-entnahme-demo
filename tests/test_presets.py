"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen, Schlüssel gedeckt, jedes Preset einzigartig; Permalink begrenzt und rastet ein."""

import pytest

import fis_constants as C
import fis_evaluation as ev
import fis_presets as P


def test_every_preset_has_help():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    assert all(C.PRESET_HELP[n] for n in C.PRESETS)


def test_preset_keys_are_known_settings():
    for name, preset in C.PRESETS.items():
        assert set(preset) <= set(P.PRESET_KEYS), name


def test_preset_values_are_inside_the_slider_bounds_and_on_the_slider_grid():
    for name, preset in C.PRESETS.items():
        for key, value in preset.items():
            spec = P.SETTING_SPECS[P.PRESET_KEYS[key]]
            assert spec.lo <= value <= spec.hi, (name, key, value)
            if spec.choices is not None:
                assert value in spec.choices, (name, key, value)
            elif spec.step is not None:
                k = (value - spec.lo) / spec.step
                assert abs(k - round(k)) < 1e-9, (name, key, value)       # der Wert liegt auf dem Raster des Reglers


def test_default_preset_changes_nothing():
    assert C.PRESETS["Grundfall (Voreinstellung)"] == {}


def test_setting_specs_cover_every_preset_key():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)


def test_defaults_inside_bounds():
    for key, spec in P.SETTING_SPECS.items():
        assert spec.lo <= spec.default <= spec.hi, key


def test_no_two_presets_set_the_same_values():
    seen = {}
    for name, preset in C.PRESETS.items():
        full = tuple(sorted({spec.url_param: preset.get(spec.url_param, spec.default) for spec in P.SETTING_SPECS.values()}.items()))
        assert full not in seen, (name, seen.get(full))
        seen[full] = name


def test_jedes_preset_waehlt_eine_regel_und_den_zugehoerigen_reglerwert():
    """Ein Preset mit Regel Quote nennt Q, mit Aufwand e, mit Schwelle S*: der Reglerwert gilt wirklich für die gewählte Regel."""
    needs = {C.RULE_QUOTE: "q", C.RULE_EFFORT: "e", C.RULE_ESCAPE: "s"}
    for name, preset in C.PRESETS.items():
        rule = preset.get("rule", C.DEFAULT_RULE)
        if rule in needs and preset:
            assert needs[rule] in preset, name


def test_lehrbuch_preset_nutzt_die_formel_der_regel():
    p = C.PRESETS["Lehrbuch: Quote am MSY"]
    assert p["q"] == pytest.approx(C.DEFAULT_R / 4) and p["sg"] == 0.01      # Q = r K / 4 bei r = 0,5
    a = C.PRESETS["Aufwand stabil"]
    assert a["e"] == pytest.approx(C.DEFAULT_R / (2 + C.DEFAULT_R))          # e = r / (2 + r) = 0,2
    f = C.PRESETS["Schneller Bestand (r = 1)"]
    assert f["r"] == 1.0 and f["rule"] == C.RULE_EFFORT


def test_regel_aus_den_reglern():
    assert ev.rule_from_settings(C.RULE_QUOTE, 0.125, 0.2, 0.4, 0.5) == ("quote", 0.125, None)
    assert ev.rule_from_settings(C.RULE_EFFORT, 0.125, 0.2, 0.4, 0.5) == ("aufwand", 0.2, None)
    assert ev.rule_from_settings(C.RULE_ESCAPE, 0.125, 0.2, 0.4, 0.5) == ("escape", 0.4, None)
    assert ev.rule_from_settings(C.RULE_DAMPED, 0.125, 0.2, 0.4, 0.5) == ("schwelle", 0.4, 0.5)
    assert ev.rule_from_settings(C.RULE_QUOTE, 0.12500000000000003, 0, 0, 0)[1] == 0.125            # Rundung gegen Gleitkommareste des Reglers


@pytest.mark.parametrize(
    "key, raw, expected",
    [
        ("q_slider", 9.0, 0.4), ("q_slider", -1.0, 0.0), ("q_slider", 0.1234, 0.125),            # begrenzen und auf 0,005 einrasten
        ("sm_slider", 0.3, 0.3), ("sm_slider", 0.31, 0.3), ("sm_slider", 0.3126, 0.325),
        ("beta_select", 0.93, 0.95), ("beta_select", 0.99, 0.98), ("beta_select", 5.0, 1.0), ("beta_select", 0.0, 0.9),    # nur die vier erlaubten Werte
        ("rule_select", 2, 2), ("rule_select", 9, 3), ("rule_select", -4, 0),
        ("ntrain_slider", 1234, 1200), ("ntrain_slider", 50, 200), ("ntest_slider", 99999, 10000),
        ("seed_train_input", 5, 5), ("seed_train_input", 10 ** 9, 999999),
    ],
)
def test_snap_begrenzt_und_rastet_ein(key, raw, expected):
    spec = P.SETTING_SPECS[key]
    assert P.snap(spec, raw) == pytest.approx(expected) and type(P.snap(spec, raw)) is spec.caster
