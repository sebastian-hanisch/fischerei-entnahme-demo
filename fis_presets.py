"""Permalink (Begrenzen, Einrasten), Presets und Zufalls-Seed-Button (Standardmuster aus dem Demo-Portfolio)."""

import random
from dataclasses import dataclass

import streamlit as st

import fis_constants as C


@dataclass(frozen=True)
class SettingSpec:
    url_param: str
    caster: type
    default: object
    lo: float | None = None
    hi: float | None = None
    step: float | None = None          # Schrittweite des Reglers: ein Permalink-Wert rastet auf das Gitter ein
    choices: tuple | None = None       # erlaubte Werte (Auswahlregler): ein Permalink-Wert rastet auf den nächsten ein


SETTING_SPECS = {
    "rule_select": SettingSpec("rule", int, C.DEFAULT_RULE, C.RULE_QUOTE, C.RULE_DAMPED, choices=tuple(C.RULE_NAMES)),
    "q_slider": SettingSpec("q", float, C.DEFAULT_Q, C.Q_MIN, C.Q_MAX, step=0.005),
    "e_slider": SettingSpec("e", float, C.DEFAULT_E, C.E_MIN, C.E_MAX, step=0.01),
    "s_slider": SettingSpec("s", float, C.DEFAULT_S, C.S_MIN, C.S_MAX, step=0.01),
    "f_slider": SettingSpec("f", float, C.DEFAULT_F, C.F_MIN, C.F_MAX, step=0.1),
    "r_slider": SettingSpec("r", float, C.DEFAULT_R, C.R_MIN, C.R_MAX, step=0.1),
    "sg_slider": SettingSpec("sg", float, C.DEFAULT_SG, C.SG_MIN, C.SG_MAX, step=0.01),
    "sm_slider": SettingSpec("sm", float, C.DEFAULT_SM, C.SM_MIN, C.SM_MAX, step=0.025),
    "beta_select": SettingSpec("beta", float, C.DEFAULT_BETA, min(C.BETA_OPTIONS), max(C.BETA_OPTIONS), choices=C.BETA_OPTIONS),
    "hmax_slider": SettingSpec("hmax", float, C.DEFAULT_HMAX, C.HMAX_MIN, C.HMAX_MAX, step=0.05),
    "thr_slider": SettingSpec("thr", float, C.DEFAULT_THR, C.THR_MIN, C.THR_MAX, step=0.05),
    "seed_view_input": SettingSpec("seed_view", int, C.DEFAULT_SEED_VIEW, 0, C.SEED_MAX),
    "ntrain_slider": SettingSpec("ntrain", int, C.DEFAULT_TRAIN, C.TRAIN_MIN, C.TRAIN_MAX, step=100),
    "ntest_slider": SettingSpec("ntest", int, C.DEFAULT_TEST, C.TEST_MIN, C.TEST_MAX, step=500),
    "seed_train_input": SettingSpec("seed_train", int, C.DEFAULT_SEED_TRAIN, 0, C.SEED_MAX),
    "seed_test_input": SettingSpec("seed_test", int, C.DEFAULT_SEED_TEST, 0, C.SEED_MAX),
}
PRESET_KEYS = {spec.url_param: state_key for state_key, spec in SETTING_SPECS.items()}


def init_session_state_defaults():
    for state_key, spec in SETTING_SPECS.items():
        if state_key not in st.session_state:
            st.session_state[state_key] = spec.default


def bounds(state_key):
    spec = SETTING_SPECS[state_key]
    return spec.lo, spec.hi


def snap(spec, value):
    """Wert eines Permalinks auf die Reglergrenzen begrenzen und auf Schritt bzw. erlaubte Werte einrasten."""
    if spec.lo is not None:
        value = max(spec.lo, value)
    if spec.hi is not None:
        value = min(spec.hi, value)
    if spec.choices is not None:
        return spec.caster(min(spec.choices, key=lambda c: abs(c - value)))
    if spec.step is not None and spec.caster is float:
        value = spec.lo + round((value - spec.lo) / spec.step) * spec.step
        value = round(min(spec.hi, max(spec.lo, value)), 6)
    if spec.step is not None and spec.caster is int:
        value = spec.lo + round((value - spec.lo) / spec.step) * spec.step
        value = int(min(spec.hi, max(spec.lo, value)))
    return spec.caster(value)


def load_permalink_settings():
    """Einmal je Sitzung: Werte aus der Adresszeile übernehmen, auf die Reglergrenzen begrenzt und eingerastet."""
    if "permalink_loaded" in st.session_state:
        return
    qp = st.query_params
    for state_key, spec in SETTING_SPECS.items():
        if spec.url_param in qp:
            try:
                value = spec.caster(qp[spec.url_param])
            except (ValueError, TypeError):
                continue
            st.session_state[state_key] = snap(spec, value)
    st.session_state["permalink_loaded"] = True


def sync_query_params(values):
    """`values`: {state_key: aktueller Wert}; die Adresszeile spiegelt die Konfiguration."""
    try:
        for state_key, value in values.items():
            st.query_params[SETTING_SPECS[state_key].url_param] = str(value)
    except Exception:
        pass


def apply_preset(name):
    """Ein Preset setzt ALLE Regler: was es nicht nennt, geht auf den Standardwert (so wie aktives_preset vergleicht)."""
    for state_key, spec in SETTING_SPECS.items():
        st.session_state[state_key] = C.PRESETS[name].get(spec.url_param, spec.default)


def aktives_preset():
    """Name des Presets, dessen Werte GENAU den aktuellen Einstellungen entsprechen, sonst None.

    Zustandslos: verglichen wird jeder Regler aus SETTING_SPECS (ein Preset ohne Eintrag für einen Regler meint dessen
    Standardwert) mit dem Session State. Ein später ergänzter Regler braucht nur seinen Eintrag in SETTING_SPECS,
    dann hebt auch seine Änderung die Markierung auf (tests/test_layout.py prüft das für jeden Regler)."""
    preset_key_of = {state_key: key for key, state_key in PRESET_KEYS.items()}
    for name, preset in C.PRESETS.items():
        if all(
            st.session_state.get(state_key, spec.default) == preset.get(preset_key_of.get(state_key), spec.default)
            for state_key, spec in SETTING_SPECS.items()
        ):
            return name
    return None


def sidebar_presets():
    """Schnellstart oben in der Seitenleiste; das aktive Preset ist hervorgehoben (`type="primary"`)."""
    if not C.PRESETS:
        return
    st.header("🎯 Schnellstart")
    st.caption("Ein Beispielszenario laden:")
    aktiv = aktives_preset()
    for name in C.PRESETS:
        st.button(
            name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP.get(name, ""), key=f"preset_{name}",
            type="primary" if aktiv == name else "secondary",
        )


def randomize_seed():
    st.session_state["seed_view_input"] = random.randint(0, C.SEED_MAX)
