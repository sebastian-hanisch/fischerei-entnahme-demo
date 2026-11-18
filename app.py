"""Fischerei und Wild: Wie viel entnehmen? Quote, Aufwand oder Schwelle - interaktive Fall-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Wachstumsmodell vom Schaefer-Typ mit Wachstums- und Messrauschen: feste Quote, fester Aufwand und Schwellenregeln auf dem gemessenen Bestand, fair abgestimmt (getrennte
Trainings- und Testpfade), Wertiteration als Referenz bei vollem Wissen, vorgerechnetes Raster „Wo kippt es?“. Alle Zahlen sind Annahmen (README, „Ehrliche Grenzen“).
Lauffähig mit: streamlit run app.py
"""

from fractions import Fraction

import streamlit as st

import fis_algorithm as A
import fis_constants as C
import fis_dp as D
import fis_evaluation as EV
import fis_grid as G
import fis_hand as H
import fis_texts as TX
import fis_visualization as V
from fis_layout import dunkles_design, kopf_css
from fis_pdf_export import build_pdf
from fis_presets import (
    SETTING_SPECS,
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sidebar_presets,
    sync_query_params,
)
from fis_scenario import Scenario, noise

st.set_page_config(page_title="Fischerei und Wild – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner=False)
def _view(scen, rule, par, par2, seed):
    return EV.view_run(scen, rule, par, par2, seed)


@st.cache_data(show_spinner="Alle Regeln werden auf den Trainingspfaden abgestimmt und auf den Testpfaden bewertet (einige Sekunden) ...")
def _compare(scen, n_train, n_test, seed_train, seed_test):
    return EV.compare_rules(scen, n_train, n_test, seed_train, seed_test)


@st.cache_data(show_spinner="Die Wertiteration rechnet (einige Sekunden) ...")
def _dp(r, sg, beta, cost):
    return D.solve_cached(r, sg, beta, cost=cost)


@st.cache_data(show_spinner=False)
def _dp_check(r, sg, beta):
    """Gegenprobe: die Politik der Wertiteration, simuliert mit exaktem Survey (Zufalls-Seed 777, 4 000 Pfade), gegen den Wert der Wertiteration."""
    g, V_, Sp = D.solve_cached(r, sg, beta)
    scen = Scenario(r=r, sg=sg, sm=0.0, beta=beta, hmax=1.0)
    ev = A.evaluate("opt", 0.0, None, scen, noise(777, C.T_YEARS, 4000), policy=(g, Sp))
    return ev["obj"], ev["se"]


@st.cache_data(show_spinner=False)
def _grid():
    return G.load()


def _slider(label, key, help_text, fmt=None, disabled=False):
    spec = SETTING_SPECS[key]
    return st.slider(label, *bounds(key), key=key, step=spec.step, help=help_text, format=fmt, disabled=disabled)


# Fester Kopf: nur der Titel bleibt beim Scrollen oben (Stylesheet in fis_layout.py).
st.html(kopf_css(dunkles_design()))
with st.container(key="kopf"):
    st.title("🐟 Fischerei und Wild: Wie viel entnehmen?")
st.markdown(
    """
Wie viel darf man einem Bestand (Fisch, Wild) in jedem Jahr entnehmen: eine feste **Quote**, einen festen **Aufwand** oder alles, was der gemessene Bestand über einer **Schwelle** liegt (Escapement)? Die Demo vergleicht diese Regeln in einem Wachstumsmodell vom Schaefer-Typ mit **Wachstumsrauschen** und **Messrauschen** des Surveys, stimmt jede Regel fair auf getrennten Pfaden ab und misst neben dem Fang immer den **Kollaps**. Bei exaktem Survey erreicht die Schwelle in den Messungen das Ziel der Wertiteration (gemessen, nicht bewiesen); mit Messrauschen kippt die Reihenfolge: im Grundfall liegt die Escapement-Schwelle mit Kollapsgrenze ab einem Messrauschen zwischen 0,15 und 0,2 (linear interpoliert etwa 0,19) hinter dem festen Aufwand, der den Survey gar nicht liest. Wo genau, hängt an der Definition des Kollapses und an der Kapazitätsgrenze (Tab „Wo kippt es?“).

Alle Zahlen sind **Annahmen** (kein Bestand wurde gemessen), die Demo gibt keine Bewirtschaftungsempfehlung. Direkt darunter erklärt der Expander **💬 In einfachen Worten** das Ganze ohne Formeln. Weiter unten: die **Kernfragen**, dann die Expander **🔧 Wie wir das erreichen – vollständiger Methodenvergleich**, **📐 Mathematische Formulierung** und **⚠️ Annahmen, Grenzen und bekannte Probleme**.
"""
)

# erklaerseite:einfach:start (erzeugt von tools/set_demo_explainer.py (Website-Repo) aus tools/content/explainers/fischerei-entnahme-erklaert.json, nicht von Hand ändern)
with st.expander("💬 In einfachen Worten"):
    st.markdown(
        """
Ein erfundener Fisch- oder Wildbestand wächst mit Zufall und wird jedes Jahr zum Teil entnommen. Verglichen werden eine **feste Quote** (jedes Jahr dieselbe Menge), ein **fester Aufwand** (derselbe Anteil des Bestands), eine **Escapement-Schwelle** (alles, was der gezählte Bestand über einer Schwelle liegt) und eine **gedämpfte Schwelle** (nur ein Teil davon). Die Schwellenregeln lesen eine Zählung des Bestands, den **Survey**, und die Zählung hat einen Fehler; Quote und Aufwand lesen sie nicht. Neben dem Fang steht immer der **Kollaps**: der Anteil der Verläufe, in denen der Bestand unter die Kollapsgrenze fällt.

Gefragt wird, ob die gewählte Regel den Bestand hält, ob das Ziel allein den Kollaps versteckt und bei welchem Messrauschen die Schwelle hinter den festen Aufwand fällt. Alle Zahlen sind Annahmen; es wurde kein echter Bestand gemessen, und die Anwendung gibt keine Bewirtschaftungsempfehlung.

[Ausführlich und verständlich erklärt, mit Fachteil](https://sebastianhanisch.net/fischerei-entnahme-erklaert.html)
        """
    )
# erklaerseite:einfach:end
st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben
    st.header("⚙️ Einstellungen")
    st.markdown("**Bestand (Annahmen)**")
    r = _slider("Wachstumsrate r", "r_slider", "Wachstum des Rests je Jahr: f(S) = S + r S (1 − S). Bis r = 1 ist die Wachstumsfunktion auf [0, 1] monoton.", "%.1f")
    sg = _slider("Wachstumsrauschen σ_g", "sg_slider", "Standardabweichung des lognormalen Wachstumsfaktors Z auf der Logskala (Mittel 1). 0 = kein Rauschen.", "%.2f")
    st.markdown("**Survey und Ziel**")
    sm = _slider("Messrauschen σ_m", "sm_slider", "Standardabweichung des lognormalen Messfehlers des Surveys auf der Logskala (Mittel 1, unabhängig). 0 = exakter Survey. Quote und Aufwand lesen den Survey nicht.", "%.3f")
    beta = st.select_slider("Diskontfaktor β", options=C.BETA_OPTIONS, key="beta_select",
                            help="Gewicht des Folgejahres im Ziel. 1,0 = nicht diskontiert: Mittel des Fangs der Jahre 50 bis 199. Die Wertiteration (Tab „Das Optimum“) braucht β unter 1.")
    st.markdown("**Grenzen**")
    hmax = _slider("Kapazitätsgrenze h_max", "hmax_slider", "Die Entnahme ist höchstens h_max mal der Bestand (Annahme: Flottenkapazität). Bei 1,0 darf eine Regel den ganzen Bestand nehmen.", "%.2f")
    thr = _slider("Kollapsgrenze (in K)", "thr_slider", "Kollaps = der Bestand fällt irgendwann in den Jahren 1 bis 200 unter diese Grenze (Anteil der Tragfähigkeit K).", "%.2f")
    st.markdown("**Entnahmeregel (Tab 1)**")
    rule_idx = st.selectbox("Regel", list(C.RULE_NAMES), format_func=C.RULE_NAMES.get, key="rule_select",
                            help="Quote: feste Menge. Aufwand: fester Anteil des Bestands. Escapement-Schwelle: alles über der Schwelle auf dem Messwert. Gedämpfte Schwelle: ein Anteil f davon.")
    q = _slider("Quote Q (Anteil von K je Jahr)", "q_slider", "Gilt für die Regel Quote; bei den anderen Regeln ausgegraut.", "%.3f", disabled=rule_idx != C.RULE_QUOTE)
    e = _slider("Aufwandsanteil e", "e_slider", "Gilt für die Regel Aufwand; bei den anderen Regeln ausgegraut.", "%.2f", disabled=rule_idx != C.RULE_EFFORT)
    s = _slider("Schwelle S* (in K)", "s_slider", "Gilt für Escapement-Schwelle und gedämpfte Schwelle; bei Quote und Aufwand ausgegraut.", "%.2f",
                disabled=rule_idx not in (C.RULE_ESCAPE, C.RULE_DAMPED))
    f = _slider("Dämpfung f", "f_slider", "Gilt für die gedämpfte Schwelle: entnommen wird der Anteil f des Messwerts über der Schwelle; bei den anderen Regeln ausgegraut.", "%.1f",
                disabled=rule_idx != C.RULE_DAMPED)
    st.markdown("**Pfade**")
    seed_view = st.number_input("Zufalls-Seed der Pfade (Tab 1)", *bounds("seed_view_input"), key="seed_view_input", step=1)
    st.button("🎲 Neue Pfade ziehen", width="stretch", on_click=randomize_seed)
    st.markdown("**Abstimmung und Test (Tab 2)**")
    n_train = _slider("Trainingspfade", "ntrain_slider", "Auf diesen Pfaden wird jede Regel abgestimmt (Voreinstellung 1 000).", "%d")
    n_test = _slider("Testpfade", "ntest_slider", "Auf diesen frischen Pfaden wird jede Regel bewertet (Voreinstellung 5 000).", "%d")
    seed_train = st.number_input("Zufalls-Seed Training", *bounds("seed_train_input"), key="seed_train_input", step=1)
    seed_test = st.number_input("Zufalls-Seed Test", *bounds("seed_test_input"), key="seed_test_input", step=1)

sync_query_params({
    "rule_select": int(rule_idx), "q_slider": float(q), "e_slider": float(e), "s_slider": float(s), "f_slider": float(f), "r_slider": float(r), "sg_slider": float(sg),
    "sm_slider": float(sm), "beta_select": float(beta), "hmax_slider": float(hmax), "thr_slider": float(thr), "seed_view_input": int(seed_view), "ntrain_slider": int(n_train),
    "ntest_slider": int(n_test), "seed_train_input": int(seed_train), "seed_test_input": int(seed_test),
})

scen = Scenario(r=round(float(r), 4), sg=round(float(sg), 4), sm=round(float(sm), 4), beta=float(beta), hmax=round(float(hmax), 4), thr=round(float(thr), 4))
rule, par, par2 = EV.rule_from_settings(rule_idx, q, e, s, f)
rule_name = C.RULE_NAMES[rule_idx]

tab1, tab2, tab3, tab4, tab_hand, tab_limits = st.tabs([
    "🐟 Bestand und Entnahme", "⚖️ Regeln im fairen Vergleich", "🔀 Wo kippt es?", "🎯 Das Optimum (Wertiteration)", "✋ Von Hand", "⚠️ Annahmen und Grenzen",
])

# --- Tab 1: Bestand und Entnahme -----------------------------------------------------------------------------------------------------------
view = _view(scen, rule, par, par2, int(seed_view))
with tab1:
    st.markdown(f"## 📊 Ergebnis: {rule_name} ({EV.fmt_par(par, par2)}) auf {view['paths']} Pfaden über {C.T_YEARS} Jahre")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Ziel: gleichwertiger Jahresfang", f"{view['obj']:.4f} K", delta=f"± {view['se']:.4f} Standardfehler", delta_color="off")
    m2.metric("Kollaps", f"{100 * view['collapse']:.1f} %", delta=f"Bestand irgendwann unter {scen.thr:.2f} K", delta_color="off")
    m3.metric("Fang-Streuung", f"{view['cv']:.2f}", delta="Variationskoeffizient, Jahre 50 bis 199", delta_color="off")
    m4.metric("Mittlerer Bestand", f"{view['meanB']:.3f} K", delta="Jahre 50 bis 199", delta_color="off")
    st.caption(
        f"Pfade mit Aussterben (Bestand irgendwann unter {C.EXTINCT:g} K): {100 * view['extinct']:.1f} %. Ziel = gleichwertiger Jahresfang in K je Jahr "
        f"({'diskontiert mit β = ' + format(scen.beta, 'g') + ', Start B_0 = K' if scen.beta < 1 else 'Mittel der Jahre 50 bis 199'}). Der Messwert (Survey) wird nur von den Schwellenregeln gelesen."
    )
    if view["overshoot"] > 0:
        st.caption(f"⚠️ Auf {100 * view['overshoot']:.1f} % der Pfade schießt der Bestand über (1 + 1/r) K hinaus; dann wäre das Wachstum negativ, und das Modell setzt den Bestand auf 0 (Absturz).")
    ymax = V.nice_ymax(view["B"], view["Y"])
    med, lo, hi = EV.band(view["B"])
    st.markdown("**Pfadbündel: Median und mittlere 80 % der Pfade**")
    st.plotly_chart(V.fig_fan(med, lo, hi, scen.thr, ymax), width="stretch", key="fan_chart")
    st.markdown("**Ein einzelner Pfad: Bestand, Messwert und Entnahme**")
    idx = st.slider("Einzelpfad Nr.", 1, view["paths"], 1, key="single_path", help="Welcher der gezogenen Pfade unten gezeigt wird.")
    st.plotly_chart(V.fig_single(view["B"][:, idx - 1], view["H"][:, idx - 1], view["Y"][:, idx - 1], scen.thr, ymax), width="stretch", key="single_chart")
    st.caption("Die Entnahme vor dem Wachstum: im Jahr t wird erst gemessen (Punkt), dann entnommen (Säule), dann wächst der Rest. Der Bestand zeigt B_t zu Beginn des Jahres.")

# --- Tab 2: Regeln im fairen Vergleich -----------------------------------------------------------------------------------------------------
with tab2:
    st.markdown("## ⚖️ Alle Regeln, fair abgestimmt und auf frischen Pfaden getestet")
    cmp = _compare(scen, int(n_train), int(n_test), int(seed_train), int(seed_test))
    ref = cmp["reference"]
    MODE_TEXT = {"frei": "frei (größtes Ziel)", "kollaps": "mit Kollapsgrenze (höchstens 10 % der Trainingspfade)", "lehrbuch": "Lehrbuch-Wert", "dp": "aus der Wertiteration"}
    table = []
    if ref is not None:
        table.append({"Regel": "Referenz: Wertiteration bei vollem Wissen", "Abstimmung": "–", "Regler": "–", "Ziel (K je Jahr)": ref, "± Standardfehler": None, "Anteil der Referenz (%)": 100.0,
                      "Kollaps (%)": None, "Aussterben (%)": None, "Überschießen (%)": None, "Mittlerer Bestand (K)": None, "Fang-Streuung": None})
    for row in cmp["rows"]:
        mode = MODE_TEXT[row["mode"]] + ("" if row["feasible"] else " – nicht zulässig")
        table.append({
            "Regel": row["name"], "Abstimmung": mode, "Regler": "–" if row["rule"] == "opt" else EV.fmt_par(row["par"], row["par2"]), "Ziel (K je Jahr)": row["obj"],
            "± Standardfehler": row["se"], "Anteil der Referenz (%)": None if ref is None else 100.0 * row["obj"] / ref, "Kollaps (%)": 100.0 * row["collapse"],
            "Aussterben (%)": 100.0 * row["extinct"], "Überschießen (%)": 100.0 * row["overshoot"], "Mittlerer Bestand (K)": row["meanB"], "Fang-Streuung": row["cv"],
        })
    st.dataframe(
        table, hide_index=True, width="stretch",
        column_config={
            "Ziel (K je Jahr)": st.column_config.NumberColumn(format="%.4f"), "± Standardfehler": st.column_config.NumberColumn(format="%.4f"),
            "Anteil der Referenz (%)": st.column_config.NumberColumn(format="%.1f %%"), "Kollaps (%)": st.column_config.NumberColumn(format="%.1f %%"),
            "Aussterben (%)": st.column_config.NumberColumn(format="%.1f %%"), "Überschießen (%)": st.column_config.NumberColumn(format="%.1f %%"),
            "Mittlerer Bestand (K)": st.column_config.NumberColumn(format="%.3f"), "Fang-Streuung": st.column_config.NumberColumn(format="%.2f"),
        },
    )
    st.caption(
        f"Abgestimmt auf {cmp['n_train']} Trainingspfaden (Zufalls-Seed {int(seed_train)}), bewertet auf {cmp['n_test']} Testpfaden (Zufalls-Seed {int(seed_test)}), alle Regeln mit denselben Zufallszahlen. "
        "Kollaps: Bestand irgendwann in den Jahren 1 bis 200 unter der Kollapsgrenze. Fang-Streuung: Variationskoeffizient des Fangs über die Jahre 50 bis 199, über die Pfade gemittelt. "
        "Die Spalten Quote und Aufwand ändern sich mit dem Messrauschen nicht, weil diese Regeln den Survey nicht lesen. "
        "Der Anteil der Referenz vergleicht mit dem Optimum bei vollem Wissen (exakter Bestand, ohne Kapazitätsgrenze); ein Optimum für den verrauschten Survey ist nicht gerechnet."
    )
    over = [f"{row['name']} ({row['mode']}) {100 * row['overshoot']:.1f} %" for row in cmp["rows"] if row["overshoot"] > 0]
    if over:
        st.caption("⚠️ Überschießen (Bestand über (1 + 1/r) K, dann Absturz auf 0) auf diesen Anteilen der Testpfade: " + ", ".join(over) + ".")
    if any(not row["feasible"] for row in cmp["rows"]):
        st.caption("„nicht zulässig“: kein Reglerwert hielt auf den Trainingspfaden die Kollapsgrenze ein; gewählt ist das kleinste Kollapsrisiko, der Kollapsanteil der Testpfade steht in der Zeile.")
    st.plotly_chart(V.fig_tradeoff(cmp["rows"], ref, cmp["cap"]), width="stretch", key="tradeoff_chart")
    st.caption("Je weiter oben, desto größer das Ziel, je weiter links, desto weniger Kollaps. Das Ziel allein verrät den Kollaps nicht: Punkte rechts der gestrichelten Linie überschreiten die Kollapsgrenze der Abstimmung.")
    st.markdown("**Gepaarte Differenzen zum festen Aufwand** (Ziel der Regel minus Ziel des Aufwands, gleiche Pfade, gleiche Art der Abstimmung)")
    st.dataframe(
        [{"Vergleich": name, "Abstimmung": MODE_TEXT[mode], "Differenz": cmp["diffs"][(key, mode)][0], "± Standardfehler": cmp["diffs"][(key, mode)][1]}
         for key, name in (("escape", "Escapement-Schwelle minus Aufwand"), ("schwelle", "Gedämpfte Schwelle minus Aufwand")) for mode in ("frei", "kollaps")],
        hide_index=True, width="stretch",
        column_config={"Differenz": st.column_config.NumberColumn(format="%+.4f"), "± Standardfehler": st.column_config.NumberColumn(format="%.4f")},
    )

# --- Tab 3: Wo kippt es? --------------------------------------------------------------------------------------------------------------------
with tab3:
    st.markdown("## 🔀 Wo kippt die Reihenfolge? Escapement gegen festen Aufwand über das Messrauschen")
    grid = _grid()
    if not grid:
        st.info("Das vorgerechnete Raster fehlt (data/kipp.csv; `python fis_grid.py`).")
    else:
        st.markdown(
            "Vorgerechnet (β = 0,95, 1 000 Trainings- und 5 000 Testpfade, Zufalls-Seeds 11 und 22): für jede Einstellung und jedes Messrauschen werden Aufwand, Escapement-Schwelle und gedämpfte Schwelle "
            "**neu abgestimmt**. Dieser Tab hat eigene Einstellungen; die Regler der Seitenleiste gelten hier nicht."
        )
        labels = {c[0]: c[1] for c in C.GRID_SCENARIOS}
        c1, c2 = st.columns([3, 2])
        sid = c1.selectbox("Einstellung", list(labels), format_func=labels.get, key="grid_scen")
        mode = c2.radio("Abstimmung", ["kollaps", "frei"], format_func=lambda m: "mit Kollapsgrenze (höchstens 10 %)" if m == "kollaps" else "frei (Ziel allein)", key="grid_mode")
        auf, esc, sch, opt = (G.series(grid, sid, rl, md) for rl, md in (("aufwand", mode), ("escape", mode), ("schwelle", mode), ("opt", "dp")))
        gref = G.series(grid, sid, "ref", "dp")[0]["obj"]
        xs = [rw["sm"] for rw in esc]
        series_obj = {"Aufwand (liest den Survey nicht)": auf, "Escapement-Schwelle": esc, "Gedämpfte Schwelle": sch, "DP-Politik auf dem Messwert": opt}
        k_esc = EV.kipp(xs, [rw["d_vs_aufwand"] for rw in esc])
        k_sch = EV.kipp(xs, [rw["d_vs_aufwand"] for rw in sch])
        st.markdown("**Ziel über dem Messrauschen**")
        st.plotly_chart(V.fig_kipp_ziel(series_obj, gref), width="stretch", key="kipp_ziel_chart")
        st.markdown("**Differenz zum festen Aufwand: unter der Nulllinie liegt der Aufwand vorn**")
        st.plotly_chart(V.fig_kipp_diff({"Escapement minus Aufwand": esc, "Gedämpfte Schwelle minus Aufwand": sch}, k_esc, f"Kippstelle Escapement {k_esc:.2f}" if k_esc is not None else ""),
                        width="stretch", key="kipp_diff_chart")
        kind = "mit Kollapsgrenze" if mode == "kollaps" else "frei abgestimmt"
        for name, k_ in (("Escapement-Schwelle gegen Aufwand", k_esc), ("Gedämpfte Schwelle gegen Aufwand", k_sch)):
            kd, kt = EV.verdict_kipp(k_, xs[-1], f"{name}, {kind}")
            getattr(st, kd)(kt)
        st.markdown("**Kollaps über dem Messrauschen (Testpfade)**")
        st.plotly_chart(V.fig_kipp_collapse({"Aufwand": auf, "Escapement-Schwelle": esc, "Gedämpfte Schwelle": sch}, C.COLLAPSE_CAP), width="stretch", key="kipp_collapse_chart")
        missed = sorted({rw["sm"] for rows_ in (auf, esc, sch) for rw in rows_ if not rw["feasible"]})
        if missed:
            st.caption("„nicht zulässig“ (Kollapsgrenze auf den Trainingspfaden nicht erreichbar; gewählt ist das kleinste Kollapsrisiko) bei Messrauschen: "
                       + ", ".join(f"{m:g}" for m in missed) + f" (Regel: {', '.join(n for n, rows_ in (('Aufwand', auf), ('Escapement', esc), ('Gedämpfte Schwelle', sch)) if any(not rw['feasible'] for rw in rows_))}).")
        st.markdown("**Zahlen der gewählten Einstellung**")
        st.dataframe(
            [{"Messrauschen σ_m": a_["sm"], "Aufwand e": a_["par"], "Aufwand Ziel": a_["obj"], "Escapement S*": e_["par"], "Escapement Ziel": e_["obj"], "Escapement Kollaps (%)": 100 * e_["collapse"],
              "Schwelle S*": s_["par"], "Schwelle f": s_["par2"], "Schwelle Ziel": s_["obj"], "Schwelle Kollaps (%)": 100 * s_["collapse"], "Escapement − Aufwand": e_["d_vs_aufwand"],
              "Schwelle − Aufwand": s_["d_vs_aufwand"]} for a_, e_, s_ in zip(auf, esc, sch)],
            hide_index=True, width="stretch",
            column_config={
                "Messrauschen σ_m": st.column_config.NumberColumn(format="%.3f"), "Aufwand e": st.column_config.NumberColumn(format="%.2f"), "Aufwand Ziel": st.column_config.NumberColumn(format="%.4f"),
                "Escapement S*": st.column_config.NumberColumn(format="%.2f"), "Escapement Ziel": st.column_config.NumberColumn(format="%.4f"),
                "Escapement Kollaps (%)": st.column_config.NumberColumn(format="%.1f %%"), "Schwelle S*": st.column_config.NumberColumn(format="%.2f"),
                "Schwelle f": st.column_config.NumberColumn(format="%.1f"), "Schwelle Ziel": st.column_config.NumberColumn(format="%.4f"),
                "Schwelle Kollaps (%)": st.column_config.NumberColumn(format="%.1f %%"), "Escapement − Aufwand": st.column_config.NumberColumn(format="%+.4f"),
                "Schwelle − Aufwand": st.column_config.NumberColumn(format="%+.4f"),
            },
        )
        st.markdown("**Kippstelle je Einstellung** (Messrauschen, ab dem der feste Aufwand vor der Schwelle liegt; lineare Interpolation zwischen den Rasterpunkten)")
        summary = []
        for sid_, label_, *_ in C.GRID_SCENARIOS:
            row_ = {"Einstellung": label_}
            for col, rl, md in (("Escapement, frei", "escape", "frei"), ("Escapement, mit Kollapsgrenze", "escape", "kollaps"), ("Gedämpfte Schwelle, mit Kollapsgrenze", "schwelle", "kollaps")):
                sr = G.series(grid, sid_, rl, md)
                k_ = EV.kipp([x["sm"] for x in sr], [x["d_vs_aufwand"] for x in sr])
                note = "" if all(x["feasible"] for x in G.series(grid, sid_, "aufwand", md)) else " (Aufwand nicht zulässig)"
                row_[col] = (f"{k_:.2f}" if k_ is not None else "kein Kipp") + note
            summary.append(row_)
        st.dataframe(summary, hide_index=True, width="stretch")
        st.caption("Die Kippstelle hängt an der Definition des Kollapses und an der Kapazitätsgrenze; sie ist eine Aussage über unabhängiges, unverzerrtes Messrauschen im Modell (β = 0,95), keine Zahl für einen echten Bestand.")

# --- Tab 4: Das Optimum --------------------------------------------------------------------------------------------------------------------
with tab4:
    st.markdown("## 🎯 Das Optimum bei vollem Wissen: Wertiteration")
    if scen.beta >= 1.0:
        st.info("Ohne Diskontierung (β = 1) ist der Wert V unendlich; die Wertiteration braucht β unter 1. Wählen Sie in der Seitenleiste β = 0,98 oder kleiner.")
    else:
        cost = st.select_slider("Aufwandkosten c je Jahr (Preis 1: Kosten c · ln(B/S))", options=C.DP_COST_OPTIONS, value=C.DEFAULT_DP_COST, key="dp_cost",
                                help="0 = ohne Kosten. Mit Kosten steigt die Schwelle (Wertiteration, Gitter von 801 Punkten); nur die Schwellenhöhe ist gerechnet, nicht die Reihenfolge der Regeln unter Kosten.")
        g, Vv, Sp = _dp(scen.r, scen.sg, scen.beta, float(cost))
        s_dp = D.threshold(Sp)
        domain_ok = 0.0 < 1.0 / scen.beta - 1.0 < scen.r
        s_formula = D.s_star_formula(scen.r, scen.beta) if domain_ok else None
        d1, d2, d3 = st.columns(3)
        d1.metric("Schwelle S* der Wertiteration", f"{s_dp:.4f} K")
        d2.metric("Formel (ohne Wachstumsrauschen)", "–" if s_formula is None else f"{s_formula:.4f} K", delta="S* = ½ (1 − (1/β − 1)/r)", delta_color="off")
        d3.metric("Gleichwertiger Jahresfang (1 − β) V(K)", f"{D.annual_value(g, Vv, scen.beta):.4f} K")
        if cost == 0.0:
            kd, kt = EV.verdict_dp(scen.sg, s_dp, s_formula, float(g[1] - g[0]))
            getattr(st, kd)(kt)
            sim_v, sim_se = _dp_check(scen.r, scen.sg, scen.beta)
            st.caption(f"Gegenprobe: die Politik der Wertiteration, simuliert mit exaktem Survey, ohne Kapazitätsgrenze und mit {C.T_YEARS} Jahren über 4 000 Pfade (Zufalls-Seed 777), bringt {sim_v:.4f} ± {sim_se:.4f} K je Jahr.")
        else:
            st.info("Mit Kosten gilt die geschlossene Formel nicht (sie gilt ohne Kosten und ohne Wachstumsrauschen); der Wert und die Schwelle stammen allein aus der Wertiteration.")
        st.markdown("**Wert in Abhängigkeit vom Anfangsbestand**")
        st.plotly_chart(V.fig_dp_value(g, Vv, scen.beta), width="stretch", key="dp_value_chart")
        st.markdown("**Politik: Rest nach der Entnahme**")
        st.plotly_chart(V.fig_dp_policy(g, Sp, s_formula if (scen.sg == 0 and cost == 0.0) else None), width="stretch", key="dp_policy_chart")
        st.caption(
            f"Gitter: {len(g)} Punkte für den Bestand auf [0, {C.DP_BMAX:g}], {C.DP_NODES} Gauß-Hermite-Knoten für das Wachstumsrauschen; die Schwelle ist auf einen Gitterschritt ({g[1] - g[0]:.4f}) genau. "
            "Unter der Schwelle wird nichts entnommen, darüber bleibt der Rest S*: eine Escapement-Schwelle. Die Politik kennt den Bestand genau; für den verrauschten Survey ist kein Optimum gerechnet."
        )

# --- Von Hand ------------------------------------------------------------------------------------------------------------------------------
with tab_hand:
    st.markdown(
        "Das Modell **ohne Zufall** (σ_g = σ_m = 0) in **exakten Brüchen**: r = 1/2, K = 1, Kapazitätsgrenze 9/10, der Survey ist exakt (Y = B). "
        "Regeln: Quote Q = 1/8 (= r K / 4), Aufwand e = 1/5 (= r / (2 + r)), Escapement-Schwelle S* = 9/20. Ein Jahr: Entnahme H = min(Regel, 9/10 · B), Rest S = B − H, nächster Bestand B_{t+1} = S + S (1 − S) / 2."
    )
    starts = {"1 (= K)": Fraction(1), "5/8 (Fixpunkt von Quote und Aufwand)": Fraction(5, 8), "31/50 (= 0,62)": Fraction(31, 50), "11/20 (= 0,55)": Fraction(11, 20)}
    hc1, hc2 = st.columns(2)
    start_name = hc1.selectbox("Anfangsbestand B_0", list(starts), key="hand_start")
    rule_name_h = hc2.selectbox("Regel", list(H.RULES), key="hand_rule")
    hrule, hpar = H.RULES[rule_name_h]
    b0 = starts[start_name]
    st.markdown("**Die ersten drei Jahre in exakten Brüchen**")
    steps = []
    b_ = b0
    for t in range(3):
        h_, s_, b_next = H.next_stock(b_, hrule, hpar)
        steps.append({"Jahr t": t, "Bestand B_t": str(b_), "Entnahme H_t": str(h_), "Rest S_t": str(s_), "Bestand B_t+1": str(b_next), "B_t+1 als Dezimalzahl": round(float(b_next), 4)})
        b_ = b_next
    st.dataframe(steps, hide_index=True, width="stretch")
    st.markdown("**Langfristig (gleitende Rechnung mit denselben Formeln)**")
    long_rows = []
    for nm, (rl, pr) in H.RULES.items():
        path = H.orbit(rl, float(pr), float(b0), 101)
        long_rows.append({"Regel": nm, **{f"B_{y}": round(path[y][0], 4) for y in (0, 5, 10, 20, 50, 100)}, "Fang im Jahr 100": round(path[100][1], 4)})
    st.dataframe(long_rows, hide_index=True, width="stretch")
    eq_b, eq_h = H.effort_equilibrium()
    st.markdown(
        f"Aufwand e = r / (2 + r) = 1/5 hat das Gleichgewicht B* = K/2 + r K/4 = **{eq_b}** mit Fang r K/4 = **{eq_h}** und kehrt von jedem Start dorthin zurück. "
        f"Die Quote Q = 1/8 hat denselben Fixpunkt {eq_b}, aber nach unten keine Rückkehr: ein Bestand darunter fällt weiter (wähle B_0 = 11/20)."
    )
    s_h, v_h = H.optimum_closed()
    gh, Vh, Sph = D.solve_cached(0.5, 0.0, 0.95)
    st.markdown(
        f"**Optimum bei Diskontfaktor 19/20, ohne Rauschen:** S* = (1 − (1/β − 1)/r)/2 = **{s_h}** = {float(s_h):.4f}, V(1) = (1 − S*) + β/(1 − β) · r S* (1 − S*) = **{v_h}** = {float(v_h):.5f}, "
        f"gleichwertiger Jahresfang (1 − β) V(1) = **{H.annual_value()}** = {float(H.annual_value()):.5f}. Die Wertiteration (Gitter von 801 Punkten) findet S* = {D.threshold(Sph):.4f} und V(1) = {D.annual_value(gh, Vh, 0.95) / (1 - 0.95):.5f}."
    )

# --- Annahmen und Grenzen (Tab) -----------------------------------------------------------------------------------------------------------
with tab_limits:
    st.markdown(TX.ASSUMPTIONS_MARKDOWN)

# --- Kernfragen ----------------------------------------------------------------------------------------------------------------------------
st.markdown("## ❓ Kernfragen")
verdicts = []
st.markdown("**Hält die gewählte Regel den Bestand?**")
kind1, text1 = EV.verdict_collapse(view["collapse"], scen.thr, view["paths"], rule_name)
getattr(st, kind1)(text1)
st.markdown("**Versteckt das Ziel allein den Kollaps?**")
kind2, text2 = EV.verdict_hidden(cmp)
getattr(st, kind2)(text2)
st.markdown("**Escapement oder fester Aufwand: wer liegt bei diesem Messrauschen vorn?**")
kind3, text3 = EV.verdict_order(cmp, scen.sm)
getattr(st, kind3)(text3)

pdf_bytes = build_pdf(
    {"r": scen.r, "sigma_g": scen.sg, "sigma_m": scen.sm, "beta": scen.beta, "h_max": scen.hmax, "Kollapsgrenze": scen.thr, "Regel": rule_name, "Regler": EV.fmt_par(par, par2),
     "Trainingspfade": cmp["n_train"], "Testpfade": cmp["n_test"]},
    {"Ziel": f"{view['obj']:.4f}", "Kollaps": f"{100 * view['collapse']:.1f} %", "Fang-Streuung": f"{view['cv']:.2f}", "mittlerer Bestand": f"{view['meanB']:.3f} K"},
    [{"name": rw["name"], "mode": rw["mode"], "par_text": "-" if rw["rule"] == "opt" else EV.fmt_par(rw["par"], rw["par2"]), "obj": rw["obj"], "collapse": rw["collapse"]} for rw in cmp["rows"]],
    [t.replace("**", "") for t in (text1, text2, text3)],
)
st.download_button("📄 Ergebnis als PDF", data=pdf_bytes, file_name="fischerei-entnahme.pdf", mime="application/pdf", key="pdf_download")

st.markdown("---")

# --- Expander ------------------------------------------------------------------------------------------------------------------------------
with st.expander("🔧 Wie wir das erreichen – vollständiger Methodenvergleich"):
    st.markdown(TX.METHODS_MARKDOWN)

with st.expander("📐 Mathematische Formulierung"):
    # erklaerseite:lesehilfe:start (erzeugt von tools/set_demo_explainer.py (Website-Repo) aus tools/content/explainers/fischerei-entnahme-erklaert.json, nicht von Hand ändern)
    st.markdown(
        r"""
**Lesehilfe.** Gegeben sind Wachstumsrate $r$, Wachstumsrauschen $\sigma_g$, Messrauschen $\sigma_m$, Diskontfaktor $\beta$, Kapazitätsgrenze $h_{\max}$ und Kollapsgrenze $\theta$. Gesucht ist je Regel der Reglerwert, der das Ziel $J$ unter der gewählten Nebenbedingung am größten macht, und das Optimum bei vollem Wissen als Referenz.

| Symbol | Bedeutung |
|---|---|
| $B_t$, $K=1$ | Bestand zu Beginn des Jahres $t$ in Einheiten der Tragfähigkeit |
| $Y_t$, $\xi_t$ | Messwert des Surveys; unabhängige Standardnormalzahl des Messfehlers |
| $Z_t$, $\eta_t$ | Wachstumsfaktor mit Mittel 1; unabhängige Standardnormalzahl des Wachstumsrauschens |
| $H_t$, $S_t$ | Entnahme und Rest (Escapement) nach der Entnahme |
| $Q$, $e$, $S^{*}$, $f$ | Reglerwerte: Quote, Aufwandsanteil, Schwelle, Dämpfung |
| $J$, $\beta$, $T$ | Ziel (gleichwertiger Jahresfang), Diskontfaktor, Zahl der Jahre ($T=200$) |
| $\theta$, $C$ | Kollapsgrenze in $K$; Kollaps-Anzeige (1 oder 0) |
| $V(B)$, $f(S)$ | Wert bei Bestand $B$ und vollem Wissen; Wachstumsfunktion $f(S)=S+rS(1-S)$ |
| $\Delta$, $\sigma_m^{*}$ | gepaarte Differenz der Ziele; Kippstelle des Messrauschens |

[Schritt für Schritt erklärt, mit Herleitung und Literatur](https://sebastianhanisch.net/fischerei-entnahme-erklaert.html#fachleute)
        """
    )
    # erklaerseite:lesehilfe:end
    st.markdown(r"""$$
Y_t=B_t\,\exp\!\Bigl(\sigma_m\,\xi_t-\tfrac12\,\sigma_m^{2}\Bigr),\qquad \xi_t\sim\mathcal N(0,1)
$$""")
    st.markdown(r"""$$
H_t=\min\bigl\{\,h_{\max}\,B_t,\ \tilde H_t\,\bigr\}
$$""")
    st.markdown(r"""$$
\tilde H_t=\begin{cases} Q & \text{Quote}\\ e\,B_t & \text{Aufwand}\\ \max(0,\,Y_t-S^{*}) & \text{Escapement-Schwelle}\\ f\,\max(0,\,Y_t-S^{*}) & \text{gedämpfte Schwelle}\end{cases}
$$""")
    st.markdown(r"""$$
S_t=B_t-H_t
$$""")
    st.markdown(r"""$$
B_{t+1}=Z_t\,\max\bigl\{0,\ S_t+r\,S_t\,(1-S_t)\bigr\}
$$""")
    st.markdown(r"""$$
Z_t=\exp\!\Bigl(\sigma_g\,\eta_t-\tfrac12\,\sigma_g^{2}\Bigr),\qquad \eta_t\sim\mathcal N(0,1)
$$""")
    st.markdown(r"""$$
J=\frac{1-\beta}{1-\beta^{T}}\sum_{t=0}^{T-1}\beta^{t}\,H_t\qquad(\beta<1)
$$""")
    st.markdown(r"""$$
J=\frac{1}{T-50}\sum_{t=50}^{T-1}H_t\qquad(\beta=1)
$$""")
    st.markdown(r"""$$
C=\mathbf 1\Bigl\{\min_{1\le t\le T}B_t<\theta\Bigr\}
$$""")
    st.markdown(r"""$$
\hat\rho_{\text{frei}}=\arg\max_{\rho}\ \overline{J}_{\text{Training}}(\rho),\qquad
\hat\rho_{\text{Kollaps}}=\arg\max_{\rho:\ \overline{C}_{\text{Training}}(\rho)\le 0{,}1}\ \overline{J}_{\text{Training}}(\rho)
$$""")
    st.markdown(r"""$$
V(B)=\max_{0\le S\le B}\Bigl[(B-S)+\beta\,\mathbb E\,V\bigl(Z\,f(S)\bigr)\Bigr]
$$""")
    st.markdown(r"""$$
S^{*}=\frac12\Bigl(1-\frac{1/\beta-1}{r}\Bigr)\qquad(\sigma_g=0)
$$""")
    st.markdown(r"""$$
V(1)=(1-S^{*})+\frac{\beta}{1-\beta}\,r\,S^{*}(1-S^{*})\qquad(\sigma_g=0)
$$""")
    st.markdown(r"""$$
Q_{\mathrm{MSY}}=\frac{rK}{4},\qquad e_{\mathrm{MSY}}=\frac{r}{2+r},\qquad S^{*}_{\mathrm{MSY}}=\frac K2
$$""")
    st.markdown(r"""$$
\sigma_m^{*}=\sigma_0+(\sigma_1-\sigma_0)\,\frac{\Delta_0}{\Delta_0-\Delta_1}\qquad\text{für }\Delta_0>0\ge\Delta_1,\quad \Delta=\overline J_{\text{Schwelle}}-\overline J_{\text{Aufwand}}
$$""")
    st.markdown(TX.MATH_NOTE)

with st.expander("⚠️ Annahmen, Grenzen und bekannte Probleme"):
    st.markdown(TX.ASSUMPTIONS_MARKDOWN)

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html)."
)
