"""Kennzahlen, Vergleich der Regeln, Kippstelle und Verdikt-Logik (reine Funktionen; jede Schwelle steht in tests/test_verdicts.py)."""

import numpy as np

import fis_algorithm as A
import fis_constants as C
import fis_dp as D
from fis_scenario import Scenario, noise

RULE_KEYS = {C.RULE_QUOTE: "quote", C.RULE_EFFORT: "aufwand", C.RULE_ESCAPE: "escape", C.RULE_DAMPED: "schwelle"}
LABELS = {"quote": "Quote", "aufwand": "Aufwand", "escape": "Escapement-Schwelle", "schwelle": "Gedämpfte Schwelle", "opt": "DP-Politik auf dem Messwert"}


def rule_from_settings(rule_index, q, e, s, f):
    """Regel-Schlüssel und Reglerwerte (par, par2) aus den Reglern der Seitenleiste."""
    key = RULE_KEYS[rule_index]
    q, e, s, f = (round(float(x), 6) for x in (q, e, s, f))
    if key == "quote":
        return key, q, None
    if key == "aufwand":
        return key, e, None
    if key == "escape":
        return key, s, None
    return key, s, f


def fmt_par(par, par2=None):
    """Reglerwert als Text; die gedämpfte Schwelle nennt beide Werte."""
    return f"{par:.3f}" if par2 is None else f"{par:.3f} / f = {par2:.1f}"


# --- Tab 1: Pfadbündel ---------------------------------------------------------------------------------------------------------------------
def view_run(scen: Scenario, rule, par, par2, seed, paths=C.N_VIEW):
    """Ein Bündel von `paths` Pfaden über T_YEARS Jahre mit einer Regel: Verläufe B [T+1, N], H [T, N], Y [T, N] und die Kennzahlen."""
    nz = noise(seed, C.T_YEARS, paths)
    res = A.simulate(rule, [par], scen, nz, par2=None if par2 is None else [par2], record=True)
    o = res["obj"][0]
    return {
        "B": res["B"][:, 0, :], "H": res["H"][:, 0, :], "Y": res["Y"][:, 0, :], "paths": paths,
        "obj": float(o.mean()), "se": float(o.std(ddof=1) / np.sqrt(o.size)), "collapse": float(res["collapse"][0].mean()),
        "extinct": float(res["extinct"][0].mean()), "meanB": float(res["meanB"][0].mean()), "cv": float(res["cv"][0].mean()),
        "overshoot": float(res["overshoot"][0].mean()),
    }


def band(B, lo=10, hi=90):
    """Median und Perzentilband über die Pfade je Jahr (Zeilen: Jahre)."""
    return np.percentile(B, 50, axis=1), np.percentile(B, lo, axis=1), np.percentile(B, hi, axis=1)


# --- Tab 2: Regeln im fairen Vergleich -----------------------------------------------------------------------------------------------------
def textbook_values(r):
    """Lehrbuch-Werte: Quote r K / 4, Aufwand e = r / (2 + r), Escapement K / 2 (aus dem Schaefer-Modell ohne Rauschen)."""
    return {"quote": r / 4.0, "aufwand": r / (2.0 + r), "escape": 0.5}


def compare_rules(scen: Scenario, n_train=C.DEFAULT_TRAIN, n_test=C.DEFAULT_TEST, seed_train=C.DEFAULT_SEED_TRAIN, seed_test=C.DEFAULT_SEED_TEST, cap=C.COLLAPSE_CAP):
    """Alle Regeln auf getrennten Pfaden: abgestimmt auf Trainingspfaden (frei und mit Kollapsgrenze), bewertet auf Testpfaden; dazu die Lehrbuch-Werte, die
    DP-Politik auf dem Messwert und die DP-Referenz bei vollem Wissen. Rückgabe: {"rows": [...], "reference": float | None, "diffs": {...}, "dp_threshold": float | None}."""
    train = noise(seed_train, C.T_YEARS, n_train)
    test = noise(seed_test, C.T_YEARS, n_test)
    rows = []
    for rule in ("quote", "aufwand", "escape", "schwelle"):
        free, capped = A.tune(rule, scen, train, cap)
        for mode, tu in (("frei", free), ("kollaps", capped)):
            ev = A.evaluate(rule, tu.par, tu.par2, scen, test)
            rows.append({"rule": rule, "mode": mode, "name": LABELS[rule], "feasible": tu.feasible, "train_collapse": tu.train_collapse, **ev})
    for rule, par in textbook_values(scen.r).items():
        ev = A.evaluate(rule, par, None, scen, test)
        rows.append({"rule": rule, "mode": "lehrbuch", "name": LABELS[rule], "feasible": True, "train_collapse": None, **ev})
    reference = dp_threshold = None
    if scen.beta < 1.0:
        g, V, Sp = D.solve_cached(scen.r, scen.sg, scen.beta)
        reference = D.annual_value(g, V, scen.beta)
        dp_threshold = D.threshold(Sp)
        ev = A.evaluate("opt", 0.0, None, scen, test, policy=(g, Sp))
        rows.append({"rule": "opt", "mode": "dp", "name": LABELS["opt"], "feasible": True, "train_collapse": None, **ev})
    by = {(r_["rule"], r_["mode"]): r_ for r_ in rows}
    diffs = {}
    for mode in ("frei", "kollaps"):
        diffs[("escape", mode)] = A.paired_difference(by[("escape", mode)]["paths"], by[("aufwand", mode)]["paths"])
        diffs[("schwelle", mode)] = A.paired_difference(by[("schwelle", mode)]["paths"], by[("aufwand", mode)]["paths"])
    return {"rows": rows, "reference": reference, "diffs": diffs, "dp_threshold": dp_threshold, "n_train": n_train, "n_test": n_test, "cap": cap}


def row_of(cmp, rule, mode):
    return next(r for r in cmp["rows"] if r["rule"] == rule and r["mode"] == mode)


# --- Kippstelle ----------------------------------------------------------------------------------------------------------------------------
def kipp(xs, diffs):
    """Erster Vorzeichenwechsel der Differenz von + nach höchstens 0, linear zwischen den Rasterpunkten; None, wenn es keinen gibt."""
    for (x0, d0), (x1, d1) in zip(zip(xs[:-1], diffs[:-1]), zip(xs[1:], diffs[1:])):
        if d0 > 0 >= d1:
            return x0 + (x1 - x0) * d0 / (d0 - d1)
    return None


# --- Verdikte ------------------------------------------------------------------------------------------------------------------------------
def _pct(x):
    return f"{100.0 * x:.1f} %"


def _num(x, nd=4):
    return f"{x:.{nd}f}"


def _signed(x, nd=4):
    return f"{x:+.{nd}f}" if x != 0 else f"{x:.{nd}f}"


def verdict_collapse(collapse, thr, paths, rule_name, cap=C.COLLAPSE_CAP):
    """Kernfrage 1: Hält die gewählte Regel den Bestand? Anteil der Pfade, die irgendwann unter die Kollapsgrenze fallen."""
    head = f"Mit der Regel „{rule_name}“ fällt der Bestand in **{_pct(collapse)}** der {paths} Pfade irgendwann in 200 Jahren unter {thr:.2f} K"
    if collapse <= cap + 1e-12:
        return "success", head + f": höchstens {100 * cap:.0f} %, also innerhalb der Kollapsgrenze, die die Abstimmung mit Kollapsgrenze verlangt."
    if collapse <= 0.5 + 1e-12:
        return "info", head + f": mehr als die {100 * cap:.0f} %, die die Abstimmung mit Kollapsgrenze zuließe, aber nicht die Mehrheit der Pfade."
    return "warning", head + ": die Mehrheit der Pfade. Der Mittelwert des Ziels sagt dazu nichts."


def verdict_hidden(cmp):
    """Kernfrage 2: Versteckt das Ziel allein den Kollaps? Zählt die frei abgestimmten Regeln, deren Kollapsanteil auf den Testpfaden über der Kollapsgrenze liegt."""
    cap = cmp["cap"]
    over = [r for r in cmp["rows"] if r["mode"] == "frei" and r["collapse"] > cap + 1e-12]
    if not over:
        return "success", f"Alle vier frei abgestimmten Regeln halten den Kollaps auf den Testpfaden bei höchstens {100 * cap:.0f} %: hier ist die Kollapsgrenze auf den Testpfaden schon ohne Nebenbedingung eingehalten (ob die Abstimmung mit Kollapsgrenze dieselben Reglerwerte wählt, sagt dieses Verdikt nicht)."
    parts = "; ".join(f"{r['name']} {_pct(r['collapse'])} (Ziel {_num(r['obj'])})" for r in over)
    return "warning", (f"{len(over)} von 4 frei abgestimmten Regeln lassen auf den Testpfaden mehr als {100 * cap:.0f} % der Pfade kollabieren: {parts}. "
                       "Das Ziel allein verrät das nicht, weil die Diskontierung ferne Jahre klein macht.")


def verdict_order(cmp, sm):
    """Kernfrage 3: Escapement-Schwelle oder fester Aufwand, wer liegt bei diesem Messrauschen vorn? Gepaarte Differenz Escapement minus Aufwand."""
    d, se = cmp["diffs"][("escape", "kollaps")]
    d_free, _ = cmp["diffs"][("escape", "frei")]
    esc, auf = row_of(cmp, "escape", "kollaps"), row_of(cmp, "aufwand", "kollaps")
    tail = f" (ohne Kollapsgrenze: {_signed(d_free)}; Standardfehler der Differenz {_num(se)}; {cmp['n_test']} Testpfade)"
    sm_txt = f"{sm:g}"
    cap_txt = f"mit Kollapsgrenze (höchstens {100 * cmp['cap']:.0f} % der Trainingspfade)"
    missed = [name for name, row in (("die Escapement-Schwelle", esc), ("den Aufwand", auf)) if not row["feasible"]]
    if missed:
        cap_txt += f", für {' und '.join(missed)} nicht erreichbar (dort ist das kleinste Kollapsrisiko gewählt)"
    if abs(d) <= 2.0 * se:
        return "info", f"Bei Messrauschen {sm_txt} liegen Escapement-Schwelle und fester Aufwand {cap_txt} gleichauf: Differenz Escapement minus Aufwand {_signed(d)}{tail}."
    if d > 0:
        return "success", f"Bei Messrauschen {sm_txt} liegt die Escapement-Schwelle {cap_txt} vorn: Escapement minus Aufwand {_signed(d)}{tail}. Der Survey ist hier genau genug."
    return "warning", (f"Bei Messrauschen {sm_txt} liegt der feste Aufwand, der den Survey gar nicht liest, {cap_txt} vor der Escapement-Schwelle: "
                       f"Escapement minus Aufwand {_signed(d)}{tail}.")


def verdict_kipp(sm_star, last_sm, kind):
    """Tab 3: Kippstelle der Differenz (Schwelle minus Aufwand) über das Messrauschen. `kind` benennt die Abstimmung, z. B. „mit Kollapsgrenze“."""
    if sm_star is None:
        return "info", f"{kind}: kein Vorzeichenwechsel im Raster bis Messrauschen {last_sm:g}; die Schwelle bleibt vorn oder liegt schon bei exaktem Survey hinten."
    return "info", f"{kind}: die Schwelle liegt bis zu einem Messrauschen von etwa **{sm_star:.2f}** vor dem festen Aufwand und danach dahinter (lineare Interpolation zwischen den Rasterpunkten)."


def verdict_dp(sg, s_dp, s_formula, step):
    """Tab 4: Schwelle der Wertiteration gegen die geschlossene Formel (die Formel gilt nur ohne Wachstumsrauschen)."""
    if s_formula is None:
        return "info", "Die geschlossene Formel gilt nur für 0 < 1/β − 1 < r; für diese Werte gibt es keine Schwelle in (0, 1/2) aus der Formel."
    if sg == 0:
        if abs(s_dp - s_formula) <= step + 1e-12:
            return "success", f"Ohne Wachstumsrauschen liegt die Schwelle der Wertiteration ({s_dp:.4f}) höchstens einen Gitterschritt ({step:.4f}) von der Formel ({s_formula:.4f}) entfernt."
        return "warning", f"Ohne Wachstumsrauschen weicht die Schwelle der Wertiteration ({s_dp:.4f}) um mehr als einen Gitterschritt ({step:.4f}) von der Formel ({s_formula:.4f}) ab."
    return "info", (f"Mit Wachstumsrauschen {sg:.2f} liegt die Schwelle der Wertiteration bei {s_dp:.4f}, die Formel (nur ohne Rauschen) bei {s_formula:.4f}: "
                    f"Differenz {s_dp - s_formula:+.4f}.")
