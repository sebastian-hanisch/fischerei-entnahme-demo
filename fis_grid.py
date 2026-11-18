"""Vorgerechnetes Raster „Wo kippt es?“: je Einstellung (r, σ_g, Kollapsgrenze, Kapazitätsgrenze) und je Messrauschen σ_m werden Aufwand, Escapement-Schwelle und
gedämpfte Schwelle neu abgestimmt (Trainingspfade, Zufalls-Seed 11, 1 000 Pfade; frei und mit Kollapsgrenze) und auf Testpfaden bewertet (Zufalls-Seed 22, 5 000 Pfade).

Vorgerechnet (python fis_grid.py [Anzahl Prozesse]), weil jede Zelle mehrere Abstimmungen über Tausende Pfade rechnet (Größenordnung Minuten für das ganze Raster).
Der Aufwand liest den Survey nicht: er wird je Einstellung einmal abgestimmt und für jedes σ_m wiederverwendet (seine Pfade sind von σ_m unabhängig).
Zeilen (data/kipp.csv): scen, sm, rule, mode, par, par2, obj, se, collapse, extinct, overshoot, feasible, d_vs_aufwand, d_se
  rule: aufwand | escape | schwelle | opt (DP-Politik auf dem Messwert) | ref (DP-Referenz bei vollem Wissen, nur Spalte obj)
  mode: frei | kollaps (Abstimmung mit höchstens COLLAPSE_CAP kollabierenden Trainingspfaden) | dp
  d_vs_aufwand: pfadweise gepaarte Differenz zum Aufwand mit derselben Abstimmung (bei der DP-Politik: zum freien Aufwand)."""

import csv
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace

import fis_algorithm as A
import fis_constants as C
import fis_dp as D
from fis_scenario import Scenario, noise

GRID_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "kipp.csv")
FIELDS = ("scen", "sm", "rule", "mode", "par", "par2", "obj", "se", "collapse", "extinct", "overshoot", "feasible", "d_vs_aufwand", "d_se")


def _row(scen_id, sm, rule, mode, ev, feasible, d=0.0, dse=0.0):
    return {
        "scen": scen_id, "sm": sm, "rule": rule, "mode": mode, "par": round(ev["par"], 6), "par2": "" if ev["par2"] is None else round(ev["par2"], 6),
        "obj": round(ev["obj"], 6), "se": round(ev["se"], 6), "collapse": round(ev["collapse"], 6), "extinct": round(ev["extinct"], 6),
        "overshoot": round(ev["overshoot"], 6), "feasible": int(feasible), "d_vs_aufwand": round(d, 6), "d_se": round(dse, 6),
    }


def scenario_rows(spec):
    """Alle Zeilen einer Einstellung des Rasters. spec = (Kennung, Beschriftung, r, sg, thr, hmax) aus C.GRID_SCENARIOS."""
    scen_id, _label, r, sg, thr, hmax = spec
    base = Scenario(r=r, sg=sg, sm=0.0, beta=C.GRID_BETA, hmax=hmax, thr=thr)
    train = noise(C.GRID_SEED_TRAIN, C.T_YEARS, C.GRID_TRAIN)
    test = noise(C.GRID_SEED_TEST, C.T_YEARS, C.GRID_TEST)
    rows = []
    auf = {}
    for mode, tu in zip(("frei", "kollaps"), A.tune("aufwand", base, train)):
        auf[mode] = (tu, A.evaluate("aufwand", tu.par, None, base, test))
    g, V, Sp = D.solve_cached(r, sg, C.GRID_BETA)
    ref = D.annual_value(g, V, C.GRID_BETA)
    for sm in C.GRID_SM:
        sc = replace(base, sm=sm)
        for mode in ("frei", "kollaps"):
            tu, ev = auf[mode]
            rows.append(_row(scen_id, sm, "aufwand", mode, ev, tu.feasible))
        for rule in ("escape", "schwelle"):
            for mode, tu in zip(("frei", "kollaps"), A.tune(rule, sc, train)):
                ev = A.evaluate(rule, tu.par, tu.par2, sc, test)
                d, dse = A.paired_difference(ev["paths"], auf[mode][1]["paths"])
                rows.append(_row(scen_id, sm, rule, mode, ev, tu.feasible, d, dse))
        ev = A.evaluate("opt", 0.0, None, sc, test, policy=(g, Sp))
        d, dse = A.paired_difference(ev["paths"], auf["frei"][1]["paths"])
        rows.append(_row(scen_id, sm, "opt", "dp", ev, True, d, dse))
        rows.append({"scen": scen_id, "sm": sm, "rule": "ref", "mode": "dp", "par": "", "par2": "", "obj": round(ref, 6), "se": 0, "collapse": "", "extinct": "",
                     "overshoot": "", "feasible": 1, "d_vs_aufwand": "", "d_se": ""})
    return rows


def build(path=GRID_PATH, workers=1, verbose=True):
    specs = list(C.GRID_SCENARIOS)
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            results = list(ex.map(scenario_rows, specs))
    else:
        results = []
        for spec in specs:
            results.append(scenario_rows(spec))
            if verbose:
                print("fertig:", spec[0], flush=True)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=FIELDS)
        wr.writeheader()
        for rows in results:
            wr.writerows(rows)
    return results


def _f(x):
    return None if x == "" else float(x)


def load(path=GRID_PATH):
    """Raster als Liste von dicts (Zahlen als float, leere Felder None); leere Liste, wenn die Datei fehlt."""
    if not os.path.exists(path):
        return []
    out = []
    with open(path, encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            out.append({
                "scen": r["scen"], "sm": float(r["sm"]), "rule": r["rule"], "mode": r["mode"], "par": _f(r["par"]), "par2": _f(r["par2"]), "obj": float(r["obj"]),
                "se": float(r["se"]), "collapse": _f(r["collapse"]), "extinct": _f(r["extinct"]), "overshoot": _f(r["overshoot"]), "feasible": bool(int(r["feasible"])),
                "d_vs_aufwand": _f(r["d_vs_aufwand"]), "d_se": _f(r["d_se"]),
            })
    return out


def series(grid, scen_id, rule, mode):
    """Zeilen einer Einstellung, Regel und Abstimmung, aufsteigend nach σ_m."""
    return sorted((r for r in grid if r["scen"] == scen_id and r["rule"] == rule and r["mode"] == mode), key=lambda r: r["sm"])


if __name__ == "__main__":
    build(workers=int(sys.argv[1]) if len(sys.argv) > 1 else 1)
