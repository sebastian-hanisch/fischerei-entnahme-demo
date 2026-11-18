"""Kern der Demo (nur numpy/Standardbibliothek): Messung, Entnahmeregeln, Wachstum, Simulation vieler Pfade, Abstimmung der Regler, Bewertung.

Bestand B_t in Einheiten der Tragfähigkeit K = 1. Reihenfolge in jedem Jahr t = 0 ... T - 1 (Entnahme VOR dem Wachstum):

    1. Messwert  Y_t = B_t * exp(sm * xi_t - sm^2 / 2)                  (Survey; lognormal mit Mittel 1; sm = 0: Y_t = B_t)
    2. Entnahme  H_t = min(Regel(B_t, Y_t), hmax * B_t),  0 <= H_t <= hmax * B_t
    3. Rest      S_t = B_t - H_t                                        (Escapement)
    4. Wachstum  B_{t+1} = Z_t * max(0, S_t + r * S_t * (1 - S_t)),  Z_t = exp(sg * eta_t - sg^2 / 2)   (Mittel 1)
       Die Wachstumsfunktion f(S) = S + r S (1 - S) wird für S > 1 + 1/r negativ (Überschießen); dann gilt B_{t+1} = 0 (Absturz), kein negativer Bestand.
       Das kommt bei den Voreinstellungen nicht vor; `overshoot` zählt die Pfade, auf denen es geschieht.

Regeln (par = Reglerwert, par2 = zweiter Wert der gedämpften Schwelle):
    quote     H = Q                      feste Jahresquote, liest den Survey nicht
    aufwand   H = e * B_t                fester Aufwandsanteil, liest den Survey nicht
    escape    H = max(0, Y_t - S*)       Schwelle auf dem Messwert
    schwelle  H = f * max(0, Y_t - S*)   gedämpfte Schwelle (zwei Regler)
    opt       H = max(0, Y_t - S(Y_t))   Escapement-Politik der Wertiteration (fis_dp), auf dem Messwert angewandt

Ziel J: gleichwertiger Jahresfang. beta < 1: (1 - beta) / (1 - beta^T) * sum_t beta^t H_t über alle Jahre ab 0; beta = 1: Mittel von H_t für t = BURN ... T - 1.
Kollaps: B_t < thr * K irgendwann für t = 1 ... T (nicht absorbierend: der Bestand kann sich erholen). Aussterben: B_t < EXTINCT * K irgendwann."""

from dataclasses import dataclass

import numpy as np

import fis_constants as C
from fis_scenario import Scenario

RULES = ("quote", "aufwand", "escape", "schwelle", "opt")


# --- kleine Einheiten ----------------------------------------------------------------------------------------------------------------------
def measure_factor(xi, sm):
    """Faktor Y / B des Surveys: exp(sm * xi - sm^2 / 2), Mittel 1 bei xi ~ N(0, 1); sm = 0 gibt genau 1."""
    xi = np.asarray(xi, dtype=float)
    return np.exp(sm * xi - 0.5 * sm * sm) if sm > 0 else np.ones_like(xi)


def growth_noise(eta, sg):
    """Wachstumsrauschen Z: exp(sg * eta - sg^2 / 2), Mittel 1; sg = 0 gibt genau 1."""
    eta = np.asarray(eta, dtype=float)
    return np.exp(sg * eta - 0.5 * sg * sg) if sg > 0 else np.ones_like(eta)


def growth_raw(S, r):
    """Wachstumsfunktion f(S) = S + r S (1 - S); für S > 1 + 1/r negativ."""
    return S + r * S * (1.0 - S)


def growth(S, r, z=1.0):
    """Wachstum des Rests: B_next = Z * max(0, f(S)). Ein negativer Wert (Überschießen über 1 + 1/r) wird zu 0: der Bestand stürzt ab, er wird nie negativ."""
    return z * np.maximum(0.0, growth_raw(S, r))


def harvest(rule, par, par2, B, Y, hmax, policy=None):
    """Entnahme H nach Regel, begrenzt auf 0 <= H <= hmax * B. `policy` = (Gitter, Escapement) der Wertiteration für die Regel „opt“."""
    if rule == "quote":
        h = np.minimum(par, B)
    elif rule == "aufwand":
        h = par * B
    elif rule == "escape":
        h = np.minimum(B, np.maximum(0.0, Y - par))
    elif rule == "schwelle":
        h = np.minimum(B, par2 * np.maximum(0.0, Y - par))
    elif rule == "opt":
        grid, esc = policy
        h = np.minimum(B, np.maximum(0.0, Y - np.interp(Y, grid, esc)))
    else:
        raise ValueError(f"unbekannte Regel {rule!r}")
    return np.minimum(h, hmax * B)


def discount_weights(beta, years):
    """Gewichte (1 - beta) / (1 - beta^T) * beta^t, Summe 1 (nur beta < 1)."""
    t = np.arange(years)
    return (1.0 - beta) / (1.0 - beta ** years) * beta ** t


# --- Simulation ----------------------------------------------------------------------------------------------------------------------------
def simulate(rule, par, scen: Scenario, noise_, par2=None, policy=None, record=False, light=False, burn=C.BURN):
    """Simuliert P Reglerwerte (par, ggf. par2) auf denselben N Pfaden. Rückgabe: dict mit Feldern der Form [P, N]:
    obj (Ziel), collapse (0/1), extinct (0/1); ohne `light` zusätzlich meanB, avgcatch, cv, overshoot (0/1: Absturz durch Überschießen). Mit `record` zusätzlich B [T+1, P, N], H [T, P, N], Y [T, P, N]."""
    if not 0.0 < scen.hmax <= 1.0:
        raise ValueError("hmax muss in (0, 1] liegen")
    zg, zm = noise_
    years, n = zg.shape
    if scen.beta >= 1.0 and years <= burn:
        raise ValueError("beta = 1 braucht mehr Jahre als die Einschwingzeit")
    par = np.atleast_1d(np.asarray(par, dtype=float))[:, None]
    par2 = None if par2 is None else np.atleast_1d(np.asarray(par2, dtype=float))[:, None]
    p = par.shape[0]
    gfac = growth_noise(zg, scen.sg)           # [T, N]
    mfac = measure_factor(zm, scen.sm)         # [T, N]
    weights = discount_weights(scen.beta, years) if scen.beta < 1.0 else None
    B = np.full((p, n), C.B0)
    obj = np.zeros((p, n))
    bmin = np.full((p, n), np.inf)
    s_c = np.zeros((p, n))
    s_c2 = np.zeros((p, n))
    s_b = np.zeros((p, n))
    over = np.zeros((p, n), dtype=bool)
    if record:
        rec_b = np.empty((years + 1, p, n))
        rec_h = np.empty((years, p, n))
        rec_y = np.empty((years, p, n))
        rec_b[0] = B
    for t in range(years):
        Y = B * mfac[t]
        H = harvest(rule, par, par2, B, Y, scen.hmax, policy)
        if weights is not None:
            obj += weights[t] * H
        elif t >= burn:
            obj += H / (years - burn)
        if t >= burn and not light:
            s_c += H
            s_c2 += H * H
            s_b += B
        raw = growth_raw(B - H, scen.r)
        if not light:
            over |= raw < 0.0
        B = gfac[t] * np.maximum(0.0, raw)
        np.minimum(bmin, B, out=bmin)
        if record:
            rec_h[t] = H
            rec_y[t] = Y
            rec_b[t + 1] = B
    out = {"obj": obj, "collapse": (bmin < scen.thr * C.K).astype(float), "extinct": (bmin < C.EXTINCT * C.K).astype(float)}
    if not light:
        m = years - burn
        mean_c = s_c / m
        var_c = np.maximum(s_c2 / m - mean_c ** 2, 0.0)
        out["meanB"] = s_b / m
        out["avgcatch"] = mean_c
        out["cv"] = np.sqrt(var_c) / np.maximum(mean_c, 1e-12)
        out["overshoot"] = over.astype(float)
    if record:
        out.update(B=rec_b, H=rec_h, Y=rec_y)
    return out


# --- Abstimmung und Bewertung --------------------------------------------------------------------------------------------------------------
def _axis(spec):
    lo, hi, step = spec
    n = int(round((hi - lo) / step)) + 1
    return np.round(lo + step * np.arange(n), 10)[:: max(1, int(C.TUNE_STRIDE))]


def tuning_grid(rule):
    """Gitter der Reglerwerte (par, par2 oder None) einer Regel."""
    if rule == "quote":
        return _axis(C.GRID_Q), None
    if rule == "aufwand":
        return _axis(C.GRID_E), None
    if rule == "escape":
        return _axis(C.GRID_S), None
    if rule == "schwelle":
        ss, ff = np.meshgrid(_axis(C.GRID_DAMPED_S), _axis(C.GRID_DAMPED_F), indexing="ij")
        return ss.ravel(), ff.ravel()
    raise ValueError(f"keine Abstimmung für die Regel {rule!r}")


@dataclass(frozen=True)
class Tuned:
    rule: str
    par: float
    par2: float | None
    train_obj: float
    train_collapse: float
    feasible: bool        # False: nichts erfüllte die Nebenbedingung, gewählt wurde das kleinste Kollapsrisiko


def pick(obj_mean, collapse_mean, cap):
    """Index des besten Reglers: größtes mittleres Ziel; mit `cap` unter der Nebenbedingung Kollaps <= cap, ist nichts zulässig, das kleinste Kollapsrisiko
    (bei Gleichstand der kleinste Index). Rückgabe (Index, zulässig)."""
    if cap is None:
        return int(np.argmax(obj_mean)), True
    ok = collapse_mean <= cap
    if ok.any():
        return int(np.argmax(np.where(ok, obj_mean, -np.inf))), True
    return int(np.argmax(-collapse_mean)), False


def tune(rule, scen: Scenario, train_noise, cap=C.COLLAPSE_CAP):
    """Stimmt eine Regel auf den Trainingspfaden ab: (frei, mit Kollapsgrenze) als zwei `Tuned`. Ein einziger Simulationslauf für beide."""
    p1, p2 = tuning_grid(rule)
    res = simulate(rule, p1, scen, train_noise, par2=p2, light=True)
    om, cm = res["obj"].mean(axis=1), res["collapse"].mean(axis=1)
    out = []
    for c in (None, cap):
        i, ok = pick(om, cm, c)
        out.append(Tuned(rule, float(p1[i]), None if p2 is None else float(p2[i]), float(om[i]), float(cm[i]), ok))
    return out[0], out[1]


def evaluate(rule, par, par2, scen: Scenario, test_noise, policy=None):
    """Bewertung eines Reglers auf Testpfaden: Mittel und Standardfehler des Ziels, Kollapsanteil, Aussterben, mittlerer Bestand, Variationskoeffizient des Fangs."""
    res = simulate(rule, [par], scen, test_noise, par2=None if par2 is None else [par2], policy=policy)
    o = res["obj"][0]
    return {
        "rule": rule, "par": float(par), "par2": None if par2 is None else float(par2), "obj": float(o.mean()), "se": float(o.std(ddof=1) / np.sqrt(o.size)),
        "collapse": float(res["collapse"][0].mean()), "extinct": float(res["extinct"][0].mean()), "meanB": float(res["meanB"][0].mean()),
        "cv": float(res["cv"][0].mean()), "overshoot": float(res["overshoot"][0].mean()), "paths": o,
    }


def paired_difference(a, b):
    """Mittel und Standardfehler der pfadweisen Differenz a - b (gleiche Zufallszahlen)."""
    d = np.asarray(a) - np.asarray(b)
    return float(d.mean()), float(d.std(ddof=1) / np.sqrt(d.size))
