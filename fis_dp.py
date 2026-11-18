"""Wertiteration (Optimum bei vollem Wissen über den Bestand) und die geschlossene Formel für den Fall ohne Wachstumsrauschen.

V(B) = max_{0 <= S <= B} [ (B - S) + beta * E V(Z * f(S)) ],  f(S) = S + r S (1 - S),  Z = exp(sg * eta - sg^2 / 2).
Gitter für B auf [0, bmax], B_next über bmax hinaus wird auf bmax gesetzt, V dazwischen linear interpoliert; E über Gauß-Hermite-Knoten.
Ohne Kosten gilt V(B) = B + max_{S <= B} [W(S) - S] mit W(S) = beta * E V(Z f(S)): ein laufendes Maximum, die Politik ist eine Escapement-Schwelle.
Mit Aufwandkosten c * ln(B / S) je Jahr (Preis 1; Fang B - S = B (1 - exp(-q E)) mit Aufwand E = ln(B / S) / q) wird das Maximum über alle S <= B gebildet."""

import numpy as np
from numpy.polynomial.hermite_e import hermegauss

import fis_constants as C


def quadrature(sg, nodes=C.DP_NODES):
    """Faktoren Z_k und Gewichte w_k (Summe 1) der Gauß-Hermite-Näherung von Z = exp(sg * eta - sg^2 / 2); sg = 0: der einzige Knoten Z = 1."""
    if sg > 0:
        x, w = hermegauss(nodes)
        return np.exp(sg * x - 0.5 * sg * sg), w / w.sum()
    return np.array([1.0]), np.array([1.0])


def value_iteration(r, sg, beta, n=C.DP_POINTS, bmax=C.DP_BMAX, nodes=C.DP_NODES, cost=0.0, tol=1e-11, maxit=20000):
    """Rückgabe (Gitter g, Wert V, Escapement-Politik Sp(B)). beta muss unter 1 liegen (ohne Diskontierung ist V unendlich)."""
    if not 0.0 < beta < 1.0:
        raise ValueError("Die Wertiteration braucht 0 < beta < 1")
    g = np.linspace(0.0, bmax, n)
    f = g + r * g * (1.0 - g)
    zk, w = quadrature(sg, nodes)
    nxt = np.minimum(f[:, None] * zk[None, :], bmax)          # [S, k]
    V = g.copy()

    def cont(v):
        return beta * (np.interp(nxt, g, v) * w[None, :]).sum(axis=1)

    if cost == 0.0:
        for _ in range(maxit):
            Vn = g + np.maximum.accumulate(cont(V) - g)
            done = np.max(np.abs(Vn - V)) < tol
            V = Vn
            if done:
                break
        a = cont(V) - g
        idx = np.zeros(n, dtype=int)
        best, bi = -np.inf, 0
        for i in range(n):
            if a[i] > best + 1e-13:      # Gleichstand: der kleinste Rest (die größte Entnahme) bleibt
                best, bi = a[i], i
            idx[i] = bi
        return g, V, g[idx]
    ii, jj = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")     # i = B, j = S
    mask = jj <= ii
    with np.errstate(divide="ignore", invalid="ignore"):
        lr = np.where((jj > 0) & (ii > 0) & mask, np.log(g[ii] / np.where(g[jj] > 0, g[jj], 1.0)), 0.0)
    rew = np.where(mask, (g[ii] - g[jj]) - cost * lr, -np.inf)
    rew = np.where(mask & (jj == 0) & (ii > 0), -np.inf, rew)          # S = 0 hieße unendlicher Aufwand
    rew = np.where((ii == 0) & (jj == 0), 0.0, rew)
    for _ in range(maxit):
        Vn = (rew + cont(V)[None, :]).max(axis=1)
        done = np.max(np.abs(Vn - V)) < tol
        V = Vn
        if done:
            break
    return g, V, g[(rew + cont(V)[None, :]).argmax(axis=1)]


def threshold(esc):
    """Schwelle S* der Politik: der Rest bei den größten Bestandswerten des Gitters (dort gilt die Schwelle)."""
    return float(esc[-1])


def annual_value(g, V, beta, b=C.B0):
    """(1 - beta) * V(b): der gleichwertige Jahresfang bei Anfangsbestand b (Einheit K je Jahr)."""
    return float((1.0 - beta) * np.interp(b, g, V))


def s_star_formula(r, beta):
    """Goldene-Regel-Schwelle ohne Wachstumsrauschen: beta * f'(S) = 1 mit f'(S) = 1 + r (1 - 2 S), also S* = (1 - (1 / beta - 1) / r) / 2.
    Gilt für 0 < 1 / beta - 1 < r (sonst liegt S* nicht in (0, 1/2))."""
    return 0.5 * (1.0 - (1.0 / beta - 1.0) / r)


def value_formula(r, beta, s_star=None, b=C.B0):
    """V(b) ohne Wachstumsrauschen: (b - S*) + beta / (1 - beta) * r S* (1 - S*), für b >= S*."""
    s = s_star_formula(r, beta) if s_star is None else s_star
    return (b - s) + beta / (1.0 - beta) * r * s * (1.0 - s)


_CACHE = {}


def solve_cached(r, sg, beta, cost=0.0, n=None, nodes=None):
    """Wertiteration mit Zwischenspeicher (gleiche Eingabe, dieselben Arrays; nur lesbar). Gitterpunkte und Knoten lesen die Konstanten beim Aufruf."""
    n = C.DP_POINTS if n is None else n
    nodes = C.DP_NODES if nodes is None else nodes
    key = (round(r, 6), round(sg, 6), round(beta, 6), round(cost, 6), n, nodes)
    if key not in _CACHE:
        if len(_CACHE) >= 24:
            _CACHE.pop(next(iter(_CACHE)))
        res = value_iteration(r, sg, beta, n=n, nodes=nodes, cost=cost)
        for a in res:
            a.flags.writeable = False
        _CACHE[key] = res
    return _CACHE[key]
