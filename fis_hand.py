"""Handbeispiel in exakten Brüchen (fractions.Fraction): das Modell ohne Zufall (σ_g = σ_m = 0), r = 1/2, K = 1, Kapazitätsgrenze 9/10.

Dieselben Regeln wie fis_algorithm, aber für einen einzigen Pfad und ohne numpy, damit sich jede Zahl von Hand nachrechnen lässt (tests/):
    H = min(Regel, 9/10 B)   Quote Q = 1/8 (= r K / 4), Aufwand e = 1/5 (= r / (2 + r)), Escapement S* = 9/20   (ohne Zufall ist der Survey exakt, Y = B)
    S = B - H,   B_next = S + r S (1 - S).
Gleichgewichte: Aufwand e = r / (2 + r): B* = K/2 + r K/4 = 5/8, Fang r K / 4 = 1/8; Quote Q = r K / 4: Fixpunkt B = 5/8 (Rest 1/2) und instabil nach unten.
Optimum bei Diskontfaktor 19/20 (ohne Rauschen): S* = (1 - (1/β - 1)/r) / 2 = 17/38 und V(1) = (1 - S*) + β/(1 - β) r S* (1 - S*) = 441/152."""

from fractions import Fraction as Fr

R = Fr(1, 2)
HMAX = Fr(9, 10)
BETA = Fr(19, 20)
RULES = {"Quote Q = 1/8": ("quote", Fr(1, 8)), "Aufwand e = 1/5": ("aufwand", Fr(1, 5)), "Escapement S* = 9/20": ("escape", Fr(9, 20))}


def harvest(rule, par, B, hmax=HMAX):
    """Entnahme ohne Zufall (Survey exakt): Quote min(Q, B), Aufwand e B, Escapement max(0, B - S*); danach höchstens hmax B."""
    if rule == "quote":
        h = min(par, B)
    elif rule == "aufwand":
        h = par * B
    elif rule == "escape":
        h = max(type(B)(0), B - par)
    else:
        raise ValueError(rule)
    return min(h, hmax * B)


def next_stock(B, rule, par, r=R, hmax=HMAX):
    """(Entnahme, Rest, Bestand im nächsten Jahr) eines Jahres."""
    h = harvest(rule, par, B, hmax)
    s = B - h
    return h, s, s + r * s * (1 - s)


def orbit(rule, par, b0, years, r=R, hmax=HMAX):
    """Liste (B_t, H_t) für t = 0 ... years - 1; mit Fraction-Eingaben exakt, mit Zahlen gleitend."""
    out = []
    b = b0
    for _ in range(years):
        h, _s, b_next = next_stock(b, rule, par, r, hmax)
        out.append((b, h))
        b = b_next
    return out


def effort_equilibrium(r=R):
    """Gleichgewicht bei Aufwand e = r / (2 + r): Bestand K/2 + r K/4 und Fang r K / 4 (K = 1)."""
    return Fr(1, 2) + r / 4, r / 4


def optimum_closed(r=R, beta=BETA):
    """(S*, V(1)) in Brüchen: S* = (1 - (1/β - 1)/r)/2, V(1) = (1 - S*) + β/(1 - β) r S* (1 - S*)."""
    s = (1 - (1 / beta - 1) / r) / 2
    return s, (1 - s) + beta / (1 - beta) * r * s * (1 - s)


def annual_value(r=R, beta=BETA):
    """(1 - β) V(1): der gleichwertige Jahresfang des Optimums."""
    return (1 - beta) * optimum_closed(r, beta)[1]
