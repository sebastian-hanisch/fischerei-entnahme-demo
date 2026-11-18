"""Szenario: die Modellgrößen eines Laufs und die Zufallszahlen der Pfade.

Zufall: je Pfad und Jahr zwei unabhängige Standardnormalzahlen, eine für das Wachstum (Z_t) und eine für den Survey (Y_t). Dieselben Zahlen gehen in
jede Regel (gemeinsame Zufallszahlen), Differenzen zwischen Regeln sind deshalb gepaart."""

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

import fis_constants as C


@dataclass(frozen=True)
class Scenario:
    r: float = C.DEFAULT_R              # Wachstumsrate
    sg: float = C.DEFAULT_SG            # Wachstumsrauschen (Logskala)
    sm: float = C.DEFAULT_SM            # Messrauschen des Surveys (Logskala)
    beta: float = C.DEFAULT_BETA        # Diskontfaktor; 1,0 = Mittel der Jahre BURN bis T - 1
    hmax: float = C.DEFAULT_HMAX        # Kapazitätsgrenze: Entnahme höchstens hmax * Bestand
    thr: float = C.DEFAULT_THR          # Kollapsgrenze in K


@lru_cache(maxsize=8)
def noise(seed: int, years: int, paths: int):
    """(Zg, Zm), je [years, paths] Standardnormalzahlen; nur lesbar, weil der Zwischenspeicher sie teilt."""
    rng = np.random.default_rng(seed)
    zg = rng.standard_normal((years, paths))
    zm = rng.standard_normal((years, paths))
    zg.flags.writeable = False
    zm.flags.writeable = False
    return zg, zm
