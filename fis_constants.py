"""Konstanten: Regler-Grenzen, feste Annahmen, Farben, Presets. ALLE Modellzahlen sind Annahmen (kein Bestand, keine Fischerei- oder Jagddaten).

Bestand in Einheiten der Tragfähigkeit K (K = 1), Jahrestakt. Die Regeln und das Modell stehen in fis_algorithm.py und im 📐-Expander der App."""

# --- Zeitraum und Startwert ------------------------------------------------------------------------------------------------------------------
K = 1.0                  # Tragfähigkeit: Einheit des Bestands
T_YEARS = 200            # simulierte Jahre je Pfad
BURN = 50                # Einschwingzeit: Mittel, Fang-Streuung und mittlerer Bestand zählen erst ab diesem Jahr; das Ziel zählt ab Jahr 0
B0 = 1.0                 # Anfangsbestand B_0 = K (der Anfangsbestand gehört zum Problem)
EXTINCT = 1e-4           # „Aussterben“: Bestand unter 1e-4 K irgendwann
COLLAPSE_CAP = 0.10      # Abstimmung „mit Kollapsgrenze“: höchstens 10 % der Trainingspfade dürfen kollabieren

# --- Pfade -----------------------------------------------------------------------------------------------------------------------------------
N_VIEW = 500                                  # Pfade im Bündel des ersten Tabs
SEED_MAX = 999999
DEFAULT_SEED_VIEW = 5
TRAIN_MIN, TRAIN_MAX, DEFAULT_TRAIN = 200, 2000, 1000
TEST_MIN, TEST_MAX, DEFAULT_TEST = 1000, 10000, 5000
DEFAULT_SEED_TRAIN, DEFAULT_SEED_TEST = 11, 22  # Abstimmung auf getrennten Pfaden, Bewertung auf frischen Pfaden

# --- Modellregler ----------------------------------------------------------------------------------------------------------------------------
R_MIN, R_MAX, DEFAULT_R = 0.2, 1.5, 0.5          # Wachstumsrate; bis r = 1 ist die Wachstumsfunktion auf [0, 1] monoton
SG_MIN, SG_MAX, DEFAULT_SG = 0.0, 0.4, 0.2       # Wachstumsrauschen (Standardabweichung auf der Logskala)
SM_MIN, SM_MAX, DEFAULT_SM = 0.0, 1.0, 0.2       # Messrauschen des Surveys (Standardabweichung auf der Logskala)
BETA_OPTIONS = (0.9, 0.95, 0.98, 1.0)            # Diskontfaktor; 1,0 = Mittel der Jahre BURN bis T_YEARS - 1
DEFAULT_BETA = 0.95
HMAX_MIN, HMAX_MAX, DEFAULT_HMAX = 0.5, 1.0, 0.9  # Kapazitätsgrenze: Entnahme höchstens h_max mal Bestand
THR_MIN, THR_MAX, DEFAULT_THR = 0.05, 0.20, 0.10  # Kollapsgrenze in K: Bestand unter dieser Grenze irgendwann in den Jahren 1 bis T

# --- Regeln und Reglerwerte ------------------------------------------------------------------------------------------------------------------
RULE_QUOTE, RULE_EFFORT, RULE_ESCAPE, RULE_DAMPED = 0, 1, 2, 3
RULE_NAMES = {RULE_QUOTE: "Quote", RULE_EFFORT: "Aufwand", RULE_ESCAPE: "Escapement-Schwelle", RULE_DAMPED: "Gedämpfte Schwelle"}
DEFAULT_RULE = RULE_DAMPED
Q_MIN, Q_MAX, DEFAULT_Q = 0.0, 0.4, 0.06
E_MIN, E_MAX, DEFAULT_E = 0.0, 0.9, 0.22
S_MIN, S_MAX, DEFAULT_S = 0.0, 1.0, 0.38
F_MIN, F_MAX, DEFAULT_F = 0.1, 1.0, 0.7

# Abstimmungsgitter (Training): Quote Q, Aufwand e, Escapement S*, gedämpfte Schwelle S* mal f
GRID_Q = (0.0, 0.4, 0.005)
GRID_E = (0.0, 0.9, 0.01)
GRID_S = (0.0, 1.0, 0.01)
GRID_DAMPED_S = (0.0, 1.0, 0.02)
GRID_DAMPED_F = (0.1, 1.0, 0.1)
TUNE_STRIDE = 1          # nur Tests setzen das größer (gröbere Gitter, kürzere Läufe); die App und die Messungen nutzen 1

# --- Wertiteration (Tab „Das Optimum“) -------------------------------------------------------------------------------------------------------
DP_POINTS = 801          # Gitterpunkte für den Bestand auf [0, DP_BMAX]
DP_BMAX = 2.0
DP_NODES = 41            # Gauß-Hermite-Knoten für das Wachstumsrauschen
DP_COST_OPTIONS = (0.0, 0.02, 0.05, 0.1, 0.2)    # Aufwandkosten c je Jahr: c * ln(B/S) bei Preis 1
DEFAULT_DP_COST = 0.0

# --- Vorgerechnetes Raster „Wo kippt es?“ -----------------------------------------------------------------------------------------------------
GRID_SM = (0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.375, 0.4, 0.5, 0.6, 0.8)
GRID_BETA = 0.95
GRID_TRAIN, GRID_TEST = 1000, 5000
GRID_SEED_TRAIN, GRID_SEED_TEST = 11, 22
# Einstellungen des Rasters: (Kennung, Beschriftung, r, sg, thr, hmax); der Grundfall steht zuerst
GRID_SCENARIOS = (
    ("grund", "Grundfall (r = 0,5; σ_g = 0,2; Kollapsgrenze 0,1 K; Kapazitätsgrenze 0,9)", 0.5, 0.2, 0.10, 0.9),
    ("thr05", "Kollapsgrenze 0,05 K (sonst Grundfall)", 0.5, 0.2, 0.05, 0.9),
    ("thr20", "Kollapsgrenze 0,2 K (sonst Grundfall)", 0.5, 0.2, 0.20, 0.9),
    ("hmax70", "Kapazitätsgrenze 0,7 (sonst Grundfall)", 0.5, 0.2, 0.10, 0.7),
    ("hmax100", "Kapazitätsgrenze 1,0, also keine (sonst Grundfall)", 0.5, 0.2, 0.10, 1.0),
    ("sg05", "Wachstumsrauschen σ_g = 0,05 (sonst Grundfall)", 0.5, 0.05, 0.10, 0.9),
    ("sg10", "Wachstumsrauschen σ_g = 0,1 (sonst Grundfall)", 0.5, 0.10, 0.10, 0.9),
    ("sg30", "Wachstumsrauschen σ_g = 0,3 (sonst Grundfall)", 0.5, 0.30, 0.10, 0.9),
    ("sg40", "Wachstumsrauschen σ_g = 0,4 (sonst Grundfall)", 0.5, 0.40, 0.10, 0.9),
    ("r02", "Wachstumsrate r = 0,2 (sonst Grundfall)", 0.2, 0.2, 0.10, 0.9),
    ("r03", "Wachstumsrate r = 0,3 (sonst Grundfall)", 0.3, 0.2, 0.10, 0.9),
    ("r08", "Wachstumsrate r = 0,8 (sonst Grundfall)", 0.8, 0.2, 0.10, 0.9),
    ("r10", "Wachstumsrate r = 1,0 (sonst Grundfall)", 1.0, 0.2, 0.10, 0.9),
)

# --- Farben ----------------------------------------------------------------------------------------------------------------------------------
COLOR_STOCK = "#2F6B65"
COLOR_BAND = "rgba(47,107,101,0.20)"
COLOR_PATH = "#4c78a8"
COLOR_CATCH = "#d98b2b"
COLOR_LINE = "#b8433b"
COLOR_GREY = "rgba(120,120,120,0.9)"
RULE_COLORS = {"quote": "#b8433b", "aufwand": "#4c78a8", "escape": "#d98b2b", "schwelle": "#2F6B65", "opt": "#7a5195", "dp": "#555555"}

# --- Presets (Schnellstart) -------------------------------------------------------------------------------------------------------------------
# Schlüssel: url_param der Regler (fis_presets.SETTING_SPECS); was ein Preset nicht nennt, geht auf den Standardwert.
PRESETS = {
    "Grundfall (Voreinstellung)": {},
    "Lehrbuch: Quote am MSY": {"rule": RULE_QUOTE, "q": 0.125, "sg": 0.01},
    "Aufwand stabil": {"rule": RULE_EFFORT, "e": 0.20},
    "Escapement bei exaktem Survey": {"rule": RULE_ESCAPE, "s": 0.45, "sm": 0.0},
    "Verrauschter Survey": {"rule": RULE_ESCAPE, "s": 0.47, "sm": 0.3},
    "Schneller Bestand (r = 1)": {"rule": RULE_EFFORT, "e": 0.35, "r": 1.0},
}
PRESET_HELP = {
    "Grundfall (Voreinstellung)": "r = 0,5, Wachstumsrauschen 0,2, Messrauschen 0,2, β = 0,95, Kapazitätsgrenze 0,9, Kollapsgrenze 0,1 K; gedämpfte Schwelle mit S* = 0,38 und Dämpfung 0,7 (die auf Trainingspfaden abgestimmten Werte aus Tab 2). Alles Annahmen.",
    "Lehrbuch: Quote am MSY": "Feste Jahresquote Q = r K / 4 = 0,125 K bei r = 0,5, mit Wachstumsrauschen nur 0,01: auf 5 000 Testpfaden (Zufalls-Seed 22) fallen schon dann rund 40 % der Pfade unter die Kollapsgrenze (README).",
    "Aufwand stabil": "Fester Aufwandsanteil e = r / (2 + r) = 0,2 des Bestands je Jahr: ohne Rauschen kehrt der Bestand von jedem Start auf 0,625 K zurück; mit Wachstumsrauschen 0,2 kollabieren 0,9 % der 5 000 Testpfade (README).",
    "Escapement bei exaktem Survey": "Messrauschen 0 und Schwelle S* = 0,45 K, nahe der Schwelle der Wertiteration (0,4525 im Grundfall, Tab 4): auf 5 000 Testpfaden kollabiert kein Pfad (README).",
    "Verrauschter Survey": "Messrauschen 0,3 und die frei abgestimmte Escapement-Schwelle S* = 0,47 K: das Ziel liegt auf 5 000 Testpfaden bei 0,13, aber rund 95 % der Pfade kollabieren irgendwann (README).",
    "Schneller Bestand (r = 1)": "Wachstumsrate 1,0 und fester Aufwand e = 0,35: auf 5 000 Testpfaden Ziel 0,25 ohne Kollaps (README); bei schnell wachsendem Bestand bringt das Lesen eines verrauschten Surveys wenig (Tab 3, Einstellung r = 1,0).",
}
