"""Plotly-Abbildungen. Alle Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht am Diagramm hängen bleiben.
Keine Abbildung nutzt Auswahl (on_select); Beschriftungen liegen als Text an den Marken oder als Annotation."""

import numpy as np
import plotly.graph_objects as go

import fis_constants as C

SHORT = {"quote": "Quote", "aufwand": "Aufwand", "escape": "Escapement", "schwelle": "Gedämpfte Schwelle", "opt": "DP-Politik"}
MODE_SYMBOL = {"frei": "circle", "kollaps": "diamond", "lehrbuch": "x", "dp": "star"}
MODE_TEXT = {"frei": "frei", "kollaps": "mit Kollapsgrenze", "lehrbuch": "Lehrbuch", "dp": ""}


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height=360):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=-0.22), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def _hline(fig, y, text, color=C.COLOR_GREY, dash="dash"):
    fig.add_hline(y=y, line=dict(color=color, width=1.5, dash=dash), annotation_text=text, annotation_position="top left", annotation_font_size=11)


# --- Tab 1 ---------------------------------------------------------------------------------------------------------------------------------
def fig_fan(med, lo, hi, thr, ymax):
    """Median und 10-bis-90-%-Band des Bestands über die Jahre; K und die Kollapsgrenze als Linien."""
    years = list(range(len(med)))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=years + years[::-1], y=list(hi) + list(lo)[::-1], fill="toself", fillcolor=C.COLOR_BAND, line=dict(width=0), hoverinfo="skip",
                             name="10 % bis 90 % der Pfade"))
    fig.add_trace(go.Scatter(x=years, y=list(med), mode="lines", line=dict(color=C.COLOR_STOCK, width=3), name="Median", hovertemplate="Jahr %{x}: %{y:.3f} K<extra></extra>"))
    _hline(fig, C.K, "Tragfähigkeit K", dash="dot")
    _hline(fig, thr, "Kollapsgrenze", color=C.COLOR_LINE)
    fig.update_xaxes(title_text="Jahr", range=[0, len(med) - 1])
    fig.update_yaxes(title_text="Bestand (K)", range=[0, ymax])
    return _base(fig)


def fig_single(B, H, Y, thr, ymax):
    """Ein Pfad: Bestand B_t (Linie), Messwert Y_t (Punkte) und Entnahme H_t (Säulen) je Jahr; B hat ein Jahr mehr als H und Y (B_T am Ende)."""
    years = list(range(len(B)))
    fig = go.Figure()
    fig.add_trace(go.Bar(x=years[:-1], y=list(H), marker_color=C.COLOR_CATCH, name="Entnahme H", hovertemplate="Jahr %{x}: %{y:.3f} K<extra></extra>"))
    fig.add_trace(go.Scatter(x=years, y=list(B), mode="lines", line=dict(color=C.COLOR_STOCK, width=2.5), name="Bestand B", hovertemplate="Jahr %{x}: %{y:.3f} K<extra></extra>"))
    fig.add_trace(go.Scatter(x=years[:-1], y=list(Y), mode="markers", marker=dict(color=C.COLOR_PATH, size=4), name="Messwert Y (Survey)",
                             hovertemplate="Jahr %{x}: %{y:.3f} K<extra></extra>"))
    _hline(fig, thr, "Kollapsgrenze", color=C.COLOR_LINE)
    fig.update_xaxes(title_text="Jahr", range=[-1, len(B)])
    fig.update_yaxes(title_text="Bestand und Entnahme (K)", range=[0, ymax])
    return _base(fig)


# --- Tab 2 ---------------------------------------------------------------------------------------------------------------------------------
def fig_tradeoff(rows, reference, cap):
    """Ziel gegen Kollapsanteil: jede Regel ein Punkt. Die Senkrechte ist die Kollapsgrenze der Abstimmung, die Waagerechte die DP-Referenz (volles Wissen)."""
    fig = go.Figure()
    for rule in ("quote", "aufwand", "escape", "schwelle", "opt"):
        sel = [r for r in rows if r["rule"] == rule]
        if not sel:
            continue
        fig.add_trace(go.Scatter(
            x=[100 * r["collapse"] for r in sel], y=[r["obj"] for r in sel], mode="markers",
            marker=dict(color=C.RULE_COLORS[rule], size=12, symbol=[MODE_SYMBOL[r["mode"]] for r in sel], line=dict(width=1, color="white")),
            name=SHORT[rule], customdata=[[MODE_TEXT[r["mode"]]] for r in sel],
            hovertemplate="%{customdata[0]}: Ziel %{y:.4f}, Kollaps %{x:.1f} %<extra></extra>",
        ))
    fig.add_vline(x=100 * cap, line=dict(color=C.COLOR_LINE, width=1.5, dash="dash"), annotation_text="Kollapsgrenze der Abstimmung", annotation_position="top right", annotation_font_size=11)
    if reference is not None:
        _hline(fig, reference, "DP-Referenz (volles Wissen)", color=C.RULE_COLORS["dp"])
    top = max([r["obj"] for r in rows] + ([reference] if reference else [0.0]))
    fig.update_xaxes(title_text="Kollaps: Anteil der Testpfade (%)", range=[-3, 103])
    fig.update_yaxes(title_text="Ziel: gleichwertiger Jahresfang (K je Jahr)", range=[0, 1.12 * top])
    return _base(fig, 400)


# --- Tab 3 ---------------------------------------------------------------------------------------------------------------------------------
def fig_kipp_ziel(series, reference):
    """Ziel über dem Messrauschen: je Regel eine Linie (Zeilen des Rasters); der Aufwand ist konstant, weil er den Survey nicht liest."""
    fig = go.Figure()
    top = reference or 0.0
    for key, rows in series.items():
        rule = rows[0]["rule"]
        xs = [r["sm"] for r in rows]
        ys = [r["obj"] for r in rows]
        top = max(top, max(ys))
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines+markers", line=dict(color=C.RULE_COLORS[rule], width=2.5), marker=dict(size=6), name=key,
                                 hovertemplate="σ_m = %{x}: Ziel %{y:.4f}<extra></extra>"))
    if reference is not None:
        _hline(fig, reference, "DP-Referenz (volles Wissen, exakter Survey)", color=C.RULE_COLORS["dp"])
    fig.update_xaxes(title_text="Messrauschen σ_m", range=[-0.02, 0.82])
    fig.update_yaxes(title_text="Ziel (K je Jahr)", range=[0, 1.12 * top])
    return _base(fig)


def fig_kipp_diff(series, sm_star=None, sm_star_label=""):
    """Differenz Regel minus fester Aufwand (gepaart) über dem Messrauschen; unter der Nulllinie liegt der feste Aufwand vorn."""
    fig = go.Figure()
    lo = hi = 0.0
    for key, rows in series.items():
        rule = rows[0]["rule"]
        xs = [r["sm"] for r in rows]
        ys = [r["d_vs_aufwand"] for r in rows]
        lo, hi = min(lo, min(ys)), max(hi, max(ys))
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines+markers", line=dict(color=C.RULE_COLORS[rule], width=2.5), marker=dict(size=6), name=key,
                                 hovertemplate="σ_m = %{x}: Differenz %{y:+.4f}<extra></extra>"))
    _hline(fig, 0.0, "gleichauf mit dem festen Aufwand", color=C.COLOR_GREY, dash="solid")
    if sm_star is not None:
        fig.add_vline(x=sm_star, line=dict(color=C.COLOR_LINE, width=1.5, dash="dot"), annotation_text=sm_star_label or f"Kippstelle {sm_star:.2f}",
                      annotation_position="top right", annotation_font_size=11)
    pad = 0.08 * max(hi - lo, 1e-3)
    fig.update_xaxes(title_text="Messrauschen σ_m", range=[-0.02, 0.82])
    fig.update_yaxes(title_text="Ziel der Regel minus Ziel des Aufwands", range=[lo - pad, hi + pad])
    return _base(fig)


def fig_kipp_collapse(series, cap):
    """Kollapsanteil über dem Messrauschen je Regel (Testpfade)."""
    fig = go.Figure()
    for key, rows in series.items():
        rule = rows[0]["rule"]
        fig.add_trace(go.Scatter(x=[r["sm"] for r in rows], y=[100 * r["collapse"] for r in rows], mode="lines+markers",
                                 line=dict(color=C.RULE_COLORS[rule], width=2.5), marker=dict(size=6), name=key, hovertemplate="σ_m = %{x}: Kollaps %{y:.1f} %<extra></extra>"))
    _hline(fig, 100 * cap, "Kollapsgrenze der Abstimmung", color=C.COLOR_LINE)
    fig.update_xaxes(title_text="Messrauschen σ_m", range=[-0.02, 0.82])
    fig.update_yaxes(title_text="Kollaps: Anteil der Testpfade (%)", range=[-3, 103])
    return _base(fig)


# --- Tab 4 ---------------------------------------------------------------------------------------------------------------------------------
def fig_dp_value(g, V, beta):
    """Gleichwertiger Jahresfang (1 - β) V(B) in Abhängigkeit vom Anfangsbestand B."""
    fig = go.Figure(go.Scatter(x=list(g), y=list((1.0 - beta) * V), mode="lines", line=dict(color=C.RULE_COLORS["opt"], width=3), name="(1 − β) V(B)",
                               hovertemplate="B = %{x:.2f}: %{y:.4f}<extra></extra>"))
    fig.update_xaxes(title_text="Bestand B (K)", range=[0, float(g[-1])])
    fig.update_yaxes(title_text="gleichwertiger Jahresfang (1 − β) V(B)", range=[0, float(1.1 * (1.0 - beta) * V[-1])])
    return _base(fig)


def fig_dp_policy(g, Sp, s_formula=None):
    """Politik der Wertiteration: Rest S(B) nach der Entnahme (unter der Schwelle wird nichts entnommen, darüber bleibt S*); die Formel als Linie, wenn sie gilt."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(g), y=list(g), mode="lines", line=dict(color=C.COLOR_GREY, width=1, dash="dot"), name="Rest = Bestand (keine Entnahme)", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=list(g), y=list(Sp), mode="lines", line=dict(color=C.RULE_COLORS["opt"], width=3), name="Rest S(B) der Wertiteration",
                             hovertemplate="B = %{x:.2f}: S = %{y:.4f}<extra></extra>"))
    if s_formula is not None:
        _hline(fig, s_formula, "Formel S* (nur ohne Wachstumsrauschen)", color=C.COLOR_LINE)
    fig.update_xaxes(title_text="Bestand B vor der Entnahme (K)", range=[0, float(g[-1])])
    fig.update_yaxes(title_text="Rest nach der Entnahme (K)", range=[0, 1.1])
    return _base(fig)


def nice_ymax(*arrays, floor=1.2, cap=3.0):
    """Obergrenze der Bestandsachse: das 99,5-%-Perzentil aller Werte mit etwas Luft, mindestens `floor`, höchstens `cap`."""
    top = max(float(np.percentile(a, 99.5)) for a in arrays)
    return float(min(cap, max(floor, 1.05 * top)))
