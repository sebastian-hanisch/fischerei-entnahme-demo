"""PDF-Export (fpdf2, Kernschriften: Umlaute gehen, andere Sonderzeichen werden ersetzt)."""

from fpdf import FPDF

_REPLACE = {
    "–": "-", "—": "-", "≥": ">=", "≤": "<=", "→": "->", "×": "x", "·": "*", "„": '"', "“": '"', "’": "'", "≈": "~", "−": "-", "σ": "sigma", "β": "beta",
    "ξ": "xi", "η": "eta", "θ": "theta", "Δ": "Delta", "⁻": "-", "⁴": "4", "²": "2", "ρ": "rho", "µ": "mu",
}


def _latin(text):
    for a, b in _REPLACE.items():
        text = text.replace(a, b)
    text = text.replace("**", "")
    return text.encode("latin-1", "replace").decode("latin-1")


def build_pdf(settings, view, rows, verdicts):
    """settings: dict der Reglerwerte; view: dict Kennzahlen des Pfadbündels (Tab 1); rows: Zeilen des Vergleichs ({name, mode, par_text, obj, collapse}); verdicts: Liste von Texten der Kernfragen."""
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 15)
    pdf.cell(0, 9, _latin("Fischerei und Wild: Wie viel entnehmen?"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 9)
    pdf.multi_cell(0, 5, _latin("Modellrechnung mit erfundenen Zahlen (kein Bestand gemessen, keine Bewirtschaftungsempfehlung). Bestand in Einheiten der Tragfähigkeit K."),
                   new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 6, "Einstellungen", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 9)
    pdf.multi_cell(0, 5, _latin("; ".join(f"{k} = {v}" for k, v in settings.items())), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 6, _latin("Gewählte Regel auf dem Pfadbündel"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 9)
    pdf.multi_cell(0, 5, _latin("; ".join(f"{k}: {v}" for k, v in view.items())), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 6, _latin("Regeln im fairen Vergleich (Testpfade)"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "B", 8)
    for head, wd in (("Regel", 62), ("Abstimmung", 38), ("Regler", 32), ("Ziel", 20), ("Kollaps", 20)):
        pdf.cell(wd, 5, head, border=1)
    pdf.ln()
    pdf.set_font("Helvetica", "", 8)
    for r in rows:
        pdf.cell(62, 5, _latin(r["name"])[:40], border=1)
        pdf.cell(38, 5, _latin(r["mode"])[:24], border=1)
        pdf.cell(32, 5, _latin(r["par_text"])[:20], border=1)
        pdf.cell(20, 5, f"{r['obj']:.4f}", border=1)
        pdf.cell(20, 5, f"{100 * r['collapse']:.1f} %", border=1)
        pdf.ln()
    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 6, "Kernfragen", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 9)
    for v in verdicts:
        pdf.multi_cell(0, 5, _latin("- " + v), new_x="LMARGIN", new_y="NEXT")
    return bytes(pdf.output())
