"""Emit the results tables as LaTeX straight from v3_data.json.

Generated, not transcribed: every number in the paper then has one source, and a
re-harvest regenerates rather than inviting hand edits.
"""
import json

P2 = "/weka/dfive-default/piperw/dev/rslearn_projects/pastis2"
D = json.load(open(f"{P2}/v3_data.json"))
M = D["meta"]
ISL, MODS, XS = M["isl"], M["mods"], M["xs"]
HL = r"\cellcolor{colorone!25}"
out = []
w = out.append


def v(c):
    return c["v"] if isinstance(c, dict) and "v" in c else None


def f(c, pts=True, bold=False, hl=False):
    x = v(c)
    if x is None:
        return "--" if not (isinstance(c, dict) and c.get("s") == "run") else "X"
    s = f"{x:.2f}" if pts else f"{x/100:.4f}"
    if isinstance(c, dict) and c.get("sd") is not None:
        s = f"${x:.2f}\\pm{c['sd']:.2f}$"
    if bold:
        s = r"\textbf{" + s + "}"
    return (HL + s) if hl else s


def row(o, key="model"):
    return o.get(key) or o.get("setting") or str(o.get("ws", "?"))


def best_idx(rows, i, skip=("Trivial",)):
    vals = [(v(r["cells"][i]), n) for n, r in enumerate(rows)
            if not row(r).startswith(skip) and v(r["cells"][i]) is not None]
    return max(vals)[1] if vals else None


def get(section, name, mod=None):
    rows = D[section][mod] if mod else D[section]
    for r in rows:
        if row(r) == name:
            return r
    return None


# ---------- 443 headline (fractions, as in the original table) --------------
t443 = {row(r): r for r in D["t443"]}
w(r"""\begin{table*}[t]
    \centering
    \small
    \setlength{\tabcolsep}{4pt}
    \begin{tabular}{l|c}
        Method & mIoU\\
        \hline""")
ZS = {z["setting"]: z for z in D["zeroshot"]}
for label, key in (("OlmoEarth v1.3 (64d)", "OlmoEarth probe · 64d"),
                   ("OlmoEarth v1.3 (128d)", "OlmoEarth probe · 128d"),
                   ("AEF / GSE", "AEF / GSE"),
                   ("TESSERAv2", "Tessera v2")):
    w(f"        {label} & {f(t443[key]['cells'][0], pts=False)}\\\\")
z = ZS["OlmoEarth v1.2 PASTIS-FT → PLANTEUR probe"]["cell"]
w(r"        \olmoearth \pastis-FT $\rightarrow$ \planteur probe & " + f(z, pts=False) + r"\\")
w(r"        \hline")
w("        UTAE (from scratch; invsqrt+focal) & "
  + f(t443["U-TAE + invsqrt·focal"]["cells"][0], pts=False) + r"\\")
w("        TSViT (from scratch; focal $\\gamma=2$) & "
  + f(t443["TSViT + focal γ=2"]["cells"][0], pts=False) + r"\\")
w(r"        \hline")
w("        OlmoEarth v1.2 FT & " + HL + r"\textbf{"
  + f(t443["OlmoEarth v1.2 fine-tune"]["cells"][2], pts=False) + r"}\\")
w(r"""    \end{tabular}
    \caption{\planteur (443-window test split). Metric is mIoU over 8 classes
    (higher is better); the \olmoearth v1.2 FT entry is its best input
    configuration (S1+S2). Best is highlighted.}
    \label{tab:planteur-443}
\end{table*}""")

# ---------- 443 input ablation ---------------------------------------------
w(r"""
\begin{table*}[t]
    \centering
    \small
    \setlength{\tabcolsep}{4pt}
    \begin{tabular}{l|ccccc}
        & S2 & S1 & S1+S2 & S2+LS & S1+S2+LS\\
        \hline""")
for label, key in (("OlmoEarth v1.3 (64d)", "OlmoEarth probe · 64d"),
                   ("OlmoEarth v1.3 (128d)", "OlmoEarth probe · 128d")):
    w(f"        {label} & " + " & ".join(f(c) for c in t443[key]["cells"]) + r"\\")
w(r"        \hline")
ft = t443["OlmoEarth v1.2 fine-tune"]["cells"]
bi = max(range(5), key=lambda i: v(ft[i]) or -1)
w("        OlmoEarth v1.2 FT & "
  + " & ".join(f(c, bold=(i == bi), hl=(i == bi)) for i, c in enumerate(ft)) + r"\\")
w(r"""    \end{tabular}
    \caption{\olmoearth input ablations on the 443-window \planteur test split.
    Metric is mIoU (points, $\times 100$). LS denotes Landsat auxiliary inputs.}
    \label{tab:planteur-443-olmoearth-inputs}
\end{table*}""")

# ---------- LOIO input ablation (model x modality) -------------------------
w(r"""
\begin{table*}[t]
    \centering
    \small
    \setlength{\tabcolsep}{4pt}
    \begin{tabular}{ll|ccccc|c}
        Model & Inputs & R\'eunion & Guadeloupe & Martinique & Guyane & Mayotte & mean\\
        \hline""")
for label, key in (("OlmoEarth v1.3 (64d)", "OlmoEarth probe · 64d"),
                   ("OlmoEarth v1.3 (128d)", "OlmoEarth probe · 128d"),
                   ("OlmoEarth v1.2 FT", "OlmoEarth v1.2 fine-tune")):
    for m in MODS:
        r = get("loio", key, m)
        cs = r["cells"]
        vals = [v(c) for c in cs if v(c) is not None]
        mean = f"{sum(vals)/len(vals):.2f}" if len(vals) == 5 else "--"
        w(f"        {label} & {m} & " + " & ".join(f(c) for c in cs)
          + f" & {mean}\\\\")
    w(r"        \hline")
w(r"""    \end{tabular}
    \caption{Leave-one-island-out (LOIO) input ablations for \olmoearth. Metric is
    mIoU on the held-out island (points, $\times 100$). LS denotes Landsat.}
    \label{tab:planteur-loio-olmoearth-inputs}
\end{table*}""")

open(f"{P2}/results_tables_updated.tex", "w").write("\n".join(out) + "\n")
print(f"  wrote results_tables_updated.tex ({len(out)} lines, part 1)")
