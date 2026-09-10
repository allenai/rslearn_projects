"""Part 3: loss ablations, temporal, window size (443 + LOIO), zero-shot,
trop-4 companion, per-island stats. The last four are new tables."""
import json
P2="/weka/dfive-default/piperw/dev/rslearn_projects/pastis2"
D=json.load(open(f"{P2}/v3_data.json")); M=D["meta"]; ISL=M["isl"]; MODS=M["mods"]
out=[]; w=out.append
def v(c): return c["v"] if isinstance(c,dict) and "v" in c else None
def tr(c): return c.get("t") if isinstance(c,dict) else None
def row(o,k="model"): return o.get(k) or o.get("setting") or str(o.get("ws","?"))
def f(c,bold=False):
    x=v(c)
    if x is None: return "X" if isinstance(c,dict) and c.get("s")=="run" else "--"
    s=f"{x:.2f}"
    return r"\textbf{"+s+"}" if bold else s
def get(sec,name,mod=None):
    rows=D[sec][mod] if mod else D[sec]
    return next((r for r in rows if row(r)==name),None)

# ---- loss / class weighting ----
w(r"""
\begin{table*}[t]
    \centering
    \small
    \setlength{\tabcolsep}{4pt}
    \begin{tabular}{l|ccccc}
        Model & baseline & inv & invsqrt & focal $\gamma=2$ & invsqrt+focal\\
        \hline""")
NM={"U-TAE":"U-TAE","TSViT":"TSViT","OlmoEarth probe · 64d":"OlmoEarth v1.3 (64d)",
    "OlmoEarth probe · 128d":"OlmoEarth v1.3 (128d)","OlmoEarth v1.2 FT":r"\olmoearth v1.2 FT"}
for r in D["clsw"]:
    cs=r["cells"]; bi=max(range(len(cs)),key=lambda i: v(cs[i]) or -1)
    w(f"        {NM.get(row(r),row(r))} & " + " & ".join(
        f(c,bold=(i==bi and v(c) is not None)) for i,c in enumerate(cs)) + r"\\")
w(r"        \hline")
w(r"        Trivial (all Background) & 10.07 & -- & -- & -- & --\\")
w(r"""    \end{tabular}
    \caption{Class-weighting and focal-loss ablations on \planteur (443-window test
    split). Metric is mIoU-8 (points, $\times 100$); best per row in bold.
    Dashes are configurations not run for that model.}
    \label{tab:planteur-loss-ablations}
\end{table*}""")

# ---- temporal (both dims) ----
w(r"""
\begin{table*}[t]
    \centering
    \small
    \setlength{\tabcolsep}{3pt}
    \begin{tabular}{l|cccccccccccc}
        Probe & 1 & 2 & 3 & 4 & 5 & 6 & 7 & 8 & 9 & 10 & 11 & 12\\
        \hline""")
for dim,cells in D["temporal"].items():
    w(f"        OlmoEarth v1.3 ({dim}) & " + " & ".join(f(c) for c in cells) + r"\\")
w(r"""    \end{tabular}
    \caption{Temporal ablation for the \olmoearth v1.3 probe on \planteur:
    mIoU-8 (points, $\times 100$) as a function of the first $N$ monthly mosaics.}
    \label{tab:planteur-temporal-ablation}
\end{table*}""")

# ---- window size 443 (NEW; the Discussion placeholder) ----
w(r"""
\begin{table*}[t]
    \centering
    \small
    \setlength{\tabcolsep}{4pt}
    \begin{tabular}{l|cccc}
        Probe & $w=1$ & $w=4$ & $w=8$ & $w=16$\\
        \hline""")
for dim,cells in D["winsize"].items():
    w(f"        OlmoEarth v1.3 ({dim}) & " + " & ".join(f(c) for c in cells) + r"\\")
w(r"""    \end{tabular}
    \caption{Spatial-context ablation on the 443-window \planteur test split:
    \olmoearth v1.3 probe at window size $w$ ($w=1$ is purely per-pixel). Metric
    is mIoU-8 (points, $\times 100$).}
    \label{tab:planteur-window-size-ablation}
\end{table*}""")

# ---- window size LOIO (NEW) ----
w(r"""
\begin{table*}[t]
    \centering
    \small
    \setlength{\tabcolsep}{4pt}
    \begin{tabular}{ll|ccccc|c}
        Probe & $w$ & R\'eunion & Guadeloupe & Martinique & Guyane & Mayotte & mean\\
        \hline""")
for dim in ("64d","128d"):
    for r in D["loiowinsize"][dim]:
        cs=r["cells"]; vals=[v(c) for c in cs if v(c) is not None]
        mean=f"{sum(vals)/len(vals):.2f}" if len(vals)==5 else "--"
        w(f"        OlmoEarth v1.3 ({dim}) & {r['ws']} & " + " & ".join(f(c) for c in cs)
          + f" & {mean}\\\\")
    w(r"        \hline")
w(r"""    \end{tabular}
    \caption{Spatial-context ablation under leave-one-island-out transfer.
    Sentinel-2 inputs; metric is mIoU on the held-out island (points,
    $\times 100$).}
    \label{tab:planteur-loio-window-size}
\end{table*}""")

# ---- zero-shot (NEW) ----
w(r"""
\begin{table*}[t]
    \centering
    \small
    \setlength{\tabcolsep}{4pt}
    \begin{tabular}{l|ccc}
        Setting & mIoU-8 & trop-4 & accuracy\\
        \hline""")
for z in D["zeroshot"]:
    c,t=z["cell"],z["trop"]
    a=f"{c['a']:.2f}\\%" if isinstance(c,dict) and c.get("a") is not None else "--"
    w(f"        {z['setting']} & {f(c)} & {f(t)} & {a}\\\\")
w(r"""    \end{tabular}
    \caption{Zero-shot / cross-region transfer to \planteur (443-window test
    split). trop-4 is mIoU restricted to the four tropical classes (Sugarcane,
    Banana, Pineapple, Trop.\ tuber). Both fully supervised models trained on
    \pastis alone score exactly 0 on the tropical classes.}
    \label{tab:planteur-zeroshot}
\end{table*}""")

# ---- trop-4 companion (NEW) ----
w(r"""
\begin{table*}[t]
    \centering
    \small
    \setlength{\tabcolsep}{4pt}
    \begin{tabular}{l|c|ccccc}
        & 443 & \multicolumn{5}{c}{LOIO (S2, held-out island)}\\
        Model & S2 & R\'eunion & Guadeloupe & Martinique & Guyane & Mayotte\\
        \hline""")
def trs(c):
    x=tr(c)
    return f"{x:.2f}" if x is not None else "--"
t443={row(r):r for r in D["t443"]}
for label,key in (("OlmoEarth v1.3 (64d)","OlmoEarth probe · 64d"),
                  ("OlmoEarth v1.3 (128d)","OlmoEarth probe · 128d"),
                  ("OlmoEarth v1.2 FT","OlmoEarth v1.2 fine-tune"),
                  ("UTAE (from scratch)","U-TAE (scratch)")):
    a=t443.get(key)
    lo=get("loio",key,"S2")
    a_s=trs(a["cells"][0]) if a else "--"
    w(f"        {label} & {a_s} & " + " & ".join(trs(c) for c in lo["cells"]) + r"\\")
lo=get("loio","OlmoEarth v1.2 PASTIS-FT → probe","S2")
zt=D["zeroshot"][2]["trop"]
w(r"        \olmoearth \pastis-FT $\rightarrow$ probe & " + f(zt) + " & "
  + " & ".join(trs(c) for c in lo["cells"]) + r"\\")
w(r"        \hline")
w(r"        Trivial (all Background) & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & 0.00\\")
w(r"""    \end{tabular}
    \caption{Tropical-class performance (trop-4: mIoU over Sugarcane, Banana,
    Pineapple, Trop.\ tuber). mIoU-8 can look respectable while trop-4 is near
    zero, because Background dominates the label distribution; Mayotte is the
    clearest case.}
    \label{tab:planteur-trop4}
\end{table*}""")

# ---- per-island stats (NEW) ----
w(r"""
\begin{table*}[t]
    \centering
    \small
    \setlength{\tabcolsep}{4pt}
    \begin{tabular}{l|cccc}
        Island & test windows & test pixels & Background \% & Trivial mIoU\\
        \hline""")
for isl,s in D["perisl"].items():
    nm={"reunion":"R\\'eunion"}.get(isl,isl.title())
    w(f"        {nm} & {s['windows']} & {s['px']:,} & {s['background_pct']:.2f}\\% "
      f"& {s['trivial']:.2f}\\\\")
w(r"""    \end{tabular}
    \caption{Per-island composition of the 443-window test split. Mayotte's
    99.45\% Background share explains both its high trivial baseline and why
    small gains over that baseline there should not be read as learning.}
    \label{tab:planteur-perisland}
\end{table*}""")
open(f"{P2}/results_tables_updated.tex","a").write("\n".join(out)+"\n")
print(f"  appended part 3 ({len(out)} lines)")
