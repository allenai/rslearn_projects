"""Part 2: LOIO summary, few-shot, arms, ablations, zero-shot, window size, trop-4."""
import json
P2 = "/weka/dfive-default/piperw/dev/rslearn_projects/pastis2"
D = json.load(open(f"{P2}/v3_data.json")); M = D["meta"]
ISL, MODS, XS = M["isl"], M["mods"], M["xs"]
HL = r"\cellcolor{colorone!25}"
out = []; w = out.append
def v(c): return c["v"] if isinstance(c, dict) and "v" in c else None
def t(c): return c.get("t") if isinstance(c, dict) else None
def row(o, k="model"): return o.get(k) or o.get("setting") or str(o.get("ws","?"))
def f(c, bold=False, hl=False):
    x = v(c)
    if x is None: return "X" if isinstance(c,dict) and c.get("s")=="run" else "--"
    s = f"${x:.2f}\\pm{c['sd']:.2f}$" if isinstance(c,dict) and c.get("sd") is not None else f"{x:.2f}"
    if bold: s = r"\textbf{"+s+"}"
    return (HL+s) if hl else s
def get(sec,name,mod=None):
    rows = D[sec][mod] if mod else D[sec]
    return next((r for r in rows if row(r)==name), None)

# ---------- LOIO summary (S2) ----------------------------------------------
w(r"""
\begin{table*}[t]
    \centering
    \small
    \setlength{\tabcolsep}{3pt}
    \begin{tabular}{ll|cccccc}
        Model & Variant & R\'eunion & Guadeloupe & Martinique & Guyane & Mayotte & mean\\
        \hline""")
S2 = D["loio"]["S2"]
order = [("OlmoEarth v1.3","64d","OlmoEarth probe · 64d"),
         ("OlmoEarth v1.3","128d","OlmoEarth probe · 128d"),
         ("TESSERAv2","precomputed","Tessera v2"),
         ("AEF / GSE","precomputed","AEF / GSE"),
         (None,None,None),
         ("UTAE","from scratch","U-TAE (scratch)"),
         ("TSViT","from scratch","TSViT (scratch)"),
         (None,None,None),
         ("OlmoEarth v1.2 FT","--","OlmoEarth v1.2 fine-tune"),
         (r"\olmoearth \pastis-FT","probe","OlmoEarth v1.2 PASTIS-FT → probe"),
         ("Trivial","--","Trivial (all Background)")]
# best per island among non-trivial, non-transfer rows
cand=[k for _,_,k in order if k and not k.startswith(("Trivial","OlmoEarth v1.2 PASTIS-FT"))]
best={i:max(((v(get("loio",k,"S2")["cells"][i]) or -1),k) for k in cand)[1] for i in range(5)}
for mdl,var,key in order:
    if key is None: w(r"        \hline"); continue
    r=get("loio",key,"S2"); cs=r["cells"]
    vals=[v(c) for c in cs if v(c) is not None]
    mean=f"{sum(vals)/len(vals):.2f}" if len(vals)==5 else "--"
    cells=[f(c, bold=(best[i]==key), hl=(best[i]==key and key.startswith("OlmoEarth v1.2 fine"))) for i,c in enumerate(cs)]
    w(f"        {mdl} & {var} & " + " & ".join(cells) + f" & {mean}\\\\")
w(r"""    \end{tabular}
    \caption{\planteur leave-one-island-out (LOIO), Sentinel-2 inputs. Metric is
    mIoU on the held-out island; mean is across the 5 folds. Best per column in
    bold. The \pastis-FT $\rightarrow$ probe row is the cross-region transfer
    setting: an encoder fine-tuned on \pastis and probed without adaptation.}
    \label{tab:planteur-loio}
\end{table*}""")

# ---------- few-shot (X=0 column dropped) ----------------------------------
w(r"""
\begin{table*}[t]
    \centering
    \small
    \setlength{\tabcolsep}{4pt}
    \begin{tabular}{l|cccc}
        Method & $X=10$ & $X=25$ & $X=100$ & $X=1000$\\
        \hline""")
FS={row(r):r for r in D["fewshot"]}
fs_order=[("OlmoEarth probe (d128\\_wideread ckpt)","OlmoEarth probe"),
          ("OlmoEarth v1.2 FT","OlmoEarth v1.2 FT"),
          ("AEF / GSE","AEF / GSE"),("TESSERAv2","Tessera v2"),
          (None,None),
          ("UTAE (from scratch)","U-TAE"),("TSViT (from scratch)","TSViT")]
bestc={i:max(((v(FS[k]["cells"][i]) or -1),k) for _,k in fs_order if k)[1] for i in range(4)}
for label,key in fs_order:
    if key is None: w(r"        \hline"); continue
    cs=FS[key]["cells"]
    w(f"        {label} & " + " & ".join(
        f(c,bold=(bestc[i]==key),hl=(bestc[i]==key)) for i,c in enumerate(cs)) + r"\\")
w(r"        \hline")
w(r"        Trivial (all Background) & 10.07 & 10.07 & 10.07 & 10.07\\")
w(r"""    \end{tabular}
    \caption{Few-shot transfer to \planteur: train on all of \pastis plus $X$
    labeled pixels per \planteur class, evaluate on the 443-window test split.
    Metric is mIoU (points, $\times 100$). The pure $X=0$ transfer point is
    reported separately in Table~\ref{tab:planteur-zeroshot}. NOTE: the frozen
    \olmoearth probe row here uses the earlier d128\_wideread checkpoint, not the
    d768\_proj128lin checkpoint used in Tables~\ref{tab:planteur-443}
    and~\ref{tab:planteur-loio}; the two are not directly comparable.}
    \label{tab:planteur-fewshot}
\end{table*}""")

# ---------- arms -----------------------------------------------------------
w(r"""
\begin{table*}[t]
    \centering
    \small
    \setlength{\tabcolsep}{4pt}
    \begin{tabular}{ll|cccc}
        Model & Arm & $X=10$ & $X=25$ & $X=100$ & $X=1000$\\
        \hline""")
NAME={"OlmoEarth probe":"OlmoEarth probe","OlmoEarth v1.2 FT":r"\olmoearth FT",
      "U-TAE":"UTAE","TSViT":"TSViT","AEF / GSE":r"\aef / GSE","Tessera v2":r"\tesseravtwo",
      "Trivial (all Background)":"Trivial"}
ARM={"unbalanced":r"all \pastis","balanced":"balanced","planteur-only":r"\planteur only","—":"--"}
prev=None
for r in D["arms"]:
    m=row(r)
    if prev is not None and m!=prev: w(r"        \hline")
    prev=m
    cs=r["cells"]
    w(f"        {NAME.get(m,m)} & {ARM.get(r.get('arm','--'),r.get('arm'))} & "
      + " & ".join(f(c) for c in cs) + r"\\")
w(r"""    \end{tabular}
    \caption{Class-balancing control for few-shot transfer: all of \pastis, a
    class-balanced \pastis+\planteur subset, and \planteur-only training. Metric is
    \planteur test mIoU (points, $\times 100$).}
    \label{tab:planteur-fewshot-class-balance}
\end{table*}""")
open(f"{P2}/results_tables_updated.tex","a").write("\n".join(out)+"\n")
print(f"  appended part 2 ({len(out)} lines)")
