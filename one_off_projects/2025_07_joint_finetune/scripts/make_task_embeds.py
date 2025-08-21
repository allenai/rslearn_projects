"""Build compact task embeddings from rslearn configs or a monolithic datasets YAML.

Mean similarity with anchor subtraction is 0.48 (including self-self similarities), 
which is much better than the 0.73 without anchor subtraction.

If you add the qwen embedding templating plus the anchor subtraction, the mean
similarity between embeddings is 0.36, but similar tasks are still high.

If you don't truncate, then you need to use a projection layer in the helios encoder
to get the dimensions down to 768. Since qwen supports MLR, just truncating is fine.

Current commands
================
V1: python make_task_embeds.py --anchor --instruct --truncate 256 --add_benchmarks
V2: python make_task_embeds.py --anchor --instruct --truncate 256 --from_yaml /weka/dfive-default/ryanp/rslearn_projects/one_off_projects/2025_07_joint_finetune/data/tasks.yaml
"""
import os
import hashlib
import argparse
import json
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple, Set, Optional

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from transformers import AutoTokenizer, AutoModel
from rslp.helios.model import deep_merge

RESET = "\x1b[0m"
KEEP_INPUT_FAMILIES = ("sentinel1", "sentinel2", "landsat", "srtm")
DROP_INPUTS = {"mask", "image", "targets", "label"}
DEFAULT_INSTRUCT_PROMPT = """
You are building a compact similarity embedding for Earth-observation tasks that will
be used to condition a satellite foundation model (Helios) during downstream finetuning.
Focus on semantic content over formatting.

Prioritize (high weight):
- Task TYPE and variant (classification / detection / segmentation / regression; multi-label vs single-label).
- Target schema (property_name, full class list; or numeric target definition/units).
- SENSOR MODALITIES actually used (sentinel1, sentinel2, landsat, srtm) and whether time series/temporal mosaics are used.
- DOMAIN keywords (vessels, cropland, wind turbines, solar farms, floods, LCZ, debris, etc.).
- GEOGRAPHY and TIMEFRAME if present (AOIs, countries, biome, years).

Down-weight (low weight):
- Boilerplate YAML structure and keys, execution/runtime flags, file paths, worker counts.
- Repetition of monthly slices/band indices (e.g., sentinel2_0..11); band visualization details (RGB).
- Generic phrasing or punctuation.
"""


# ---------------- File gathering ----------------

def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}

def gather_task_files(base_dir: str, task: str) -> List[str]:
    """
    Return sorted YAML files under {base_dir}/{task} excluding *soup.yaml.
    Merges in filename order for determinism.
    """
    d = Path(base_dir) / task
    if not d.is_dir():
        raise FileNotFoundError(f"Task directory not found: {d}")
    files = [str(p) for p in sorted(d.glob("*.yaml")) if not str(p).endswith("soup.yaml")]
    if not files:
        raise FileNotFoundError(f"No YAML configs (excluding *soup.yaml) in {d}")
    return files


# ---------------- Compact extraction & summary ----------------

def _sorted_items(d: Dict[str, Any]) -> List[Tuple[str, Any]]:
    return sorted((d or {}).items(), key=lambda kv: kv[0])

def _family_from_key(k: str) -> Optional[str]:
    k = k.lower()
    if k in DROP_INPUTS:
        return None
    # map many spellings to canonical families
    if k.startswith(("s1", "sentinel1")):
        return "sentinel1"
    if k.startswith(("s2", "sentinel2")):
        return "sentinel2"
    if k.startswith("landsat"):
        return "landsat"
    if k.startswith("srtm"):
        return "srtm"
    return None  # ignore everything else

def _short_task_type(class_path: Optional[str]) -> Optional[str]:
    if not class_path:
        return None
    tail = class_path.split(".")[-1]  # e.g., ClassificationTask
    if tail.endswith("Task"):
        tail = tail[:-4]
    return tail.lower()  # classification, detection, segmentation, regression, etc.

def extract_task_core(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Keep only: collapsed input families, and task definition (type, property_name, classes).
    """
    data = (cfg.get("data") or {})
    init = (data.get("init_args") or {})
    keep: Dict[str, Any] = {}

    # Collapse inputs to canonical families
    inputs = (init.get("inputs") or {})
    fams: Set[str] = set()
    for key in inputs.keys():
        fam = _family_from_key(str(key))
        if fam and any(fam == f for f in KEEP_INPUT_FAMILIES):
            fams.add(fam)
    keep["inputs_families"] = sorted(fams)

    # Task(s)
    task = (init.get("task") or {})
    class_path = task.get("class_path", "")
    tinit = (task.get("init_args") or {})

    def _salient(sub_def: Dict[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        tp = _short_task_type(sub_def.get("class_path"))
        if tp:
            out["type"] = tp
        ia = sub_def.get("init_args") or {}
        if "property_name" in ia:
            out["property_name"] = ia["property_name"]
        if isinstance(ia.get("classes"), list):
            out["classes"] = ia["classes"]
        return out

    if "MultiTask" in class_path or class_path.endswith("MultiTask"):
        tasks = (tinit.get("tasks") or {})
        keep["tasks"] = {name: _salient(defn or {}) for name, defn in _sorted_items(tasks)}
        # input_mapping often repeats {'targets': ...}; skip unless special cases needed
    else:
        single = {"class_path": class_path, "init_args": tinit}
        keep["task"] = _salient(single)

    return keep

def build_task_summary(task_name: str, cfg: Dict[str, Any], skip_extract: bool = False) -> str:
    """
    Compact representation:
      task=<dir>
      inputs=<comma-separated families>   # subset of sentinel1,sentinel2,landsat,srtm
      For multitask: one block per subtask with short type & classes.
      For single task: type/classes inline.
    """
    if not skip_extract:
        core = extract_task_core(cfg)
    else:
        core = cfg
    lines: List[str] = [f"task={task_name}"]

    fams = core.get("inputs_families") or []
    if fams:
        lines.append("inputs=" + ",".join(fams))

    if "tasks" in core:
        for sub_name in sorted(core["tasks"].keys()):
            sub = core["tasks"][sub_name] or {}
            lines.append(f"subtask:{sub_name}")
            if sub.get("type"): 
                lines.append(f"  type={sub['type']}")
            if sub.get("property_name"): 
                lines.append(f"  property_name={sub['property_name']}")
            if sub.get("classes"):
                lines.append("  classes=" + ",".join(map(str, sub["classes"])))
    else:
        st = core.get("task", {})
        if st.get("type"):
            lines.append(f"type={st['type']}")
        if st.get("property_name"):
            lines.append(f"property_name={st['property_name']}")
        if st.get("classes"):
            lines.append("classes=" + ",".join(map(str, st["classes"])))

    return "\n".join(lines)

def extract_decoder_primary_key(cfg: Dict[str, Any]) -> Optional[str]:
    """
    Scrape model.init_args.model.init_args.decoders and return the FIRST key.
    """
    try:
        model = (cfg.get("model") or {})
        m_i = (model.get("init_args") or {})
        inner = (m_i.get("model") or {})
        inner_i = (inner.get("init_args") or {})
        decs = (inner_i.get("decoders") or {})
        for k in decs.keys():
            return k
        return None
    except Exception:
        return None


# ---------------- Color helpers ----------------

def cell_color(v: float, vmin: float=-1, vmax: float=1) -> str:
    # normalize v to [0,1]
    t = 0.0 if vmax <= vmin else (v - vmin) / (vmax - vmin)
    t = max(0, min(1, t))

    # interpolate RGB: 0=red(255,0,0), 0.5=yellow(255,255,0), 1=green(0,255,0)
    if t < 0.5:  # red → yellow
        r, g, b = 255, int(510*t), 0
    else:        # yellow → green
        r, g, b = int(510*(1-t)), 255, 0

    # quantize to xterm-256 cube
    steps = [0,95,135,175,215,255]
    q = lambda x: min(range(6), key=lambda i: abs(steps[i]-x))
    idx = 16 + 36*q(r) + 6*q(g) + q(b)

    # text color for contrast
    lum = 0.2126*r + 0.7152*g + 0.0722*b
    fg = 30 if lum > 140 else 97

    return f"\x1b[{fg};48;5;{idx}m"

def clip(s: str, w: int) -> str:
    return s if len(s) <= w else s[: max(1, w - 1)] + "…"


# ---------------- Embedding helpers ----------------

def get_detailed_instruct(task_description: str, query: str) -> str:
    """Instruction template for qwen embedding models."""
    return f"Instruct: {task_description}\nQuery:{query}"

def load_embedder(model_name: str, device: Optional[str]):
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    mdl = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    mdl.eval()
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    mdl.to(dev)
    return tok, mdl, dev

@torch.inference_mode()
def embed_texts_with(
    tokenizer, model, device: str, 
    texts: List[str], truncate: int | None = None, max_len: int = 8192
) -> torch.Tensor:
    enc = tokenizer(texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}
    out = model(**enc)

    hidden = out.last_hidden_state  # [B, T, H]
    mask = enc["attention_mask"].unsqueeze(-1)  # [B, T, 1]
    if truncate is not None:
        hidden = hidden[..., :truncate]

    emb = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
    emb = F.normalize(emb, p=2, dim=-1)

    return emb.detach().cpu()  # [N, D]


# ---------------- Anchor summary ----------------

def build_anchor_summary() -> str:
    """
    Build a boilerplate placeholder config and summarize it using the same code path.
    This captures template structure so its embedding can be subtracted.
    """
    dummy_cfg = {
        "data": {
            "init_args": {
                "inputs": {
                    "sentinel1": {},
                    "sentinel2": {},
                    "landsat": {},
                    "srtm": {},
                },
                "task": {
                    "class_path": "rslearn.train.tasks.task.TaskTask",
                    "init_args": {
                        "property_name": "category",
                        "classes": ["A", "B"]
                    }
                }
            }
        }
    }
    return build_task_summary("placeholder_task", dummy_cfg)


# ---------------- CLI ----------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build compact task embeddings from rslearn configs.")
    p.add_argument(
        "--base-dir",
        help="Base dir containing per-task subdirs.",
        default="/weka/dfive-default/ryanp/rslearn_projects/one_off_projects/2025_07_joint_finetune/configs"
    )
    p.add_argument(
        "--out-path",
        default="/weka/dfive-default/ryanp/rslearn_projects/one_off_projects/2025_07_joint_finetune/data/task_embeds_{info}.pt",
        help="Path to save torch dict {task_key: tensor[D]}."
    )
    p.add_argument(
        "--dump-texts",
        default="/weka/dfive-default/ryanp/rslearn_projects/one_off_projects/2025_07_joint_finetune/data/task_texts_{info}.jsonl",
        help="Optional JSONL for {task_dir, task_key, text}."
    )
    p.add_argument("--tasks", nargs="*", help="Task dir names (none = auto: subdirs starting with v2_*).")
    p.add_argument("--model", default="Qwen/Qwen3-Embedding-8B", help="HF embedding model to use.")
    p.add_argument("--device", default=None, help="Device override, e.g. 'cuda' or 'cpu'.")
    p.add_argument("--anchor", action="store_true", help="Enable anchor subtraction.")
    p.add_argument("--instruct", action="store_true", help="Use instruct template for embedding models.")
    p.add_argument(
        "--prompt",
        default=DEFAULT_INSTRUCT_PROMPT,
        help="Prompt to add to the task description if instruct is enabled."
    )
    p.add_argument("--truncate", type=int, default=None, help="Truncate task embeddings to this dimension.")
    p.add_argument("--add_benchmarks", action="store_true", help="Add pretrain evals to tasks.")
    p.add_argument("--from_yaml", type=str, default=None,
                   help="If set, load tasks directly from a single YAML of datasets (bypass rslearn parsing).")
    p.add_argument("--combine_descriptions", action="store_true",
                   help="If set with --from-yaml, append each dataset's description to the summary text.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    keys: List[str] = []
    summaries: List[str] = []
    anchor_text = None

    if args.from_yaml:
        # Load from monolithic datasets YAML (preferred)
        data = load_yaml(args.from_yaml)
        tasks: List[str] = []
        for ds_name, ds in data.items():
            summary = yaml.dump(ds)
            if ds_name == "anchor_dataset":
                anchor_text = summary
            else:
                tasks.append(ds_name)
                keys.append(ds_name)
                summaries.append(summary)

    else:
        # Load from rslearn configs
        tasks = args.tasks
        if not tasks:
            tasks = [name for name in os.listdir(args.base_dir) if name.startswith("v2_")]
            # special case - treat vessel classification + detection separately
            landsat_index = tasks.index("v2_landsat_vessels")
            tasks.insert(landsat_index, "v2_landsat_vessels")
        else:
            landsat_index = tasks.index("v2_landsat_vessels")

        for i, task_dir in enumerate(tasks):
            merged: Dict[str, Any] = {}
            is_landsat_vessels = (task_dir == "v2_landsat_vessels")

            if is_landsat_vessels:
                task = "detector" if i == landsat_index else "classifier"
                path = "finetune_{}_cosinelr.yaml".format(task)
                gathered = [os.path.join(args.base_dir, task_dir, path)]
            else:
                gathered = gather_task_files(args.base_dir, task_dir)

            for p in gathered:
                merged = deep_merge(merged, load_yaml(p))
            summaries.append(build_task_summary(task_dir, merged))

            key = extract_decoder_primary_key(merged)
            if not key:
                print(f"WARNING: Could not find decoders key for {task_dir}; using directory name.")
                key = task_dir
            keys.append(key)

    # Sort by key alphabetically
    keys_index = list(range(len(keys)))
    keys_index = list(sorted(keys_index, key=lambda i: keys[i]))

    tasks = [tasks[i] for i in keys_index]
    keys = [keys[i] for i in keys_index]
    summaries = [summaries[i] for i in keys_index]

    # Add pretrain eval benchmarks - don't use this with from_yaml
    # Add after the other tasks so that we can add new tasks without disrupting order of existing ones
    # In general, we should always add new tasks to the end of the list
    if args.add_benchmarks:
        assert not args.from_yaml, "Benchmarks are not supported when loading from YAML"
        benchmarks = load_yaml(os.path.join(args.base_dir, "../data/benchmark_info.yaml"))
        benchmarks = {
            k: benchmarks[k] for k in 
            sorted(benchmarks, key=lambda k: list(benchmarks[k]["tasks"].keys())[0])
        }
        for task_name, task_info in benchmarks.items():
            assert len(task_info["tasks"]) == 1, "Benchmark info should have only one task"
            summaries.append(build_task_summary(task_name, task_info, skip_extract=True))
            keys.append(list(task_info["tasks"].keys())[0])
            tasks.append(task_name)
    assert len(keys) == len(set(keys)), "Keys must be unique"

    # Get info string
    info = f"__{args.model.split('/')[-1]}__{args.truncate}d"
    if args.anchor:
        info += "__anchor"
    if args.instruct:
        info += "__instruct"
    if args.from_yaml:
        info += f"__from_yaml"

    print("=== Configuration ===")
    print(f"Info: {info}")
    print(f"Model: {args.model}")
    if args.from_yaml:
        print(f"Datasets YAML: {args.from_yaml}")
        print("Sorted subtask key list:")
        for key in keys:
            print(f" - {key}")
    else:
        print(f"Base dir: {args.base_dir}")
        print("Sorted task dir -> key:")
        for task_dir, key in zip(tasks, keys):
            print(f" - {task_dir:<30} -> {key}")
    print()

    # Load model once
    tokenizer, model, device = load_embedder(args.model, args.device)
    if args.instruct:
        print(f"\n[Instruct] Using prompt: {args.prompt}")

    # Build anchor embedding from placeholder summary
    if args.anchor:
        if not args.from_yaml:
            anchor_text = build_anchor_summary()
        
        fmt_anchor_text = anchor_text
        if args.instruct:
            fmt_anchor_text = get_detailed_instruct(fmt_anchor_text, args.prompt)

        print("\n[Anchor] Using placeholder summary for anchor subtraction:\n" + anchor_text + "\n")
        anchor_emb = embed_texts_with(
            tokenizer, model, device, [fmt_anchor_text], args.truncate
        )[0]  # [D]

    # Embed summaries
    if args.instruct:
        fmt_summaries = [get_detailed_instruct(summary, args.prompt) for summary in summaries]
    else:
        fmt_summaries = summaries
    embs = embed_texts_with(tokenizer, model, device, fmt_summaries, args.truncate)

    # Subtract anchor (if enabled), then renormalize
    if anchor_emb is not None:
        embs = F.normalize(embs - anchor_emb.unsqueeze(0), p=2, dim=-1)

    # Save dict {task_key: tensor[D]}
    result: Dict[str, torch.Tensor] = {k: e.detach().cpu() for k, e in zip(keys, embs)}

    # Add hash of current code for future checking
    result["code_hash"] = hashlib.sha256(open(__file__, "rb").read()).hexdigest()
    torch.save(result, args.out_path.format(info=info))

    # Optional: dump texts
    if args.dump_texts:
        with open(args.dump_texts.format(info=info), "w") as f:
            for task_dir, key, text in zip(tasks, keys, summaries):
                print(key, task_dir)
                print(text)
                print()
                print()
                f.write(json.dumps({"task_dir": task_dir, "task_key": key, "text": text}) + "\n")
            if args.instruct:
                f.write(json.dumps({"prompt": args.prompt}) + "\n")

    # Cosine similarity matrix
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    print(f"=== Cosine similarity matrix ===")

    mat = torch.stack([result[k] for k in keys], dim=0)  # [N, D]
    sim = F.cosine_similarity(mat.unsqueeze(1), mat.unsqueeze(0), dim=-1)  # [N, N]

    A = sim.detach().cpu().numpy()
    N = len(keys)
    prec = 3

    # Label and column widths
    clip = lambda s, w: s if len(s) <= w else s[: w - 1] + "…"
    idx_digits = max(2, len(str(max(0, N - 1))))
    num_w = max(prec + 2, idx_digits)
    label_w = max(16, min(36, max(len(k) for k in keys) + 1 + 2 + idx_digits))

    vmin, vmax = float(np.nanmin(A)), float(np.nanmax(A))

    row_means = A.mean(axis=1)
    col_means = A.mean(axis=0)
    overall_mean = A.mean()

    # Header: indices only
    header = " ".join(f"{i:>{num_w}d}" for i in range(N))
    header += f" {'mean':>{num_w}}"
    print(f"{'':>{label_w}} {header}")

    # Rows
    for i, name in enumerate(keys):
        row_label = f"{name} ({i})"
        row_label = clip(row_label, label_w)
        cells = []
        for j in range(N):
            v = float(A[i, j])
            prec_l = prec if v > 0 else prec - 1
            fmt = f"{v:>{num_w}.{prec_l}f}"
            cells.append(cell_color(v, vmin, vmax) + fmt + RESET)
        # row mean
        v = float(row_means[i])
        fmt = f"{v:>{num_w}.{prec}f}"
        cells.append(cell_color(v, vmin, vmax) + fmt + RESET)
        print(f"{row_label:>{label_w}} " + " ".join(cells))

    # final row: column means + overall mean
    row_label = clip("mean", label_w)
    cells = []
    for j in range(N):
        v = float(col_means[j])
        fmt = f"{v:>{num_w}.{prec}f}"
        cells.append(cell_color(v, vmin, vmax) + fmt + RESET)
    fmt = f"{overall_mean:>{num_w}.{prec}f}"
    cells.append(cell_color(overall_mean, vmin, vmax) + fmt + RESET)
    print(f"{row_label:>{label_w}} " + " ".join(cells))

    # Summary
    print(f"\nRange: [{vmin:.{prec}f}, {vmax:.{prec}f}]  Mean similarity: {A.mean():.{prec}f}")

    print(f"\nSaved dict with {len(result)} entries to {args.out_path.format(info=info)}")
    if args.dump_texts:
        print(f"Wrote plaintext rows to {args.dump_texts.format(info=info)}")


if __name__ == "__main__":
    main()
