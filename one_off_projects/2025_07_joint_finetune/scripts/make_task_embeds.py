"""Build compact task embeddings from rslearn configs.

Mean similarity with anchor subtraction is 0.48 (including self-self similarities), 
which is much better than the 0.73 without anchor subtraction.

If you add the qwen embedding templating plus the anchor subtraction, the mean
similarity between embeddings is 0.36, but similar tasks are still high.

If you don't truncate, then you need to use a projection layer in the helios encoder
to get the dimensions down to 768. With MLR, qwen's embeddings are pretty even so.

Current commands
================
python make_task_embeds.py --anchor --instruct --truncate 768
"""
import os
import hashlib
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple, Set, Optional

import torch
import torch.nn.functional as F
import yaml
from transformers import AutoTokenizer, AutoModel
from rslp.helios.model import deep_merge

KEEP_INPUT_FAMILIES = ("sentinel1", "sentinel2", "landsat", "srtm")
DROP_INPUTS = {"mask", "image", "targets", "label"}
DEFAULT_INSTRUCT_PROMPT = """
You are creating a similarity embedding for remote-sensing tasks.
Focus on the *content*, not formatting. Down-weight boilerplate and shared structure.

Emphasize:
- Task TYPE (classification/detection/segmentation/regression and any special variants)
- Target schema (property_name, classes/labels, numeric target meaning)
- Sensing MODALITIES actually used (sentinel1, sentinel2, landsat)
- Domain/topic keywords (e.g., vessels, cropland, wind turbines, solar farms)
- Geographic or dataset hints (if present)

De-emphasize:
- Generic YAML keys, ordering, punctuation
- Training/runtime settings (batch size, paths, workers, dtypes, masks)
- Repeated time slices or band indices (e.g., sentinel2_0..11) 
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

def build_task_summary(task_name: str, cfg: Dict[str, Any]) -> str:
    """
    Compact representation:
      task=<dir>
      inputs=<comma-separated families>   # subset of sentinel1,sentinel2,landsat,srtm
      For multitask: one block per subtask with short type & classes.
      For single task: type/classes inline.
    """
    core = extract_task_core(cfg)
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
        default="/weka/dfive-default/ryanp/rslearn_projects/one_off_projects/2025_07_joint_finetune/data/2025_08_12_task_embeds.pt",
        help="Path to save torch dict {task_key: tensor[D]}."
    )
    p.add_argument(
        "--dump-texts",
        default="/weka/dfive-default/ryanp/rslearn_projects/one_off_projects/2025_07_joint_finetune/data/2025_08_12_task_texts.jsonl",
        help="Optional JSONL for {task_dir, task_key, text}."
    )
    p.add_argument("--tasks", nargs="*", help="Task dir names (none = auto: subdirs starting with v2_*)")
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
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.tasks:
        args.tasks = [name for name in os.listdir(args.base_dir) if name.startswith("v2_")]

    summaries: List[str] = []
    keys: List[str] = []
    merged_cfgs: List[Dict[str, Any]] = []

    for task_dir in args.tasks:
        merged: Dict[str, Any] = {}
        for p in gather_task_files(args.base_dir, task_dir):
            merged = deep_merge(merged, load_yaml(p))
        merged_cfgs.append(merged)

        summaries.append(build_task_summary(task_dir, merged))

        key = extract_decoder_primary_key(merged)
        if not key:
            print(f"WARNING: Could not find decoders key for {task_dir}; using directory name.")
            key = task_dir
        keys.append(key)

    print("=== Configuration ===")
    print(f"Model: {args.model}")
    print(f"Base dir: {args.base_dir}")
    print("Task dirs:")
    for task_dir in args.tasks:
        print(f" - {task_dir}")
    print()

    # Load model once
    tokenizer, model, device = load_embedder(args.model, args.device)
    if args.instruct:
        print(f"\n[Instruct] Using prompt: {args.prompt}")

    # Build anchor embedding from placeholder summary (same summarizer path)
    anchor_emb = None
    if args.anchor:
        anchor_text = build_anchor_summary()
        fmt_anchor_text = anchor_text
        if args.instruct:
            fmt_anchor_text = get_detailed_instruct(anchor_text, args.prompt)
        anchor_emb = embed_texts_with(
            tokenizer, model, device, [fmt_anchor_text], args.truncate
        )[0]  # [D]
        print("\n[Anchor] Using placeholder summary for anchor subtraction:\n" + anchor_text + "\n")

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
    torch.save(result, args.out_path)

    # Optional: dump texts
    if args.dump_texts:
        with open(args.dump_texts, "w") as f:
            for task_dir, key, text in zip(args.tasks, keys, summaries):
                f.write(json.dumps({"task_dir": task_dir, "task_key": key, "text": text}) + "\n")
            f.write(json.dumps({"code_hash": result["code_hash"]}) + "\n")
            if args.anchor:
                f.write(json.dumps({"anchor_text": anchor_text}) + "\n")
            if args.instruct:
                f.write(json.dumps({"prompt": args.prompt}) + "\n")

    # Pretty print keys
    result.pop("code_hash")
    sorted_keys = sorted(result.keys())
    print("\n=== Saved task keys ===")
    for k in sorted_keys:
        print(" -", k)

    # Cosine similarity matrix
    mat = torch.stack([result[k] for k in sorted_keys], dim=0)  # [N, D]
    sim = F.cosine_similarity(mat.unsqueeze(1), mat.unsqueeze(0), dim=-1)  # [N, N]

    print("\n=== Cosine similarity matrix ===")
    header = ["{:>15}".format(k[:13]) for k in sorted_keys]
    print("{:>15} {}".format("", " ".join(header)))
    for i, k in enumerate(sorted_keys):
        row_vals = " ".join(f"{v:>13.3f}" for v in sim[i])
        print("{:>15} {}".format(k[:13], row_vals))
    print("\nMean similarity:", round(sim.mean().item(), 3))

    print(f"\nSaved dict with {len(result)} entries to {args.out_path}")
    if args.dump_texts:
        print(f"Wrote plaintext rows to {args.dump_texts}")


if __name__ == "__main__":
    main()
