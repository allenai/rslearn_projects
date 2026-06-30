"""Collapse disk_usage.py JSONL into a bounded nested tree for the web app.

The scanner emits one line per directory (direct files only). This script does a
single streaming pass to:

- keep directories up to ``--max_depth`` as tree nodes, and
- fold everything deeper than ``--max_depth`` into its depth-``max_depth``
  ancestor (so memory is bounded by the number of dirs within max_depth levels),

then builds the nested tree, collapses any folder whose total size is below
``--collapse_gb``, and writes the result as a single JSON file that app.py serves
verbatim. Re-run with different thresholds without rescanning the filesystem.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

from tqdm import tqdm


def entry_bytes(e: dict) -> int:
    """Self bytes of an entry: whole-subtree total for recursive (full-scan)
    entries, direct files only otherwise."""
    return e["total_bytes"] if e.get("recursive") else e.get("direct_bytes", 0)


def entry_files(e: dict) -> int:
    return e["total_file_count"] if e.get("recursive") else e.get("direct_file_count", 0)


def load_nodes(input_path: str, max_depth: int) -> dict[str, dict]:
    """Stream the JSONL, keeping depth<=max_depth entries and folding the rest.

    Bytes/file-counts of directories deeper than ``max_depth`` are added to their
    depth-``max_depth`` ancestor, which becomes a leaf carrying the subtree total.
    """
    nodes: dict[str, dict] = {}
    deep_bytes: dict[str, int] = defaultdict(int)
    deep_files: dict[str, int] = defaultdict(int)
    # The scanner can write a directory more than once after a resume; keep only
    # the first occurrence so the additive deep-fold below never double-counts.
    seen: set[str] = set()

    with open(input_path) as f:
        for line in tqdm(f, desc="lines", unit="line"):
            line = line.strip()
            if not line:
                continue
            e = json.loads(line)
            if e["path"] in seen:
                continue
            seen.add(e["path"])
            depth = e["depth"]
            if depth <= max_depth:
                nodes[e["path"]] = e
            else:
                p = e["path"]
                for _ in range(depth - max_depth):
                    p = os.path.dirname(p)
                deep_bytes[p] += entry_bytes(e)
                deep_files[p] += entry_files(e)

    for p, extra in deep_bytes.items():
        node = nodes.get(p)
        if node is None:
            # Ancestor not seen (shouldn't happen since the scanner emits every
            # directory), but guard rather than KeyError.
            continue
        node["direct_bytes"] += extra
        node["direct_file_count"] += deep_files[p]
        node["absorbed"] = True

    return nodes


def build_tree(nodes: dict[str, dict]) -> dict:
    """Build the nested tree from the (already depth-bounded) node entries."""
    children: dict[str, list[str]] = defaultdict(list)
    roots: list[str] = []
    for e in nodes.values():
        parent = os.path.dirname(e["path"])
        if parent != e["path"] and parent in nodes:
            children[parent].append(e["path"])
        else:
            roots.append(e["path"])

    def make_node(p: str) -> dict:
        e = nodes[p]
        child_nodes = [make_node(cp) for cp in children.get(p, [])]
        child_nodes.sort(key=lambda n: n["size"], reverse=True)
        own = entry_bytes(e)
        return {
            "name": os.path.basename(p) or p,
            "path": p,
            "depth": e.get("depth", 0),
            "errors": len(e.get("errors", [])),
            "own_bytes": own,
            "file_count": entry_files(e),
            "children": child_nodes,
            "size": own + sum(c["size"] for c in child_nodes),
            # Truncated if its subtree is collapsed into this node: a full-scan
            # (recursive) entry, or a leaf that had subdirs folded at max_depth.
            "truncated": e.get("recursive", False)
            or (not child_nodes and e.get("num_subdirs", 0) > 0),
        }

    root_nodes = [make_node(r) for r in roots]
    root_nodes.sort(key=lambda n: n["size"], reverse=True)
    if len(root_nodes) == 1:
        return root_nodes[0]
    return {
        "name": "(multiple roots)",
        "path": "",
        "depth": -1,
        "errors": 0,
        "own_bytes": 0,
        "file_count": 0,
        "children": root_nodes,
        "size": sum(n["size"] for n in root_nodes),
        "truncated": False,
    }


def prune(node: dict, threshold: int, max_children: int) -> dict:
    """Drop children of folders under ``threshold`` or over ``max_children``.

    The folder's aggregate ``size`` is kept; only its subtree is removed, which
    is what shrinks the payload sent to the browser.
    """
    if node.get("children"):
        if node["size"] < threshold or len(node["children"]) > max_children:
            node["collapsed"] = True
            node["children"] = []
        else:
            node["collapsed"] = False
            for child in node["children"]:
                prune(child, threshold, max_children)
    return node


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", default="disk_usage.jsonl", help="JSONL produced by disk_usage.py."
    )
    parser.add_argument(
        "--output", default="collapsed.json", help="Output nested-tree JSON path."
    )
    parser.add_argument(
        "--max_depth",
        type=int,
        default=10,
        help="Directories deeper than this (relative to the scan root) are folded "
        "into their depth-max_depth ancestor. Bounds memory and tree size.",
    )
    parser.add_argument(
        "--collapse_gb",
        type=float,
        default=10.0,
        help="Folders whose total size is below this (GiB) are emitted without "
        "their children (aggregate size kept) for a lean payload.",
    )
    parser.add_argument(
        "--max_children",
        type=int,
        default=100,
        help="Folders with more than this many direct children are emitted without "
        "their children (aggregate size kept) for a lean payload.",
    )
    args = parser.parse_args()

    input_path = os.path.abspath(args.input)
    threshold = int(args.collapse_gb * 1024**3)
    print(f"loading {input_path} (fold below depth {args.max_depth})...")
    nodes = load_nodes(input_path, args.max_depth)
    print(f"kept {len(nodes)} nodes within depth {args.max_depth}")
    tree = prune(build_tree(nodes), threshold, args.max_children)
    response = {
        "input": input_path,
        "tree": tree,
        "collapse_gb": threshold / 1024**3,
        "max_depth": args.max_depth,
        "max_children": args.max_children,
    }
    with open(args.output, "w") as f:
        json.dump(response, f)
    print(f"wrote {args.output} (total {tree['size']} bytes)")


if __name__ == "__main__":
    main()
