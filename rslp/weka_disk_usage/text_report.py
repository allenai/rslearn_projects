"""Render the collapsed disk-usage tree (from collapse.py) as a plain-text report.

This is an alternative to the Flask viewer in app.py for when hosting a web app
is inconvenient: it prints an ncdu/``tree``-style listing with fixed-width size
and percent-of-total columns on the left and a box-drawing tree on the right.

The collapsed JSON can still hold many thousands of nodes, so a second size
threshold (``--min_gb``) is applied at render time: folders below it are not
listed individually but are folded into one trailing ``(N smaller folders)``
line per parent carrying their summed size, so the listed children plus the
remainder always add up to the parent. Re-render at a different cutoff without
touching the JSONL; it only takes milliseconds on the collapsed JSON.

Tags mirror the web app:

- ``subtree``: the node's whole subtree was accounted as one total by the
  scanner (full-scan or folded at ``--max_depth``), so it has no children.
- ``collapsed``: the node's children were pruned by collapse.py
  (``--collapse_gb`` / ``--max_children``), so they are absent from the JSON.
- ``N err``: the scanner hit N OSErrors in this directory.
"""

from __future__ import annotations

import argparse
import json
import sys

UNITS = ["B", "KB", "MB", "GB", "TB", "PB"]


def human_bytes(n: int | float) -> str:
    """Format bytes like the web app's humanBytes (1024-based, 1 decimal below 100)."""
    v = float(n)
    i = 0
    while v >= 1024 and i < len(UNITS) - 1:
        v /= 1024
        i += 1
    if i == 0 or v >= 100:
        return f"{v:.0f} {UNITS[i]}"
    return f"{v:.1f} {UNITS[i]}"


def node_tags(node: dict) -> list[str]:
    """Bracketed status tags for a node, matching the web app's tags."""
    tags = []
    if node.get("truncated"):
        tags.append("subtree")
    if node.get("collapsed"):
        tags.append("collapsed")
    errors = node.get("errors", 0)
    if errors:
        tags.append(f"{errors} err")
    return tags


def format_line(
    size: int, total: int, tree_part: str, tags: list[str] | None = None
) -> str:
    """One report line: fixed-width size and percent-of-total, then the tree text."""
    pct = 100.0 * size / total if total > 0 else 0.0
    line = f"{human_bytes(size):>9}  {pct:5.1f}%  {tree_part}"
    if tags:
        line += f"  [{', '.join(tags)}]"
    return line


def render_node(
    node: dict,
    total: int,
    min_bytes: int,
    out: list[str],
    prefix: str = "",
    connector: str = "",
    is_root: bool = True,
) -> None:
    """Append the lines for ``node`` and its (thresholded) subtree to ``out``.

    ``prefix`` is the vertical-guide string inherited from ancestors and
    ``connector`` is this node's own branch glyph (empty for the root).
    """
    # The root shows its full path; the synthetic multi-root node has path "".
    label = node["path"] if is_root and node.get("path") else node["name"]
    out.append(
        format_line(node["size"], total, prefix + connector + label, node_tags(node))
    )

    children = node.get("children") or []
    shown = [c for c in children if c["size"] >= min_bytes]
    hidden = [c for c in children if c["size"] < min_bytes]

    if is_root:
        child_prefix = ""
    else:
        child_prefix = prefix + ("    " if connector == "└── " else "│   ")

    for i, child in enumerate(shown):
        last = i == len(shown) - 1 and not hidden
        render_node(
            child,
            total,
            min_bytes,
            out,
            prefix=child_prefix,
            connector="└── " if last else "├── ",
            is_root=False,
        )
    if hidden:
        hidden_size = sum(c["size"] for c in hidden)
        out.append(
            format_line(
                hidden_size,
                total,
                f"{child_prefix}└── ({len(hidden)} smaller folders)",
            )
        )


def render_tree(response: dict, min_bytes: int) -> list[str]:
    """Render the collapse.py output dict into report lines (header + tree)."""
    tree = response["tree"]
    total = tree["size"]
    lines = [
        f"Disk usage report for {response.get('input', '?')}",
        f"Total: {human_bytes(total)} ({total} bytes)",
        f"Collapse params: collapse_gb={response.get('collapse_gb')}, "
        f"max_depth={response.get('max_depth')}, "
        f"max_children={response.get('max_children')}",
        f"Listing folders >= {min_bytes / 1024**3:g} GB; smaller siblings are "
        "summed into one '(N smaller folders)' line per parent.",
        "Tags: [subtree] = whole subtree counted as one total by the scanner; "
        "[collapsed] = children pruned by collapse.py; [N err] = scan errors.",
        "",
        f"{'size':>9}  {'%tot':>6}  path",
    ]
    render_node(tree, total, min_bytes, lines)
    return lines


def main() -> None:
    """CLI: read collapse.py JSON and write the text report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", default="collapsed.json", help="JSON produced by collapse.py."
    )
    parser.add_argument(
        "--output",
        default="-",
        help="Output text path, or '-' for stdout (default).",
    )
    parser.add_argument(
        "--min_gb",
        type=float,
        default=1000.0,
        help="Folders below this size (GiB) are not listed individually but summed "
        "into a '(N smaller folders)' line under their parent. Should be >= the "
        "--collapse_gb used in collapse.py, since smaller folders have no children "
        "in the JSON anyway.",
    )
    args = parser.parse_args()

    with open(args.input) as f:
        response = json.load(f)
    lines = render_tree(response, int(args.min_gb * 1024**3))
    text = "\n".join(lines) + "\n"
    if args.output == "-":
        sys.stdout.write(text)
    else:
        with open(args.output, "w") as f:
            f.write(text)
        print(f"wrote {len(lines)} lines to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
