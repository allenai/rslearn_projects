"use strict";

const INDENT_PX = 18;

function humanBytes(n) {
  const units = ["B", "KB", "MB", "GB", "TB", "PB"];
  let i = 0;
  let v = n;
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024;
    i++;
  }
  const digits = v >= 100 || i === 0 ? 0 : 1;
  return `${v.toFixed(digits)} ${units[i]}`;
}

function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text;
  return e;
}

// Renders a node and returns its DOM element. maxSibling scales the bar so
// siblings are comparable; parentSize gives the percent-of-parent figure.
function renderNode(node, depth, maxSibling, parentSize) {
  const wrap = el("div", "node");

  const row = el("div", "row");
  row.style.setProperty("--indent", `${depth * INDENT_PX}px`);

  const hasChildren = node.children && node.children.length > 0;
  const caret = el("span", "caret" + (hasChildren ? "" : " leaf"), hasChildren ? "▶" : "•");

  const size = el("span", "size", humanBytes(node.size));

  const barWrap = el("span", "bar-wrap");
  const bar = el("span", "bar" + (node.truncated ? " truncated" : ""));
  const frac = maxSibling > 0 ? node.size / maxSibling : 0;
  bar.style.width = `${Math.max(frac * 100, node.size > 0 ? 1 : 0)}%`;
  barWrap.appendChild(bar);

  const pct = el("span", "pct", parentSize > 0 ? `${((node.size / parentSize) * 100).toFixed(0)}%` : "");

  const name = el("span", "name", node.name);

  row.appendChild(caret);
  row.appendChild(size);
  row.appendChild(barWrap);
  row.appendChild(pct);
  row.appendChild(name);

  if (node.truncated) {
    row.appendChild(el("span", "tag", "subtree"));
  }
  if (node.collapsed) {
    row.appendChild(el("span", "tag collapsed", "collapsed"));
  }
  if (node.errors > 0) {
    row.appendChild(el("span", "tag err", `${node.errors} err`));
  }

  const detailBits = [];
  if (node.own_bytes > 0) detailBits.push(`${humanBytes(node.own_bytes)} direct`);
  if (node.file_count > 0) detailBits.push(`${node.file_count} files`);
  if (detailBits.length) row.appendChild(el("span", "detail", detailBits.join(" · ")));

  wrap.appendChild(row);

  if (hasChildren) {
    const childContainer = el("div", "children collapsed");
    const childMax = Math.max(...node.children.map((c) => c.size), 1);
    let built = false;
    const build = () => {
      for (const child of node.children) {
        childContainer.appendChild(renderNode(child, depth + 1, childMax, node.size));
      }
      built = true;
    };
    const toggle = () => {
      if (!built) build();
      const collapsed = childContainer.classList.toggle("collapsed");
      caret.textContent = collapsed ? "▶" : "▼";
    };
    caret.addEventListener("click", toggle);
    row.addEventListener("dblclick", toggle);
    wrap.appendChild(childContainer);
  }

  return wrap;
}

async function load() {
  const status = document.getElementById("status");
  const treeEl = document.getElementById("tree");
  status.textContent = "Loading...";
  treeEl.innerHTML = "";
  try {
    const resp = await fetch("/api/tree");
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.error || resp.statusText);
    document.getElementById("input-path").textContent = data.input;
    document.getElementById("total-size").textContent = `Total: ${humanBytes(data.tree.size)}`;
    const depthNote = data.max_depth !== undefined ? `, max depth ${data.max_depth}` : "";
    document.getElementById("collapse-note").textContent = `collapsed < ${data.collapse_gb} GB${depthNote}`;
    status.textContent = "";
    const rootEl = renderNode(data.tree, 0, data.tree.size, data.tree.size);
    treeEl.appendChild(rootEl);
    // Expand the root level automatically.
    const rootCaret = rootEl.querySelector(".caret");
    if (rootCaret && !rootCaret.classList.contains("leaf")) rootCaret.click();
  } catch (err) {
    status.textContent = `Error: ${err.message}`;
  }
}

document.getElementById("refresh").addEventListener("click", load);
load();
