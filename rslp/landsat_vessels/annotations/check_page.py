"""Static checks on the annotation page.

There is no node or headless browser on this host, so the page cannot be executed in a
smoke test. These are the checks that catch the failure modes a static read misses: an
element id the JS looks up but the HTML never defines (a silent null dereference at
runtime), an unbalanced bracket in the script, a CSS variable used but never declared,
and duplicate ids. Run after editing static/index.html.

Usage:
    python -m rslp.landsat_vessels.annotations.check_page
"""

import re
import sys
from pathlib import Path

PAGE = Path(__file__).parent / "static" / "index.html"


# A '/' in one of these positions opens a regex literal rather than being division.
# Without this, a regex like /"/g reads as the start of a string and swallows real code.
REGEX_PRECEDERS = set("(,=:[!&|?{};+-*%~^<>") | {"\n"}


def strip_js_noise(source: str) -> str:
    """Blank out string literals, comments and regexes so bracket counting is honest."""
    out: list[str] = []
    i = 0
    n = len(source)

    def last_significant() -> str:
        for char in reversed(out):
            if not char.isspace():
                return char
        return "\n"

    while i < n:
        char = source[i]
        pair = source[i : i + 2]
        if (
            char == "/"
            and pair not in ("//", "/*")
            and last_significant() in REGEX_PRECEDERS
        ):
            # Consume the regex body, honouring escapes and character classes, then the
            # flags. The whole literal becomes a placeholder.
            i += 1
            in_class = False
            while i < n:
                if source[i] == "\\":
                    i += 2
                    continue
                if source[i] == "[":
                    in_class = True
                elif source[i] == "]":
                    in_class = False
                elif source[i] == "/" and not in_class:
                    i += 1
                    break
                elif source[i] == "\n":
                    break  # not a regex after all; bail out rather than run away
                i += 1
            while i < n and source[i].isalpha():
                i += 1
            out.append("0")
            continue
        if pair == "//":
            end = source.find("\n", i)
            i = n if end == -1 else end
            continue
        if pair == "/*":
            end = source.find("*/", i + 2)
            i = n if end == -1 else end + 2
            continue
        if char in "\"'`":
            quote = char
            i += 1
            while i < n:
                if source[i] == "\\":
                    i += 2
                    continue
                if source[i] == quote:
                    i += 1
                    break
                # Keep the contents of template literals' ${...} so bracket balance and
                # id lookups inside them are still checked.
                if quote == "`" and source[i : i + 2] == "${":
                    depth = 0
                    while i < n:
                        if source[i] == "{":
                            depth += 1
                        elif source[i] == "}":
                            depth -= 1
                            if depth == 0:
                                out.append("}")
                                i += 1
                                break
                        out.append(source[i])
                        i += 1
                    continue
                i += 1
            out.append('""')
            continue
        out.append(char)
        i += 1
    return "".join(out)


def main() -> int:
    """Check the annotation static page for required elements; return an exit code."""
    html = PAGE.read_text()
    failures: list[str] = []

    script_match = re.search(r"<script>(.*)</script>", html, re.DOTALL)
    if not script_match:
        print("FAIL: no <script> block found")
        return 1
    script = script_match.group(1)
    clean = strip_js_noise(script)

    # 1. bracket balance
    pairs = {"(": ")", "[": "]", "{": "}"}
    stack: list[tuple[str, int]] = []
    line = 1
    for char in clean:
        if char == "\n":
            line += 1
        elif char in pairs:
            stack.append((char, line))
        elif char in pairs.values():
            if not stack:
                failures.append(f"unmatched closing '{char}' on script line {line}")
                break
            opener, opened_at = stack.pop()
            if pairs[opener] != char:
                failures.append(
                    f"'{opener}' opened on script line {opened_at} closed by '{char}' on line {line}"
                )
                break
    if stack:
        failures.append(
            "unclosed brackets: "
            + ", ".join(f"'{c}' from script line {ln}" for c, ln in stack[:5])
        )

    # 2. ids: every one the JS resolves must exist exactly once in the markup
    defined = re.findall(r'\bid="([^"]+)"', html)
    duplicates = {i for i in defined if defined.count(i) > 1}
    if duplicates:
        failures.append(f"duplicate element ids: {sorted(duplicates)}")

    # These live inside string literals, so they come from the raw script: `clean` has
    # blanked every literal out and would make the check pass vacuously.
    looked_up = set(re.findall(r'\$\("([^"]+)"\)', script)) | set(
        re.findall(r'getElementById\("([^"]+)"\)', script)
    )
    missing = sorted(looked_up - set(defined))
    if missing:
        failures.append(f"JS looks up ids the HTML does not define: {missing}")

    # Ids referenced from inline handlers too (help overlay's close button).
    for handler_id in re.findall(r"getElementById\('([^']+)'\)", html):
        if handler_id not in defined:
            failures.append(f"inline handler references missing id: {handler_id}")

    # 3. CSS custom properties: everything var(--x) must be declared somewhere
    declared = set(re.findall(r"(--[a-z0-9-]+)\s*:", html))
    used = set(re.findall(r"var\((--[a-z0-9-]+)", html))
    used |= set(re.findall(r'cssv\("(--[a-z0-9-]+)"\)', script))
    used |= set(re.findall(r'color:\s*"(--[a-z0-9-]+)"', script))
    undeclared = sorted(used - declared)
    if undeclared:
        failures.append(f"CSS variables used but never declared: {undeclared}")

    # 4. the series colour keys the chart code reads must be real variables
    series_vars = re.findall(r'color:\s*"(--s-[a-z]+)"', script)
    if len(series_vars) != 5:
        failures.append(f"expected 5 series colour vars, found {len(series_vars)}")

    # 5. fetch paths must match the routes the app serves
    routes = {"/api/session", "/api/item/", "/api/label", "/assets/"}
    for path in re.findall(r'fetch\(\s*[`"](/[a-z/]+)', script):
        if not any(path.startswith(route.rstrip("/")) for route in routes):
            failures.append(f"fetch to unknown route: {path}")

    if failures:
        print(f"{len(failures)} problem(s) in {PAGE.name}:")
        for failure in failures:
            print(f"  FAIL {failure}")
        return 1

    print(
        f"{PAGE.name} ok: {len(defined)} ids, {len(looked_up)} looked up, "
        f"{len(declared)} CSS variables, brackets balanced"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
