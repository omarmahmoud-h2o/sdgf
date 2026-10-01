"""Build docs/sdgf-guide.html, the single-file interactive guide.

Sources, all read at build time so the guide can't drift from them:

    docs/overview.md, docs/how-it-works.md, docs/new-use-case.md   the three tabs
    CONTEXT.md                                                     glossary hover + panel
    docs/diagrams/<name>.html                                      embedded Archify diagrams
    docs/guide_assets/guide.css, guide.js                          inlined styles and script

The six-checks table in overview.md gives the layer explorer's cards; the L1-L6 sections
of how-it-works.md give each card's detail, with its FAG | CFA table turned into a toggle.
Diagrams are embedded as-is: rebuild them with Archify first if their JSON changed.

    pip install -e '.[docs]'
    python docs/build_guide.py [--out docs/sdgf-guide.html]
"""

from __future__ import annotations

import argparse
import html
import json
import re
from dataclasses import dataclass
from pathlib import Path

import markdown

DOCS = Path(__file__).resolve().parent
ROOT = DOCS.parent
ASSETS = DOCS / "guide_assets"

TABS = (
    ("overview", "Overview", "overview.md"),
    ("how-it-works", "How it works", "how-it-works.md"),
    ("new-use-case", "New use case", "new-use-case.md"),
)
DOC_TAB = {file: slug for slug, _, file in TABS}

# ![alt](diagrams/x.svg) followed by [Interactive version](diagrams/x.html)
DIAGRAM_RE = re.compile(
    r"!\[([^\]]*)\]\(diagrams/([\w-]+)\.svg\)\s*\n\s*\[Interactive version\]\(diagrams/\2\.html\)"
)
LAYERS_START = "### L1 "
LAYERS_END = "### Sent back and dropped"
EXPLORER_MARK = '<div class="layer-explorer-slot"></div>'


class GuideError(ValueError):
    pass


def md_to_html(text: str) -> str:
    return markdown.markdown(text, extensions=["tables", "fenced_code", "toc"])


def inline_md(text: str) -> str:
    out = md_to_html(text).strip()
    return out[3:-4] if out.startswith("<p>") and out.endswith("</p>") else out


def slug_text(text: str) -> str:
    return re.sub(r"[^\w]+", "-", text.lower()).strip("-")


# ------------------------------------------------------------------ tables


def table_rows(block: str) -> list[list[str]]:
    """Cells of a markdown table, header first, separator row dropped."""
    rows = []
    for line in block.strip().splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if all(re.fullmatch(r":?-+:?", c) for c in cells if c):
            continue
        rows.append(cells)
    return rows


def find_tables(text: str) -> list[tuple[int, int, str]]:
    """(start, end, block) of every markdown table in text."""
    return [
        (m.start(), m.end(), m.group(0)) for m in re.finditer(r"(?m)(?:^\|.*\|[ \t]*\n?)+", text)
    ]


# ------------------------------------------------------------------ layers


@dataclass
class Layer:
    code: str  # L1
    name: str  # Shape
    question: str
    if_fails: str
    cost: str
    body_html: str = ""
    compare: list[tuple[str, str, str]] | None = None  # (row label, FAG, CFA)

    @property
    def anchor(self) -> str:
        return f"how-it-works-{slug_text(self.code + ' ' + self.name)}"


def overview_layers(overview_md: str) -> list[Layer]:
    for _, _, block in find_tables(overview_md):
        rows = table_rows(block)
        if rows and rows[0][1:] == ["Check", "Question it answers", "If it fails", "Cost"]:
            return [Layer(r[0], r[1], r[2], r[3], r[4]) for r in rows[1:]]
    raise GuideError("overview.md has no six-checks table (| | Check | Question it answers ...)")


def attach_layer_details(layers: list[Layer], section: str) -> None:
    parts = re.split(r"(?m)^### (L\d) (.+)$", section)
    found = {parts[i]: (parts[i + 1].strip(), parts[i + 2]) for i in range(1, len(parts), 3)}
    for layer in layers:
        if layer.code not in found:
            raise GuideError(f"how-it-works.md has no '### {layer.code} ...' section")
        name, body = found[layer.code]
        if name != layer.name:
            raise GuideError(
                f"{layer.code} is '{layer.name}' in overview.md, '{name}' in how-it-works.md"
            )
        for start, end, block in find_tables(body):
            rows = table_rows(block)
            if rows and rows[0][1:] == ["FAG", "CFA"]:
                layer.compare = [(r[0], r[1], r[2]) for r in rows[1:]]
                body = body[:start] + '\n<div class="compare-slot"></div>\n' + body[end:]
                break
        rendered = md_to_html(body)
        layer.body_html = rendered.replace(
            '<div class="compare-slot"></div>', compare_html(layer) if layer.compare else ""
        )


def compare_html(layer: Layer) -> str:
    rows = "".join(
        f'<div class="cmp-row"><div class="cmp-label">{html.escape(label)}</div>'
        f'<div class="cmp-cell" data-side="fag">{inline_md(fag)}</div>'
        f'<div class="cmp-cell" data-side="cfa">{inline_md(cfa)}</div></div>'
        for label, fag, cfa in layer.compare or []
    )
    return (
        f'<div class="compare" data-show="fag">'
        f'<div class="seg" role="group" aria-label="Example">'
        f'<button type="button" data-show="fag" aria-pressed="true">FAG</button>'
        f'<button type="button" data-show="cfa" aria-pressed="false">CFA</button>'
        f'<button type="button" data-show="both" aria-pressed="false">Side by side</button>'
        f"</div>"
        f'<div class="cmp-head"><div></div><div data-side="fag">FAG</div>'
        f'<div data-side="cfa">CFA</div></div>{rows}</div>'
    )


def explorer_html(layers: list[Layer]) -> str:
    cards = "".join(
        f'<button type="button" class="layer-card" data-layer="{layer.code}" '
        f'data-outcome="{"dropped" if "dropped" in layer.if_fails else "back"}" '
        f'aria-controls="pane-{layer.code}">'
        f'<span class="lc-code">{layer.code}</span><span class="lc-name">{html.escape(layer.name)}</span>'
        f'<span class="lc-q">{inline_md(layer.question)}</span>'
        f'<span class="lc-meta"><span class="pill">{html.escape(layer.if_fails)}</span>'
        f'<span class="pill pill-cost">{html.escape(layer.cost)}</span></span></button>'
        for layer in layers
    )
    panes = "".join(
        f'<section class="layer-pane" id="pane-{layer.code}" data-layer="{layer.code}" hidden>'
        f'<h3 id="{layer.anchor}">{layer.code} {html.escape(layer.name)}</h3>{layer.body_html}</section>'
        for layer in layers
    )
    return (
        f'<div class="layer-explorer"><div class="layer-cards">{cards}</div>'
        f'<div class="layer-panes">{panes}</div></div>'
    )


# ------------------------------------------------------------------ glossary


GLOSSARY_RE = re.compile(
    r"^\*\*(?P<term>[^*]+)\*\*(?: \((?P<code>[^)]+)\))?:\n(?P<defn>.+?)\n_Avoid_: (?P<avoid>.+)$",
    re.M,
)


def glossary(context_md: str) -> list[dict[str, str]]:
    terms = [
        {
            "term": m["term"].strip(),
            "code": inline_md(m["code"]) if m["code"] else "",
            "defn": inline_md(m["defn"].strip()),
        }
        for m in GLOSSARY_RE.finditer(context_md)
    ]
    if not terms:
        raise GuideError("CONTEXT.md has no glossary entries")
    return terms


# ------------------------------------------------------------------ rendering


def diagram_block(alt: str, name: str) -> str:
    if not (DOCS / "diagrams" / f"{name}.html").exists():
        raise GuideError(f"docs/diagrams/{name}.html is missing; build it with Archify")
    return (
        f'<figure class="diagram" data-diagram="{name}">'
        f'<div class="diagram-frame"><div class="diagram-loading">Loading diagram…</div></div>'
        f'<figcaption>{html.escape(alt)} <button type="button" class="diagram-open">'
        f"Open full screen</button></figcaption></figure>"
    )


LINK_TITLES = {file: title for _, title, file in TABS} | {
    "../CONTEXT.md": "Glossary",
    "../README.md": "README",
}


def rewrite_links(page_html: str, slug: str) -> str:
    # [how-it-works.md](how-it-works.md) reads as a file name; in the guide, show the tab name
    page_html = re.sub(
        r'<a href="([^"#]+)">\1</a>',
        lambda m: f'<a href="{m[1]}">{LINK_TITLES.get(m[1], m[1])}</a>',
        page_html,
    )

    def href(m: re.Match[str]) -> str:
        target = m.group(1)
        if target.startswith(("http://", "https://", "mailto:")):
            return m.group(0)
        if target.startswith("#"):
            return f'href="#{slug}-{target[1:]}"'
        file, _, frag = target.partition("#")
        if file in DOC_TAB:
            tab = DOC_TAB[file]
            return f'href="#{tab}-{frag}"' if frag else f'href="#{tab}"'
        if file == "../CONTEXT.md":
            return 'href="#glossary"'
        return m.group(0)  # the guide sits in docs/, so other relative links still resolve

    page_html = re.sub(r'href="([^"]*)"', href, page_html)
    page_html = re.sub(r' id="([^"]+)"', lambda m: f' id="{slug}-{m.group(1)}"', page_html)
    return page_html.replace("<table>", '<div class="table-wrap"><table>').replace(
        "</table>", "</table></div>"
    )


def render_tab(slug: str, text: str, layers: list[Layer] | None) -> str:
    text = DIAGRAM_RE.sub(lambda m: "\n" + diagram_block(m[1], m[2]) + "\n", text)
    if layers is not None:
        start, end = text.find(LAYERS_START), text.find(LAYERS_END)
        if start < 0 or end < start:
            raise GuideError("how-it-works.md: L1-L6 sections or 'Sent back and dropped' missing")
        attach_layer_details(layers, text[start:end])
        text = text[:start] + "\n" + EXPLORER_MARK + "\n\n" + text[end:]
    page = rewrite_links(md_to_html(text), slug)
    # the h1 becomes the tab title
    page = re.sub(r"<h1[^>]*>.*?</h1>", "", page, count=1)
    if layers is not None:
        # layer panes are rendered separately, so their ids and links need the same rewrite
        page = page.replace(EXPLORER_MARK, rewrite_explorer(explorer_html(layers), slug))
    return page


def rewrite_explorer(fragment: str, slug: str) -> str:
    def href(m: re.Match[str]) -> str:
        return rewrite_links(f'href="{m.group(1)}"', slug)

    fragment = re.sub(r'href="([^"]*)"', href, fragment)
    return fragment.replace("<table>", '<div class="table-wrap"><table>').replace(
        "</table>", "</table></div>"
    )


def script_json(data: object) -> str:
    """JSON safe inside <script type="application/json">."""
    return json.dumps(data, ensure_ascii=False).replace("</", "<\\/")


def build(out: Path) -> Path:
    texts = {slug: (DOCS / file).read_text(encoding="utf-8") for slug, _, file in TABS}
    layers = overview_layers(texts["overview"])
    panels = []
    for slug, title, _ in TABS:
        body = render_tab(slug, texts[slug], layers if slug == "how-it-works" else None)
        panels.append(
            f'<section class="tab-panel" id="{slug}" data-tab="{slug}" role="tabpanel" '
            f'aria-labelledby="tab-{slug}" hidden><h1>{html.escape(title)}</h1>{body}</section>'
        )
    # overview's six-checks rows jump to the matching card
    layer_links = {layer.code: layer.anchor for layer in layers}

    tabs = "".join(
        f'<button type="button" role="tab" id="tab-{slug}" aria-controls="{slug}" '
        f'data-tab="{slug}">{html.escape(title)}</button>'
        for slug, title, _ in TABS
    )
    context_md = (ROOT / "CONTEXT.md").read_text(encoding="utf-8")
    terms = glossary(context_md)
    glossary_html = md_to_html(re.sub(r"^# .*\n", "", context_md, count=1))

    diagrams = {
        p.stem: p.read_text(encoding="utf-8") for p in sorted((DOCS / "diagrams").glob("*.html"))
    }
    template = (ASSETS / "template.html").read_text(encoding="utf-8")
    page = (
        template.replace("{{css}}", (ASSETS / "guide.css").read_text(encoding="utf-8"))
        .replace("{{js}}", (ASSETS / "guide.js").read_text(encoding="utf-8"))
        .replace("{{tabs}}", tabs)
        .replace("{{panels}}", "".join(panels))
        .replace("{{glossary_html}}", glossary_html)
        .replace("{{glossary_json}}", script_json(terms))
        .replace("{{layers_json}}", script_json(layer_links))
        .replace("{{diagrams_json}}", script_json(diagrams))
    )
    out.write_text(page, encoding="utf-8")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=DOCS / "sdgf-guide.html")
    args = parser.parse_args()
    path = build(args.out)
    print(f"wrote {path} ({path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
