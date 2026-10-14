"""Markdown and layout categories for mineru-vl-utils blocks (our own output format)."""

from __future__ import annotations

CATEGORY = {
    "text": "Text", "title": "Title", "section_header": "Section-header", "table": "Table", "image": "Picture", "chart": "Chart",
    "equation": "Formula", "header": "Page-header", "footer": "Page-footer", "page_number": "Page-footer",
    "page_footnote": "Footnote", "image_caption": "Caption", "table_caption": "Caption", "list": "List-item",
    "code": "Text", "image_footnote": "Footnote", "table_footnote": "Footnote", "aside_text": "Text",
}
# Empty blocks that wrap other blocks (an image_block wraps its image blocks). They give no item and no markdown.
WRAPPER_TYPES = {"image_block"}
HEADINGS = {"title": "#", "section_header": "##"}  # the heading marker of a block that has none
_NOT_IN_BODY = {"header", "footer", "page_number"}


def _one(b) -> str:
    c = (b.content or "").strip()
    if not c:
        return ""
    if b.type in HEADINGS:
        return c if c.startswith("#") else f"{HEADINGS[b.type]} {c}"
    if b.type == "equation":
        return f"$$\n{c.strip('$').strip()}\n$$"
    if b.type == "code":
        return f"```\n{c}\n```"
    return c


def blocks_to_markdown(blocks, paratext: bool = True) -> str:
    """Join the blocks in reading order. `paratext` keeps header, footer and page number text."""
    return "\n\n".join(s for b in blocks if (paratext or b.type not in _NOT_IN_BODY) and (s := _one(b)))
