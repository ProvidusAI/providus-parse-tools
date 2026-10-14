"""Providus parse pipeline. It extends mineru-vl-utils (OpenDataLab, Apache-2.0).

The MinerU client does two steps: layout detection on the page, then recognition of each region.
This module adds our output fixes and our markdown and layout output.
"""

from __future__ import annotations

import html
import itertools
import re
from dataclasses import dataclass, field

from providus_parse import fixes, textlayer
from providus_parse.extract import extract_blocks
from providus_parse.markdown import CATEGORY, HEADINGS, WRAPPER_TYPES, blocks_to_markdown


@dataclass
class Block:
    type: str
    bbox: tuple[float, float, float, float]
    angle: int
    content: str
    paras: tuple[tuple[tuple[float, float, float, float], int, int], ...] = ()  # (box, line count, character count) of each paragraph


@dataclass
class Item:
    category: str
    bbox: tuple[float, float, float, float]
    text: str
    type: str = ""  # the mineru-vl-utils block type, for example page_number (its category is Page-footer)


@dataclass
class PageResult:
    markdown: str
    items: list[Item] = field(default_factory=list)
    text_layer: dict[str, int] = field(default_factory=lambda: {"used": 0, "kept_model": 0})


def _item_text(block) -> str:
    """The text of a layout item. A heading has no marker here; the markdown keeps it.
    A chart with an HTML table gives its plain cell text (no tags, no caption)."""
    if block.type == "chart" and "<table" in block.content:
        t = re.sub(r"<caption>.*?</caption>", " ", block.content, flags=re.S)
        return " ".join(html.unescape(re.sub(r"<[^>]+>", " ", t)).split())
    return re.sub(r"^#{1,6}\s+", "", block.content) if block.type in HEADINGS else block.content


def _split_text(text: str, paras: tuple) -> list[str] | None:
    """The text of each paragraph: by lines if the text has as many lines as the layer, else by the share of characters."""
    lines = text.split("\n")
    if len(lines) == sum(n for _, n, _ in paras):
        out, i = [], 0
        for _, n, _ in paras:
            out.append("\n".join(lines[i:i + n]))
            i += n
    else:  # cut at word boundaries by the share of characters (a word and its space count len + 1)
        words, total = text.split(), sum(c for _, _, c in paras)
        ends = list(itertools.accumulate(len(w) + 1 for w in words))
        out, i, done = [], 0, 0
        for k, (_, _, c) in enumerate(paras):
            done += c
            target = ends[-1] * done / total
            j = len(words) if k == len(paras) - 1 else sum(1 for w, e in zip(words, ends) if e - (len(w) + 1) / 2 <= target)
            out.append(" ".join(words[i:j]))
            i = max(i, j)
    return out if all(t.strip() for t in out) else None


def _close_marks(t: str) -> str:
    """Drop the marks that a split left unbalanced in one item."""
    if t.count("<sup>") != t.count("</sup>"):
        t = t.replace("<sup>", "").replace("</sup>", "")
    if t.count("**") % 2:
        t = t.replace("**", "")
    star = r"(?<!\*)\*(?!\*)"
    return re.sub(star, "", t) if len(re.findall(star, t)) % 2 else t


def _items(block) -> list[Item]:
    category = CATEGORY.get(block.type, "Text")
    if block.paras:
        try:
            texts = _split_text(block.content, block.paras)
            if texts:
                return [Item(category, box, _close_marks(t), block.type) for (box, _, _), t in zip(block.paras, texts)]
        except Exception:
            pass  # one item for the block
    return [Item(category, block.bbox, _item_text(block), block.type)]


class ProvidusParser:
    def __init__(self, server_url: str, fixes: dict[str, bool] | None = None, client=None):
        if client is None:
            from mineru_vl_utils import MinerUClient

            client = MinerUClient(backend="http-client", server_url=server_url,
                                  image_analysis=True, enable_table_formula_eq_wrap=True)
        self.client = client
        self.flags = fixes or {}

    def parse_page(self, image, pdf: bytes | None = None) -> PageResult:
        """Parse one page image. `pdf` is an optional one-page PDF of the same page (its text layer)."""
        blocks = [Block(b.type, tuple(float(v) for v in b.bbox), b.angle or 0, b.content or "")
                  for b in extract_blocks(self.client, image, self.flags) if b.type not in WRAPPER_TYPES]
        # Wrappers go out before fixes.apply, so no fix sees an empty block that overlaps its children.
        blocks, counts = textlayer.apply(blocks, pdf, self.flags)
        blocks = fixes.apply(blocks, self.flags)
        items = [i for b in blocks for i in _items(b)]
        markdown = blocks_to_markdown(blocks, paratext=fixes.enabled("header_footer", self.flags))
        return PageResult(markdown, items, counts)
