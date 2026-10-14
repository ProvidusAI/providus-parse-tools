"""General output fixes. Each fix has its own flag. Most are on by default; text_layer_chars, heading_class and
para_split are off by default (the settings of the ParseBench runs).

Turn a fix off with the environment variable PROVIDUS_FIX_<NAME>=0, or on with =1, or use the `fixes` argument of
ProvidusParser. The `fixes` argument wins over the environment.
"""

import dataclasses
import html as _html
import os
import re


def enabled(name: str, flags: dict[str, bool] | None = None, default: bool = True) -> bool:
    if flags and name in flags:
        return bool(flags[name])
    return os.environ.get(f"PROVIDUS_FIX_{name.upper()}", "1" if default else "0").strip().lower() not in {"0", "false", "off", "no"}


_ROW = re.compile(r"<tr\b[^>]*>.*?</tr>", re.S | re.I)
_TD = re.compile(r"<td\b[^>]*>", re.I)


def _span(cell_tag: str, name: str) -> int:
    m = re.search(rf"\b{name}\s*=\s*[\"']?(\d+)", cell_tag, re.I)
    return max(1, int(m.group(1))) if m else 1  # a span below 1 is 1


def header_row_count(rows: list[str], first_row: bool = True) -> int:
    """How many top rows are header rows. 0 only for a table with one row.

    The count starts at 1 (the first row). It grows to cover the rowspans of the header rows.
    If the last header row has a colspan, the next row is a header row too. At least one body row stays:
    the count is clamped, and the colspan row is not added if it would use the last row.
    With first_row=False, a table needs a rowspan in its first row, else it has no header.
    """
    if len(rows) < 2:
        return 0
    cells = [_TD.findall(r) for r in rows]
    k, i = 1, 0
    while i < min(k, len(rows)):  # rowspan closure over the header rows; a span past the table end is clamped below
        k = max(k, i + max((_span(t, "rowspan") for t in cells[i]), default=1))
        i += 1
    if k == 1 and not first_row:
        return 0
    k = min(k, len(rows) - 1)
    if k + 1 < len(rows) and any(_span(t, "colspan") > 1 for t in cells[k - 1]):
        k += 1
    return k


def add_table_header(html: str, first_row: bool = True) -> str:
    """Put header rows of a MinerU table (all <td>) into <thead> with <th> cells. No header found: no change."""
    if re.search(r"<(?:th|thead|tbody)\b", html, re.I) or len(re.findall(r"<table\b", html, re.I)) > 1:
        return html  # already has a header, or is a nested table (the row regex cannot tell the rows apart)
    rows = list(_ROW.finditer(html))
    n = header_row_count([m.group(0) for m in rows], first_row)
    if not n:
        return html
    head = re.sub(r"<(/?)td\b", r"<\1th", html[rows[0].start():rows[n - 1].end()], flags=re.I)
    return f"{html[:rows[0].start()]}<thead>{head}</thead>{html[rows[n - 1].end():]}"


_CAPTION_TYPES = {"image_caption", "table_caption", "title", "section_header"}


def caption_for_chart(blocks, i):
    """The chart's title or caption: the nearest caption or title up to 2 blocks before it, else a caption just
    after it. A short text line (12 words or fewer, no final full stop, with a letter) also counts."""
    def usable(b):
        text = re.sub(r"^\*\*(.+)\*\*$", r"\1", (b.content or "").strip().lstrip(">#").strip())  # a trailing * is a footnote mark
        if not text or "<table" in text or text.startswith("|"):
            return None
        if b.type in _CAPTION_TYPES:
            return text
        if b.type == "text" and len(text.split()) <= 12 and not text.endswith(".") and re.search(r"[^\W\d_]", text):
            return text
        return None
    for j in (i - 1, i - 2):
        if j >= 0 and (t := usable(blocks[j])):
            return t
    if i + 1 < len(blocks) and blocks[i + 1].type in {"image_caption", "table_caption"}:
        return usable(blocks[i + 1])
    return None


def _cell(tag: str, text: str) -> str:
    text = re.sub(r"&lt;(br\s*/?)&gt;", r"<\1>", _html.escape(text, quote=False))  # a <br> in a cell stays a <br>
    return f"<{tag}>{text}</{tag}>"


def md_table_to_html(md: str) -> str | None:
    """A markdown table (as MinerU writes a chart) as HTML. The first row is the header. Not a table: None."""
    lines = [ln.strip() for ln in md.strip().splitlines()]
    if not lines or not all(ln.startswith("|") for ln in lines):
        return None
    rows = [[c.strip() for c in ln.strip("|").split("|")] for ln in lines
            if not re.fullmatch(r"\|?[\s:|-]+\|?", ln)]  # the |---| line gives no row
    if not rows:
        return None
    head = "<thead><tr>" + "".join(_cell("th", c) for c in rows[0]) + "</tr></thead>"
    body = "".join("<tr>" + "".join(_cell("td", c) for c in r) + "</tr>" for r in rows[1:])
    return f"<table>{head}{body}</table>"


def add_table_caption(table, caption):
    if "<caption>" in table:
        return table
    return re.sub(r"(<table\b[^>]*>)", lambda m: m.group(1) + f"<caption>{_html.escape(caption)}</caption>",
                  table, count=1)


def apply(blocks, flags):
    if enabled("table_header", flags):
        first_row = enabled("table_header_first_row", flags)
        blocks = [_safe_header(b, first_row) if b.type == "table" else b for b in blocks]
    if enabled("chart_caption", flags):
        blocks = [_safe_chart(blocks, i) if b.type == "chart" else b for i, b in enumerate(blocks)]
    return blocks


def _safe_chart(blocks, i):
    """Same rule as _safe_header: if the fix raises on one chart, the chart stays as it is."""
    block = blocks[i]
    try:
        caption = caption_for_chart(blocks, i)
        table = block.content if "<table" in block.content else md_table_to_html(block.content)
        if not caption or not table:
            return block
        return dataclasses.replace(block, content=add_table_caption(table, caption))
    except Exception:
        return block


def _safe_header(block, first_row: bool):
    """A fix must not fail a page. If it raises on one block, the block stays as it is."""
    try:
        return dataclasses.replace(block, content=add_table_header(block.content, first_row))
    except Exception:
        return block
