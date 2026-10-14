"""Hybrid text: the PDF text layer replaces the model's text of a block when the two agree.

The layer holds the exact characters, their fonts and their positions. We read it with pypdfium2
(Apache-2.0 or BSD-3-Clause). The fonts give the bold, italic and superscript marks, and the title sizes
give the heading levels.
If an error occurs, the block keeps the model's text.
"""

from __future__ import annotations

import ctypes
import difflib
import math
import re
import statistics
import threading
from collections import Counter
from dataclasses import dataclass, replace

from providus_parse import fixes

AGREE = 0.85  # the minimum similarity of the layer text and the model's text
MIN_WORDS = 20  # a page with fewer layer words has no usable layer
GAP = 0.1  # a gap wider than this part of the font size is a space
SUP_SIZE, SUP_RISE = 0.8, 0.15  # superscript: at most 0.8 of the line size, and raised by 0.15 of it or more
LEVEL_STEP = 0.9  # title sizes within 10 percent are one heading level
MAX_TURN = 10  # degrees: characters turned more than this on the shown page are not read
HEADING_SIZE = 1.2  # a line this many times the body size can be a heading
HEADING_WORDS = 12  # a heading has at most this many words
MIN_LETTERS = 0.6  # a heading line is at least this part letters (not digits or symbols)
BOLD_PAGE = 0.5  # on a page where more than this part of the characters is bold, bold does not mark a heading
# Block types that hold text lines of their own. A text block inside one of them is not a heading.
_HOSTS = {"table", "image", "chart", "code", "equation"}
# Block types that can get the layer text. Code, formulas and tables keep the model's text.
TEXT_TYPES = {"text", "title", "list", "page_footnote", "image_caption", "table_caption", "image_footnote",
              "table_footnote", "header", "footer", "page_number", "aside_text"}
# Font names: "Bold", "Medi" (with weight 700), "CMBX" (TeX bold); "Italic", "Ital", "-It", "Oblique", "CMTI" (TeX italic).
_BOLD = re.compile(r"bold|black|heavy|semibold|demi(?!-?light)|cmbx", re.I)  # DemiLight is a light weight
_ITALIC = re.compile(r"Ital|italic|[Oo]blique|CMTI|(?:-|Bold|Semibold|Medium|Light)It$")
# Inline math: \( or \[, or a $...$ pair. A $ before a digit or a space is a currency sign, and \$ is escaped.
_MATH = re.compile(r"\\\(|\\\[|(?<!\\)\$(?![\d\s])[^$]+(?<!\\)\$")
_LIGATURES = {0xFB00: "ff", 0xFB01: "fi", 0xFB02: "fl", 0xFB03: "ffi", 0xFB04: "ffl", 0xFB05: "st", 0xFB06: "st"}
_HYPHENS = {0x02, 0xAD}  # pdfium gives 0x02 for a hyphen at a line end; 0xAD is a soft hyphen
_BULLETS = set("•●▪◦‣∙·■□◆○–")
_LIST_MARK = re.compile(r"•\s*|(?:[-*+]|\d+[.)])\s+")
# Marks that the layer cannot write (subscript, underline, strikeout, highlight). If the model wrote one, keep its text.
_MODEL_ONLY_MARKS = re.compile(r"<(?:sub|u|ins|s|del|strike|mark)>|~~", re.I)
UNMAPPED = "\ufffd"  # a glyph with no Unicode value
SHORT = 0.95  # a layer text shorter than this part of the model text keeps the model text
# PDFium is not thread-safe, and the server answers in threads. Every PDFium call takes this lock.
# Close every PDFium object inside the lock. If the garbage collector closes it, it can close on any thread.
PDFIUM_LOCK = threading.Lock()


@dataclass
class Char:
    text: str
    x0: float  # the loose box in points, origin at the top left of the shown page (crop box, after rotation)
    y0: float
    x1: float
    y1: float
    base: float  # the baseline y
    size: float  # the font size in points, with the text matrix
    bold: bool
    italic: bool
    hidden: bool  # invisible text (render mode 3), for example an OCR layer


@dataclass
class Page:
    chars: list[Char]
    width: float
    height: float


def _char_text(code: int) -> str:
    if code in _HYPHENS:
        return "-"
    if code in _LIGATURES:
        return _LIGATURES[code]
    if code > 0x10FFFF or (code < 32 and chr(code) not in "\t\n\r\x0b\x0c"):
        return UNMAPPED  # no Unicode value (0) or a control code
    return "" if chr(code).isspace() else chr(code)


def font_style(name: str, weight: int) -> tuple[bool, bool]:
    """(bold, italic) from the font name and the PDFium font weight (0 or -1 if unknown)."""
    return bool(_BOLD.search(name)) or weight >= 600, bool(_ITALIC.search(name))


def read_page(pdf: bytes) -> Page:
    """Read the characters of the first page of a PDF. Generated spaces and line breaks are not read."""
    with PDFIUM_LOCK:
        return _read_page(pdf)


def _read_page(pdf: bytes) -> Page:
    import pypdfium2 as pdfium
    import pypdfium2.raw as raw

    doc = pdfium.PdfDocument(pdf)
    try:
        page = doc[0]
        left, bottom, right, top = page.get_cropbox()
        cw, ch = right - left, top - bottom
        rot = page.get_rotation() % 360

        def show(x: float, y: float) -> tuple[float, float]:  # PDF user space to the shown page
            u, v = x - left, top - y
            return {0: (u, v), 90: (ch - v, u), 180: (cw - u, ch - v), 270: (v, cw - u)}[rot]

        tp = page.get_textpage()
        name, flags, m, box = ctypes.create_string_buffer(512), ctypes.c_int(), raw.FS_MATRIX(), raw.FS_RECTF()
        ox, oy = ctypes.c_double(), ctypes.c_double()
        chars = []
        for i in range(raw.FPDFText_CountChars(tp)):
            text = "" if raw.FPDFText_IsGenerated(tp, i) else _char_text(raw.FPDFText_GetUnicode(tp, i))
            if not text:
                continue
            if raw.FPDFText_HasUnicodeMapError(tp, i) == 1:
                text = UNMAPPED
            raw.FPDFText_GetLooseCharBox(tp, i, box)  # x: origin to origin + advance; y: font ascent to descent
            raw.FPDFText_GetCharOrigin(tp, i, ox, oy)
            raw.FPDFText_GetMatrix(tp, i, m)
            turn = math.degrees(math.atan2(m.b, m.a)) - rot  # 0 for text that is upright on the shown page
            if abs((turn + 180) % 360 - 180) > MAX_TURN:
                continue  # vertical or turned text, for example a stamp in the margin
            n = raw.FPDFText_GetFontInfo(tp, i, name, len(name), flags)
            font = name.value.decode("utf-8", "replace") if 0 < n <= len(name) else ""
            (ax, ay), (bx, by) = show(box.left, box.bottom), show(box.right, box.top)
            obj = raw.FPDFText_GetTextObject(tp, i)
            mode = raw.FPDFTextObj_GetTextRenderMode(obj) if obj else raw.FPDF_TEXTRENDERMODE_FILL
            bold, italic = font_style(font, raw.FPDFText_GetFontWeight(tp, i))
            chars.append(Char(
                text, min(ax, bx), min(ay, by), max(ax, bx), max(ay, by), show(ox.value, oy.value)[1],
                raw.FPDFText_GetFontSize(tp, i) * math.hypot(m.c, m.d),
                bold or mode == raw.FPDF_TEXTRENDERMODE_FILL_STROKE,  # fill and stroke: a "fake bold"
                italic, mode == raw.FPDF_TEXTRENDERMODE_INVISIBLE))
        return Page(chars, cw if rot in (0, 180) else ch, ch if rot in (0, 180) else cw)
    finally:
        doc.close()  # also closes the page and the text page


def _inside(page: Page, bbox) -> list[Char]:
    x0, y0, x1, y1 = bbox[0] * page.width, bbox[1] * page.height, bbox[2] * page.width, bbox[3] * page.height
    return [c for c in page.chars if x0 <= (c.x0 + c.x1) / 2 <= x1 and y0 <= (c.y0 + c.y1) / 2 <= y1]


def _lines(chars: list[Char]) -> list[list[Char]]:
    """Group characters into lines by baseline, then sort each line by x.

    In baseline order, a jump larger than half the median font size starts a new line.
    A raised superscript or a lowered subscript stays in its line. One large character cannot merge two lines.
    """
    if not chars:
        return []
    tol = 0.5 * statistics.median(c.size for c in chars)
    lines: list[list[Char]] = []
    prev = None
    for c in sorted(chars, key=lambda c: c.base):
        if prev is not None and c.base - prev.base <= tol:
            lines[-1].append(c)
        else:
            lines.append([c])
        prev = c
    return [sorted(line, key=lambda c: c.x0) for line in lines]


PARA_GAP = 1.6  # a baseline step larger than this many median line heights starts a paragraph
PARA_INDENT = 1.5  # an indent of this many median font sizes can start a paragraph
PARA_EDGE = 0.5  # a line is "at the left edge" if it starts within this many median font sizes of it


def _paragraphs(chars: list[Char]) -> list[list[list[Char]]]:
    """Group the lines of a block into paragraphs. A wide gap starts a new one.
    So does an indented line that follows a left-edge line and has a left-edge line after it (or is the last line)."""
    lines = _lines(chars)
    if len(lines) < 2:
        return [lines] if lines else []
    height = statistics.median(max(c.y1 for c in ln) - min(c.y0 for c in ln) for ln in lines)
    size = statistics.median(c.size for c in chars)
    left = min(c.x0 for c in chars)
    base = [statistics.median(c.base for c in ln) for ln in lines]
    edge = [ln[0].x0 - left <= PARA_EDGE * size for ln in lines]
    out = [[lines[0]]]
    for i in range(1, len(lines)):
        gap = base[i] - base[i - 1] > PARA_GAP * height
        indent = (lines[i][0].x0 - left > PARA_INDENT * size and edge[i - 1] and (i == len(lines) - 1 or edge[i + 1]))
        if gap or indent:
            out.append([])
        out[-1].append(lines[i])
    return out


def _box(chars) -> tuple[float, float, float, float]:
    return (min(c.x0 for c in chars), min(c.y0 for c in chars), max(c.x1 for c in chars), max(c.y1 for c in chars))


def paragraph_boxes(chars: list[Char]) -> list[tuple[float, float, float, float]]:
    """One box per paragraph of the block, in the units of the characters."""
    return [_box([c for ln in para for c in ln]) for para in _paragraphs(chars)]


def _line_pieces(line: list[Char]) -> list[tuple[str, Char | None]]:
    """The characters of one line, with a space piece (char None) where the gap is wide."""
    out: list[tuple[str, Char | None]] = []
    for prev, c in zip([None, *line], line):
        if prev is not None and c.x0 - prev.x1 > GAP * max(prev.size, c.size):
            out.append((" ", None))
        out.append((c.text, c))
    return out


def _word_at(pieces, end: bool) -> str:
    """The first (end=False) or last (end=True) word of a line."""
    seq = reversed(pieces) if end else pieces
    word = []
    for s, c in seq:
        if c is None:
            break
        word.append(s)
    return "".join(reversed(word) if end else word)


def _pieces(lines: list[list[Char]], model: str, newline: bool) -> list[tuple[str, Char | None]]:
    """Join the lines. A line break joins them if the model text has one, else a space.

    A hyphen at a line end follows the model text: "infor-" + "mation" gives "information" if the model wrote it.
    """
    out: list[tuple[str, Char | None]] = []
    for line in map(_line_pieces, lines):
        if out:
            sep = "\n" if newline else " "
            head, tail = _word_at(out, True), _word_at(line, False)
            if not newline and len(head) > 1 and head.endswith("-") and head[-2].isalpha() and tail[:1].islower():
                if (head[:-1] + tail).casefold() in model:
                    out.pop()
                    sep = ""
                elif (head + tail).casefold() in model:
                    sep = ""
            out.append((sep, None))
        out += line
    return out


def _list_mark(pieces, model_text: str):
    """A bullet glyph at the start becomes the list marker of the model text (or none)."""
    if not pieces or pieces[0][0] not in _BULLETS:
        return pieces
    rest = pieces[1:]
    while rest and rest[0][1] is None:
        rest = rest[1:]
    mark = _LIST_MARK.match(model_text.lstrip())
    return ([(mark.group(0), None)] if mark else []) + rest


def _superscripts(lines: list[list[Char]]) -> set[int]:
    """The ids of the characters that are clearly smaller than the line's main size and raised above its baseline."""
    out = set()
    for line in lines:
        main = Counter(round(c.size, 1) for c in line).most_common(1)[0][0]
        base = statistics.median(c.base for c in line if round(c.size, 1) == main)
        out |= {id(c) for c in line if c.size <= SUP_SIZE * main and c.base < base - SUP_RISE * main}
    return out


_OPEN = ("**", "*", "<sup>")  # bold, italic, sup: the outer mark first
_CLOSE = ("**", "*", "</sup>")


def _render(pieces, mark=lambda c: (False, False, False)) -> str:
    """Join the pieces and write the marks. `mark` gives (bold, italic, sup) for a character.

    The marks nest: a mark stays open while the next characters have it, and only the marks that change
    close or open. A space or line break between two characters with the same marks stays inside the marks.
    """
    out: list[str] = []
    stack: list[int] = []  # the open marks, outer first
    gap = ""
    for s, c in pieces:
        if c is None:
            gap += s
            continue
        want = [k for k, on in enumerate(mark(c)) if on]
        keep = 0
        while keep < len(stack) and stack[keep] in want:
            keep += 1
        out += [_CLOSE[k] for k in reversed(stack[keep:])]
        del stack[keep:]
        out.append(gap)
        out += [_OPEN[k] for k in want if k not in stack]
        stack += [k for k in want if k not in stack]
        out.append(s)
        gap = ""
    out += [_CLOSE[k] for k in reversed(stack)]
    return "".join(out) + gap


_MARK_TOKEN = re.compile(r"(?<!\\)(\*\*|\*|</?sup>)")


def _marked_words(marked: str) -> tuple[list[tuple[str, str, tuple[bool, bool, bool]]], bool]:
    """(gap before, token, (bold, italic, sup)) for each word or punctuation mark of a marked text, and if the marks balance.

    A `*` or `**` is a mark only when it is attached to a word: an opening mark has no space after it,
    a closing mark has no space before it. Else it is text, for example a "* " bullet or "5 * 3".
    """
    out: list[tuple[str, str, tuple[bool, bool, bool]]] = []
    gap, on = "", [False, False, False]

    def add(text: str) -> None:
        nonlocal gap
        for part in re.split(r"(\s+)", text):
            if part.isspace():
                gap += part
            else:  # a word and its punctuation can have different marks
                for tok in re.findall(r"\w+|[^\w\s]", part):
                    out.append((gap, tok, tuple(on)))
                    gap = ""

    pos = 0
    for m in _MARK_TOKEN.finditer(marked):
        add(marked[pos:m.start()])
        pos, tag = m.end(), m.group(1)
        before, after = marked[m.start() - 1:m.start()], marked[m.end():m.end() + 1]
        if tag in ("<sup>", "</sup>"):
            on[2] = tag == "<sup>"
            continue
        k = len(tag) == 1  # 0: bold, 1: italic
        if (not on[k] and after and not after.isspace()) or (on[k] and before and not before.isspace()):
            on[k] = not on[k]
        else:
            add(tag)
    add(marked[pos:])
    return out, not any(on)


def transfer_marks(model: str, marked_layer: str) -> str:
    """The model's text with the marks of the matching layer words. Words that do not match keep the model form.

    The marks nest as in `_render`. A gap between two words with the same marks stays inside the marks.
    """
    words, balanced = _marked_words(model)
    if not balanced:  # a lone * is text, not a mark: do not touch it
        return model
    layer, _ = _marked_words(marked_layer)
    if not any(any(m) for _, _, m in layer):  # the layer has no mark: nothing to move
        return model
    sm = difflib.SequenceMatcher(a=[normalize(w) for _, w, _ in words], b=[normalize(w) for _, w, _ in layer],
                                 autojunk=False)
    old = [m for _, _, m in words]
    marks = list(old)
    for a, b, n in sm.get_matching_blocks():
        for i in range(n):
            marks[a + i] = tuple(x or y for x, y in zip(marks[a + i], layer[b + i][2]))
    if marks == old:
        return model
    out: list[str] = []
    stack: list[int] = []  # the open marks, outer first
    for (gap, w, _), m in zip(words, marks, strict=True):
        want = [k for k, on in enumerate(m) if on]
        keep = 0
        while keep < len(stack) and stack[keep] in want:
            keep += 1
        shut = [_CLOSE[k] for k in reversed(stack[keep:])]
        del stack[keep:]
        out += shut
        out.append(gap)
        out += [_OPEN[k] for k in want if k not in stack]
        stack += [k for k in want if k not in stack]
        out.append(w)
    return "".join(out) + "".join(_CLOSE[k] for k in reversed(stack)) + model[len(model.rstrip()):]


def normalize(text: str) -> str:
    """The comparison form: no markup, no backslash escapes, case folded, one space between words."""
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\\([!-/:-@\[-`{-~])", r"\1", text)
    text = re.sub(r"^\s*#+\s*", "", text, flags=re.M).replace("*", "")
    return " ".join(text.casefold().split())


def similarity(a: str, b: str) -> float:
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    if sm.real_quick_ratio() < AGREE or sm.quick_ratio() < AGREE:  # cheap upper bounds first
        return 0.0
    return sm.ratio()


def _main_size(chars: list[Char]) -> float:
    return Counter(round(c.size, 1) for c in chars).most_common(1)[0][0]


def heading_levels(sizes: list[float]) -> dict[float, int]:
    """Rank the title sizes of a page: the largest is level 1. Sizes within 10 percent are one level. At most 3."""
    levels: dict[float, int] = {}
    tops: list[float] = []
    for size in sorted(set(sizes), reverse=True):
        if not tops or size < LEVEL_STEP * tops[-1]:
            tops.append(size)
        levels[size] = min(len(tops), 3)
    return levels


_NOT_HEADING = re.compile(r"[.,;]$|:\s*\S")  # a sentence end, or a colon with text after it


def looks_like_heading(text: str, chars: list[Char], body_size: float, bold_ok: bool = True) -> bool:
    """True for one short line that is all bold (if `bold_ok`) or at least HEADING_SIZE times the body size."""
    text = re.sub(r"\*+|<[^>]+>", "", text).strip()
    letters = [c for c in chars if c.text.isalpha()]
    if not letters or not 0 < len(text.split()) <= HEADING_WORDS or "\n" in text:
        return False
    if _NOT_HEADING.search(text) or text[0] in _BULLETS or any(c.hidden for c in chars) or len(_lines(chars)) != 1:
        return False
    words = text.split()
    solid = sum(not ch.isspace() for ch in text)
    if len(letters) < 2 or len(letters) < MIN_LETTERS * solid or sum(any(ch.isdigit() for ch in w) for w in words) > 1:
        return False  # "(a)", "Total $5,000", "Q1 Q2 Q3 Q4", "Page 3 of 10", "12 March 2024"
    if statistics.median(c.size for c in letters) >= HEADING_SIZE * body_size:
        return True
    # Bold alone is not enough for one word: "Name:", "Note", "Total", "Yes" are labels.
    return bold_ok and len(words) > 1 and all(c.bold for c in letters)


def _heading_sizes(blocks, page: Page, bold_ok: bool) -> dict[int, float]:
    """The index and main font size of each `text` block that is a heading. A failure skips that block."""
    body = statistics.median(c.size for c in page.chars)
    hosts = [b.bbox for b in blocks if b.type in _HOSTS]
    out = {}
    for i, b in enumerate(blocks):
        if b.type != "text" or _MATH.search(b.content or ""):
            continue
        cx, cy = (b.bbox[0] + b.bbox[2]) / 2, (b.bbox[1] + b.bbox[3]) / 2
        if any(h[0] <= cx <= h[2] and h[1] <= cy <= h[3] for h in hosts):
            continue
        try:
            chars = _inside(page, b.bbox)
            if chars and looks_like_heading(b.content, chars, body, bold_ok):
                out[i] = _main_size(chars)
        except Exception:
            pass
    return out


def _section_header(block, level: int | None, chars: list[Char]):
    """Retype a text block. The heading marker carries the bold, so a fully bold text loses its bold marks.
    The same holds for italic. A mark that covers only part of the text stays. Never `#`."""
    text = block.content.strip()
    letters = [c for c in chars if c.text.isalpha()]
    if letters and all(c.bold for c in letters):
        text = text.replace("**", "")
    if letters and all(c.italic for c in letters):
        text = re.sub(r"(?<!\*)\*(?!\*)", "", text)
    return replace(block, type="section_header", content="#" * max(2, level or 2) + " " + text.lstrip("#").strip())


def _layer_text(block, chars: list[Char], marks: bool, keep_chars: bool = True) -> str | None:
    """The layer text of a text block, or None if it does not agree with the model's text."""
    model = block.content
    if _MODEL_ONLY_MARKS.search(model) or any(c.text == UNMAPPED for c in chars):
        return None
    model_norm = normalize(model)
    lines = _lines(chars)
    pieces = _list_mark(_pieces(lines, model_norm, "\n" in model.strip()), model)
    layer_norm = normalize(_render(pieces))
    if len(layer_norm) < SHORT * len(model_norm) or similarity(layer_norm, model_norm) < AGREE:
        return None
    if "\\$" in model:  # the model escapes the currency sign: do the same
        pieces = [(t.replace("$", "\\$"), c) for t, c in pieces]
    if not marks or any(c.hidden for c in chars):  # an invisible (OCR) layer has no real fonts
        text = _render(pieces)
    else:
        sups = _superscripts(lines)
        no_bold = block.type == "title" and all(c.bold for c in chars)  # the heading marker carries a bold title
        text = _render(pieces, lambda c: (c.bold and not no_bold, c.italic, id(c) in sups))
    return text if keep_chars else transfer_marks(model, text)


def _para_info(page: Page, chars: list[Char]) -> tuple:
    """(box in 0 to 1 page units, line count, character count) for each paragraph of a text block, or () for fewer than 2."""
    paras = _paragraphs(chars)
    if len(paras) < 2:
        return ()
    out = []
    for para in paras:
        x0, y0, x1, y1 = _box([c for ln in para for c in ln])
        out.append(((x0 / page.width, y0 / page.height, x1 / page.width, y1 / page.height),
                    len(para), sum(len(ln) for ln in para)))
    return tuple(out)


def _block(block, page: Page, marks: bool, levels: dict[float, int], keep_chars: bool = True, split: bool = False):
    """Return the block and "used", "kept_model" or None (not a text block).

    A title with layer characters gets its heading level from `levels`.
    """
    model = block.content or ""
    chars = _inside(page, block.bbox)
    status = None
    if block.type in TEXT_TYPES and model.strip():
        text = _layer_text(block, chars, marks, keep_chars) if chars and not _MATH.search(model) else None
        status = "kept_model" if text is None else "used"
        block = block if text is None else replace(block, content=text)
        if split and text is not None and block.type == "text":
            try:  # a failure here keeps one item for the block
                block = replace(block, paras=_para_info(page, chars))
            except Exception:
                pass
    if block.type == "title" and levels and chars and block.content.strip("# \n"):
        level = levels[_main_size(chars)]
        block = replace(block, content="#" * level + " " + block.content.lstrip("#").strip())
    return block, status


def apply(blocks, pdf: bytes | None, flags: dict[str, bool] | None = None):
    """Use the text layer of a one-page PDF in the text blocks. Return the blocks and the counts per page."""
    counts = {"used": 0, "kept_model": 0}
    if not pdf or not fixes.enabled("text_layer", flags):
        return blocks, counts
    try:
        page = read_page(pdf)
        words = len(_render(_pieces(_lines(page.chars), "", False)).split())
    except Exception:
        return blocks, counts
    if words < MIN_WORDS:
        return blocks, counts
    counts["bold_share"] = round(sum(c.bold for c in page.chars) / len(page.chars), 3)
    marks = fixes.enabled("text_layer_marks", flags)
    keep_chars = fixes.enabled("text_layer_chars", flags, default=False)  # off: the model's characters stay, only the marks move
    heads = {}
    if fixes.enabled("heading_class", flags, default=False):  # off by default
        try:
            heads = _heading_sizes(blocks, page, counts["bold_share"] <= BOLD_PAGE)
        except Exception:
            heads = {}
    levels = {}
    if fixes.enabled("text_layer_headings", flags):
        try:
            sizes = [_main_size(cs) for b in blocks if b.type == "title" and (cs := _inside(page, b.bbox))]
            levels = heading_levels(sizes + list(heads.values()))
            if not sizes:  # no title: the largest section header is `##`, the next `###`
                levels = {k: min(v + 1, 3) for k, v in levels.items()}
        except Exception:
            levels = {}
    split = fixes.enabled("para_split", flags, default=False)
    out = []
    for i, b in enumerate(blocks):
        try:
            b2, status = _block(b, page, marks, levels, keep_chars, split)
            if i in heads:
                b2 = _section_header(b2, levels.get(heads[i]), _inside(page, b.bbox))
        except Exception:  # never fail a page: keep the model's text of this block
            b2, status = b, "kept_model"
        if status:
            counts[status] += 1
        out.append(b2)
    return out, counts
