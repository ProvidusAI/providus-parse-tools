"""Text layer: tiny PDFs written by hand in each test, so no PDF writer package is needed."""

import base64
import json
import math
from types import SimpleNamespace

import pytest

from providus_parse import textlayer
from providus_parse.pipeline import Block, ProvidusParser
from providus_parse.server import predict

W, H = 612, 792  # page size in points


def tiny_pdf(spans, rotate=0, cropbox=None, angle=0):
    """A one-page PDF. A span is (x, baseline_y_from_top, font, size, text, rise=0, render_mode=0).

    `angle` turns the text counterclockwise (degrees) in PDF space; angle = rotate gives upright text on the shown page.
    The fonts are the standard 14 fonts (Helvetica, Helvetica-Bold, Times-Italic, ...), so nothing is embedded.
    """
    fonts = sorted({s[2] for s in spans})
    fid = {f: f"F{i}" for i, f in enumerate(fonts)}
    ops = []
    for x, y, font, size, text, *rest in spans:
        rise, mode = (list(rest) + [0, 0])[:2]
        t = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        cos, sin = (round(f(math.radians(angle)), 6) for f in (math.cos, math.sin))
        ops.append(f"BT /{fid[font]} {size} Tf {mode} Tr {rise} Ts {cos} {sin} {-sin} {cos} {x} {H - y} Tm ({t}) Tj ET")
    stream = "\n".join(ops).encode("latin-1")
    extra = (f" /Rotate {rotate}" if rotate else "") + (" /CropBox [%s]" % " ".join(map(str, cropbox)) if cropbox else "")
    res = " ".join(f"/{fid[f]} {5 + i} 0 R" for i, f in enumerate(fonts))
    objs = ["<< /Type /Catalog /Pages 2 0 R >>", "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {W} {H}]{extra} /Resources << /Font << {res} >> >> "
            "/Contents 4 0 R >>", None]
    objs += [f"<< /Type /Font /Subtype /Type1 /BaseFont /{f} /Encoding /WinAnsiEncoding >>" for f in fonts]
    out, offsets = bytearray(b"%PDF-1.4\n"), []
    for i, o in enumerate(objs, 1):
        offsets.append(len(out))
        body = b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream" if o is None else o.encode()
        out += b"%d 0 obj\n" % i + body + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objs) + 1) + b"".join(b"%010d 00000 n \n" % o for o in offsets)
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objs) + 1, xref)
    return bytes(out)


FILLER = [(72, 760, "Helvetica", 8, " ".join(f"word{i}" for i in range(25)))]  # 25 words low on the page
BODY = (0.1, 0.08, 0.9, 0.16)  # 0-1 box around the lines at y = 80 and y = 100 (points from the top)


def B(t, content, bbox=BODY):
    return Block(type=t, bbox=bbox, angle=0, content=content)


def uk(counts):
    """The two block counts (the page also reports bold_share when it has a layer)."""
    return {k: counts[k] for k in ("used", "kept_model")}


def run(spans, blocks, flags=None, **pdf_args):
    return textlayer.apply(blocks, tiny_pdf(spans + FILLER, **pdf_args), flags)


@pytest.fixture(autouse=True)
def _layer_characters(monkeypatch):
    """The tests below check the layer characters. The default (marks only) has its own tests at the end."""
    monkeypatch.setenv("PROVIDUS_FIX_TEXT_LAYER_CHARS", "1")


# --- the reader ---

def test_read_page_gives_characters_with_fonts_sizes_and_shown_page_boxes():
    page = textlayer.read_page(tiny_pdf([(72, 100, "Helvetica-Bold", 12, "Hi"), (100, 100, "Times-Italic", 12, "a"),
                                         (72, 200, "Helvetica", 12, "x", 0, 3)]))
    assert (page.width, page.height) == (W, H)
    h, i, a, x = page.chars
    assert (h.text, h.bold, h.italic, h.size, round(h.base)) == ("H", True, False, 12, 100)
    assert h.x0 == pytest.approx(72) and h.x1 < 82 and h.y0 < 100 < h.y1  # y grows down the page
    assert (a.bold, a.italic) == (False, True)
    assert x.hidden and not h.hidden  # render mode 3: invisible text, as an OCR layer writes it


def test_read_page_uses_the_text_matrix_for_the_size():
    pdf = tiny_pdf([(72, 100, "Helvetica", 12, "Hi")]).replace(b"12 Tf 0 Tr 0 Ts 1 0 0 1", b"1 Tf 0 Tr 0 Ts 12 0 0 12")
    assert textlayer.read_page(pdf).chars[0].size == pytest.approx(12)


def _centre(c):
    return (c.x0 + c.x1) / 2, (c.y0 + c.y1) / 2


@pytest.mark.parametrize("rotate", [0, 90, 180, 270])
def test_read_page_boxes_match_the_rendered_page_for_each_rotation(rotate):
    pdf = tiny_pdf([(306, 396, "Helvetica", 12, "Hi there")], rotate=rotate, angle=rotate)  # upright when shown
    page = textlayer.read_page(pdf)
    assert (page.width, page.height) == ((W, H) if rotate in (0, 180) else (H, W))
    import pypdfium2 as pdfium
    with textlayer.PDFIUM_LOCK:  # PDFium is not thread-safe: every call takes the one lock
        doc = pdfium.PdfDocument(pdf)
        try:
            img = doc[0].render(scale=1).to_pil().convert("L")  # 1 pixel = 1 point, rotation applied
        finally:
            doc.close()  # an open document can close later on a thread of the thread test, outside the lock
    assert img.size == (page.width, page.height)
    for c in page.chars:
        assert min(img.crop((int(c.x0), int(c.y0), int(c.x1) + 1, int(c.y1) + 1)).getextrema()) < 128, c.text
    assert "".join(c.text for c in page.chars) == "Hithere"
    assert [c.x0 for c in page.chars] == sorted(c.x0 for c in page.chars)  # left to right on the shown page
    assert max(c.base for c in page.chars) - min(c.base for c in page.chars) < 0.5


def test_vertical_text_is_not_read():
    pdf = tiny_pdf([(306, 396, "Helvetica", 12, "up")], angle=90)  # for example a stamp in the margin
    assert textlayer.read_page(pdf).chars == []


def test_a_large_stray_character_does_not_merge_two_lines():
    spans = [(72, 100, R, 10, "The quick brown fox"), (72, 111, R, 10, "jumps over the dog."), (400, 101.3, R, 20, "]")]
    blocks, _ = run(spans, [B("text", "The quick brovvn fox ]\njumps over the dog.")])
    assert blocks[0].content == "The quick brown fox ]\njumps over the dog."


def test_read_page_follows_the_crop_box():
    page = textlayer.read_page(tiny_pdf([(72, 100, "Helvetica", 12, "H")], cropbox=(50, 92, 562, 742)))
    assert (page.width, page.height) == (512, 650)
    # glyph boxes differ a little by platform (pdfium substitutes Helvetica on Linux): 0.3 pt is far below a crop-box error
    assert _centre(page.chars[0]) == pytest.approx((26.33, 45.84), abs=0.3)


# --- block text and agreement ---

LINES = [(72, 80, "Helvetica", 12, "The quick brown fox"), (72, 100, "Helvetica", 12, "jumps over the dog.")]


def test_layer_text_replaces_the_model_text_when_they_agree():
    blocks, counts = run(LINES, [B("text", "The quick brovvn fox jumps over the dog.")])
    assert blocks[0].content == "The quick brown fox jumps over the dog."
    assert uk(counts) == {"used": 1, "kept_model": 0}


def test_lines_join_with_a_line_break_when_the_model_text_has_one():
    blocks, _ = run(LINES, [B("text", "The quick brown fox\njumps over the dog.")])
    assert blocks[0].content == "The quick brown fox\njumps over the dog."


def test_model_text_stays_when_the_layer_disagrees():
    blocks, counts = run(LINES, [B("text", "Something else entirely, not this layer.")])
    assert blocks[0].content == "Something else entirely, not this layer."
    assert uk(counts) == {"used": 0, "kept_model": 1}


def test_space_comes_from_the_gap_between_characters():
    spans = [(72, 80, "Helvetica", 12, "ab"), (86.0, 80, "Helvetica", 12, "cd"), (102, 80, "Helvetica", 12, "ef")]
    blocks, _ = run(spans, [B("text", "abcd ef")])
    assert blocks[0].content == "abcd ef"  # 1.5 pt is a kerning gap; about 3 pt is a word space


def test_hyphen_at_a_line_end_follows_the_model_text():
    spans = [(72, 80, "Helvetica", 12, "a long infor-"), (72, 100, "Helvetica", 12, "mation and a well-"),
             (72, 120, "Helvetica", 12, "known name")]
    box = (0.1, 0.08, 0.9, 0.16)
    blocks, _ = run(spans, [B("text", "a long information and a well-known name", box)])
    assert blocks[0].content == "a long information and a well-known name"


def test_only_characters_inside_the_block_box_are_used():
    spans = LINES + [(72, 300, "Helvetica", 12, "other block text")]
    blocks, _ = run(spans, [B("text", "The quick brown fox jumps over the dog."),
                            B("text", "other blocc text", (0.1, 0.36, 0.9, 0.39))])
    assert [b.content for b in blocks] == ["The quick brown fox jumps over the dog.", "other block text"]


def test_list_marker_follows_the_model_text():
    spans = [(72, 80, "Helvetica", 12, "\x95 first item of the list")]  # WinAnsi 0x95 is a bullet
    blocks, _ = run(spans, [B("text", "- first itern of the list")])
    assert blocks[0].content == "- first item of the list"


def test_math_code_tables_and_empty_blocks_keep_the_model_text():
    blocks = [B("text", "The quick brown fox $x$ jumps over the dog."), B("code", "The quick brown fox"),
              B("table", "<table></table>"), B("text", "")]
    out, counts = run(LINES, blocks)
    assert out == blocks and uk(counts) == {"used": 0, "kept_model": 1}  # the math text block counts as kept_model


@pytest.mark.parametrize("pdf", [None, b"", b"not a pdf"])
def test_no_pdf_or_a_bad_pdf_changes_nothing(pdf):
    blocks = [B("text", "x")]
    assert textlayer.apply(blocks, pdf) == (blocks, {"used": 0, "kept_model": 0})


def test_a_page_with_few_words_has_no_layer():
    blocks = [B("text", "The quick brown fox jumps over the dog.")]
    out, counts = textlayer.apply(blocks, tiny_pdf(LINES))  # 8 words, no filler
    assert out == blocks and counts == {"used": 0, "kept_model": 0}


def test_flag_off_changes_nothing(monkeypatch):
    blocks = [B("text", "The quick brovvn fox jumps over the dog.")]
    assert run(LINES, blocks, {"text_layer": False})[0] == blocks
    monkeypatch.setenv("PROVIDUS_FIX_TEXT_LAYER", "0")
    assert run(LINES, blocks)[0] == blocks


def test_an_error_in_one_block_keeps_the_model_text_of_that_block(monkeypatch):
    real = textlayer._block

    def flaky(block, *args):
        if block.content.startswith("boom"):
            raise RuntimeError("bad block")
        return real(block, *args)

    monkeypatch.setattr(textlayer, "_block", flaky)
    blocks, counts = run(LINES, [B("text", "boom"), B("text", "The quick brovvn fox jumps over the dog.")])
    assert [b.content for b in blocks] == ["boom", "The quick brown fox jumps over the dog."]
    assert uk(counts) == {"used": 1, "kept_model": 1}


# --- the parser and the server ---

class _C:
    def __init__(self, blocks):
        self.blocks = blocks

    def two_step_extract(self, image):
        return self.blocks


OLD_PATH = {"image_block": False, "text_request_form": False, "chart_request_form": False}


def test_parser_and_server_use_the_pdf_and_report_the_counts():
    model = [SimpleNamespace(type="text", bbox=BODY, angle=0, content="The quick brovvn fox jumps over the dog.")]
    parser = ProvidusParser("u", fixes=OLD_PATH, client=_C(model))
    r = parser.parse_page(None, pdf=tiny_pdf(LINES + FILLER))
    assert r.markdown == "The quick brown fox jumps over the dog." and uk(r.text_layer) == {"used": 1, "kept_model": 0}

    import io

    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (40, 20), "white").save(buf, "PNG")
    body = {"image_base64": base64.b64encode(buf.getvalue()).decode(),
            "pdf_base64": base64.b64encode(tiny_pdf(LINES + FILLER)).decode()}
    out = predict(json.dumps(body).encode(), parser)
    assert out["status"] == "success" and uk(out["text_layer"]) == {"used": 1, "kept_model": 0}
    assert out["markdown"] == "The quick brown fox jumps over the dog."
    del body["pdf_base64"]
    out = predict(json.dumps(body).encode(), parser)
    assert out["markdown"] == "The quick brovvn fox jumps over the dog." and out["text_layer"] == {"used": 0, "kept_model": 0}
    body["pdf_base64"] = "%%%not base64%%%"
    assert predict(json.dumps(body).encode(), parser)["status"] == "success"


# --- marks (text_layer_marks) ---

def words(y, *spec, x=72, step=60):
    """One span for each (font, word), `step` points apart, so the gaps give the spaces."""
    return [(x + k * step, y, font, 12, w) for k, (font, w) in enumerate(spec)]


R, BD, IT, BI = "Helvetica", "Helvetica-Bold", "Times-Italic", "Times-BoldItalic"


def test_bold_and_italic_runs_get_markdown_marks_and_adjacent_runs_merge():
    spans = words(80, (R, "Normal"), (BD, "bold"), (BD, "words"), (R, "then"), (IT, "slanted"), (BI, "both"), (R, "end."))
    blocks, _ = run(spans, [B("text", "Normal bold words then slanted both end.")])
    assert blocks[0].content == "Normal **bold words** then *slanted **both*** end."  # italic stays open


def test_a_run_continues_over_a_line_break():
    spans = words(80, (BD, "all"), (BD, "bold")) + words(100, (BD, "next"), (BD, "line"))
    blocks, _ = run(spans, [B("text", "all bold next line")])
    assert blocks[0].content == "**all bold next line**"


def test_small_raised_characters_are_superscript_and_lowered_ones_are_not():
    spans = [(72, 80, R, 12, "E = mc"), (110.5, 80, R, 7, "2", 4), (130, 80, R, 12, "and H"), (162.2, 80, R, 7, "2", -2),
             (166.3, 80, R, 12, "O")]
    blocks, _ = run(spans, [B("text", "E = mc2 and H2O")])
    assert blocks[0].content == "E = mc<sup>2</sup> and H2O"


def test_footnote_number_at_the_line_start_is_superscript():
    spans = [(72, 80, R, 7, "1", 4), (80, 80, R, 12, "A footnote")]
    blocks, _ = run(spans, [B("page_footnote", "1 A footnote")])
    assert blocks[0].content == "<sup>1</sup> A footnote"


def test_a_title_that_is_all_bold_gets_no_bold_mark_but_a_partly_bold_title_does():
    spans = words(80, (BD, "Results")) + [(72, 300, BD, 12, "Part"), (132, 300, R, 12, "two")]
    blocks, _ = run(spans, [B("title", "Results"), B("title", "Part two", (0.1, 0.36, 0.9, 0.39))],
                    {"text_layer_headings": False})
    assert [b.content for b in blocks] == ["Results", "**Part** two"]


def test_marks_flag_off_gives_plain_layer_text():
    spans = words(80, (R, "Normal"), (BD, "bold"))
    blocks, counts = run(spans, [B("text", "Normal boId")], {"text_layer_marks": False})
    assert blocks[0].content == "Normal bold" and counts["used"] == 1


def test_invisible_text_gets_no_marks():
    spans = [(72, 80, BD, 12, "Hidden", 0, 3), (132, 80, BD, 12, "words", 0, 3)]
    blocks, _ = run(spans, [B("text", "Hidden words")])
    assert blocks[0].content == "Hidden words"


# --- heading levels (text_layer_headings) ---

def titles(*sizes):
    """One bold title for each size, 100 points apart, with its 0-1 box."""
    spans, boxes = [], []
    for k, size in enumerate(sizes):
        y = 100 + 100 * k
        spans.append((72, y, BD, size, f"Heading {k}"))
        boxes.append((0.1, (y - 30) / H, 0.9, (y + 10) / H))
    return spans, [B("title", f"Heading {k}", box) for k, box in enumerate(boxes)]


def test_title_sizes_rank_into_three_heading_levels():
    spans, blocks = titles(20, 16, 13, 11)
    out, _ = run(spans, blocks)
    assert [b.content for b in out] == ["# Heading 0", "## Heading 1", "### Heading 2", "### Heading 3"]


def test_title_sizes_within_ten_percent_are_one_level():
    spans, blocks = titles(20, 18.5, 14)
    out, _ = run(spans, blocks)
    assert [b.content for b in out] == ["# Heading 0", "# Heading 1", "## Heading 2"]


def test_one_title_size_keeps_one_level_and_the_model_marker_is_replaced():
    spans, blocks = titles(14, 14)
    blocks[1] = B("title", "# Heading 1", blocks[1].bbox)
    out, _ = run(spans, blocks)
    assert [b.content for b in out] == ["# Heading 0", "# Heading 1"]


def test_headings_flag_off_keeps_the_title_text_without_a_level():
    spans, blocks = titles(20, 16)
    out, _ = run(spans, blocks, {"text_layer_headings": False})
    assert [b.content for b in out] == ["Heading 0", "Heading 1"]


def test_marks_nest_when_a_run_changes_without_a_space():
    spans = [(72, 80, IT, 12, "(see"), (98, 80, BI, 12, "Smith"), (127.4, 80, IT, 12, ", 2004)"), (180, 80, R, 12, "end")]
    blocks, _ = run(spans, [B("text", "(see Smith, 2004) end")])
    assert blocks[0].content == "*(see **Smith**, 2004)* end"  # never "*(see* ***Smith***, ..." with colliding runs


def test_dollar_amounts_are_not_math_and_follow_the_model_escaping():
    spans = [(72, 80, R, 12, "Revenue was $1,234 and costs $56.")]
    blocks, counts = run(spans, [B("text", "Revenue was $1,284 and costs $56.")])
    assert blocks[0].content == "Revenue was $1,234 and costs $56." and counts["used"] == 1
    blocks, _ = run(spans, [B("text", "Revenue was \\$1,284 and costs \\$56.")])
    assert blocks[0].content == "Revenue was \\$1,234 and costs \\$56."


@pytest.mark.parametrize("model", ["The quick brown fox $x^{2}$ jumps over the dog.", "The quick \\(x\\) fox jumps over the dog."])
def test_latex_math_keeps_the_model_text(model):
    blocks, counts = run(LINES, [B("text", model)])
    assert blocks[0].content == model and uk(counts) == {"used": 0, "kept_model": 1}


@pytest.mark.parametrize("name,weight,style", [
    ("Helvetica-Bold", 0, (True, False)), ("ABCDEF+Calibri-Bold", 700, (True, False)), ("NimbusRomNo9L-Medi", 700, (True, False)),
    ("CMBX10", 570, (True, False)), ("Arial-BoldItalicMT", 0, (True, True)), ("NimbusRomNo9L-ReguItal", 390, (False, True)),
    ("MinionPro-It", 0, (False, True)), ("MinionPro-BoldIt", 0, (True, True)), ("CMTI10", 0, (False, True)),
    ("Helvetica-Oblique", 0, (False, True)), ("DigitalSans", 400, (False, False)), ("CMR10", 345, (False, False)),
    ("ArialMT", 400, (False, False))])
def test_font_style_from_the_name_and_weight(name, weight, style):
    assert textlayer.font_style(name, weight) == style  # names seen in real LaTeX and Office PDFs



# --- thread safety and block edge cases ---

def test_read_page_is_safe_from_many_threads():
    import threading
    spans = [(40 + 3 * (k % 7), 40 + 12 * k, font, 9, f"Line {k} with the fonts of a real page, 0123456789.")
             for k, font in zip(range(55), [R, BD, IT, BI, "Courier", "Times-Roman", "Symbol"] * 8)]
    pdf = tiny_pdf(spans)
    want = [(c.text, c.x0, c.base, c.bold) for c in textlayer.read_page(pdf).chars]
    got, errors = [], []

    def go():
        try:
            for _ in range(10):
                got.append([(c.text, c.x0, c.base, c.bold) for c in textlayer.read_page(pdf).chars])
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=go) for _ in range(8)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert not errors and len(got) == 80 and all(g == want for g in got)


@pytest.mark.parametrize("model", ["The quick brown fox H<sub>2</sub>O jumps over the dog.",
                                   "The quick brown <u>fox</u> jumps over the dog.",
                                   "The quick brown <ins>fox</ins> jumps over the dog.",
                                   "The quick brown <s>fox</s> jumps over the dog.",
                                   "The quick brown <del>fox</del> jumps over the dog.",
                                   "The quick brown <strike>fox</strike> jumps over the dog.",
                                   "The quick brown <mark>fox</mark> jumps over the dog.",
                                   "The quick brown ~~fox~~ jumps over the dog."])
def test_model_marks_the_layer_cannot_write_keep_the_model_text(model):
    blocks, counts = run(LINES, [B("text", model)])
    assert blocks[0].content == model and uk(counts) == {"used": 0, "kept_model": 1}


def test_fill_and_stroke_text_is_bold():
    spans = [(72, 80, R, 12, "Fake", 0, 2), (132, 80, R, 12, "bold")]  # render mode 2: fill, then stroke
    blocks, _ = run(spans, [B("text", "Fake bold")])
    assert blocks[0].content == "**Fake** bold"


def test_a_shorter_layer_keeps_the_model_text():
    spans = [(72, 80, R, 12, "The quick brown fox jumps over the")]  # the layer lacks the last word
    blocks, counts = run(spans, [B("text", "The quick brown fox jumps over the dog.")])
    assert blocks[0].content == "The quick brown fox jumps over the dog." and uk(counts) == {"used": 0, "kept_model": 1}


def test_an_unmapped_glyph_keeps_the_model_text():
    spans = [(72, 80, R, 12, "The quick brown fox jumps \x01 over the dog.")]  # 0x01: no Unicode for this glyph
    blocks, counts = run(spans, [B("text", "The quick brown fox jumps over the dog.")])
    assert blocks[0].content == "The quick brown fox jumps over the dog." and uk(counts) == {"used": 0, "kept_model": 1}


def test_a_demi_light_font_is_not_bold():
    assert textlayer.font_style("SegoeUI-DemiLight", 350) == (False, False)
    assert textlayer.font_style("ABCD+Futura-DemiBold", 0) == (True, False)


def test_the_page_reports_its_share_of_bold_characters():
    spans = words(80, (BD, "aaaa"), (R, "bbbb"))
    _, counts = run(spans, [B("text", "aaaa bbbb")])
    total = 8 + len("".join(FILLER[0][4].split()))
    assert counts["bold_share"] == round(4 / total, 3)


def test_a_word_space_in_slanted_standard_fonts_is_found():
    # PDFium slants Helvetica-Oblique itself, so the loose box of "f" reaches 0.14 em into the space after it.
    spans = [(72, 80, "Helvetica-BoldOblique", 10, "Cost of Revenues")]
    blocks, counts = run(spans, [B("text", "Cost of Revenues")])
    assert blocks[0].content == "***Cost of Revenues***" and uk(counts) == {"used": 1, "kept_model": 0}


def test_a_bullet_without_a_space_follows_the_model_text():
    spans = [(72, 80, R, 12, "\x95employee training")]
    blocks, counts = run(spans, [B("text", "•employee traininq")])
    assert blocks[0].content == "•employee training" and uk(counts) == {"used": 1, "kept_model": 0}


# --- marks from the layer on the model's characters ---

def test_marks_move_onto_the_model_characters():
    model = "Net income rose 5% in 2024."
    layer = "Net **income** rose 5% in <sup>2024</sup>."
    assert textlayer.transfer_marks(model, layer) == "Net **income** rose 5% in <sup>2024</sup>."


def test_model_characters_win_where_the_layer_differs():
    model = "The cost of revenues fell."
    layer = "The **cost** ofrevenues fell."  # the layer lost a space
    assert textlayer.transfer_marks(model, layer) == "The **cost** of revenues fell."


def test_unmatched_marked_words_are_not_marked():
    assert textlayer.transfer_marks("Alpha beta gamma.", "Alpha *delta* gamma.") == "Alpha beta gamma."


def test_existing_model_marks_are_kept():
    assert textlayer.transfer_marks("**Total** sales", "Total *sales*") == "**Total** *sales*"


def test_marks_nest_and_line_breaks_stay():
    assert textlayer.transfer_marks("a b\nc d", "**a *b***\n*c* d") == "**a *b***\n*c* d"
    assert textlayer.transfer_marks("a\n\nb", "*a b*") == "*a\n\nb*"


def test_an_unclosed_mark_in_the_model_leaves_it_unchanged():
    assert textlayer.transfer_marks("a *b c", "**a** b c") == "a *b c"


def test_marks_only_is_the_default_and_keeps_the_model_spacing(monkeypatch):
    monkeypatch.delenv("PROVIDUS_FIX_TEXT_LAYER_CHARS")
    spans = [(72, 100, R, 10, "The quick brown fox"), (72, 111, "Helvetica-Bold", 10, "jumps over the dog.")]
    model = "The quick brown fox jumps over the dog."
    chars_on = {"text_layer_chars": True}
    (on,), _ = run(spans, [B("text", model)], chars_on)
    (off,), counts = run(spans, [B("text", model)])
    assert on.content == "The quick brown fox **jumps over the dog.**"
    assert off.content == on.content  # same characters here; the layer and the model agree
    assert counts["used"] == 1
    model2 = "The quick  brown fox jumps over the dog."  # a double space is a model character
    (off2,), _ = run(spans, [B("text", model2)])
    assert off2.content == "The quick  brown fox **jumps over the dog.**"


def test_a_star_bullet_list_and_arithmetic_are_text_not_marks():
    bullets = "* item one\n* item two"
    assert textlayer.transfer_marks(bullets, bullets) == bullets
    assert textlayer.transfer_marks(bullets, "* item **one**\n* item two") == "* item **one**\n* item two"
    assert textlayer.transfer_marks("5 * 3 = 15 and 2 * 4", "5 * 3 = 15 and 2 * 4") == "5 * 3 = 15 and 2 * 4"
    assert textlayer.transfer_marks("5 * 3 = 15 and 2 * 4", "**5** * 3 = 15 and 2 * 4") == "**5** * 3 = 15 and 2 * 4"


def test_a_model_text_with_no_layer_marks_stays_byte_identical():
    model = " a  b\n\nc \n"
    assert textlayer.transfer_marks(model, "a b c") == model
    assert textlayer.transfer_marks(model, "a b *zzz*") == model  # a marked word with no match
    assert textlayer.transfer_marks(model, "**a** b c") == " **a**  b\n\nc \n"  # the trailing whitespace stays


def test_marks_only_title_keeps_the_heading_level_and_the_bold_rule(monkeypatch):
    monkeypatch.delenv("PROVIDUS_FIX_TEXT_LAYER_CHARS")
    spans = words(80, (BD, "Results")) + [(72, 300, BD, 12, "Part"), (132, 300, R, 12, "two")]
    blocks, counts = run(spans, [B("title", "Results"), B("title", "Part  two", (0.1, 0.36, 0.9, 0.39))])
    assert [b.content for b in blocks] == ["# Results", "# **Part**  two"]  # all bold: no mark; model spacing stays
    assert counts["used"] == 2


def test_marks_only_with_marks_off_keeps_the_model_text(monkeypatch):
    monkeypatch.delenv("PROVIDUS_FIX_TEXT_LAYER_CHARS")
    spans = words(80, (R, "Normal"), (BD, "bold"))
    blocks, counts = run(spans, [B("text", "Normal boId")], {"text_layer_marks": False})
    assert blocks[0].content == "Normal boId" and counts["used"] == 1
