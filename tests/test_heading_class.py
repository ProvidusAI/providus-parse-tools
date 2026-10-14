"""A short bold or large single line of a text block becomes a section header."""

import pytest
from types import SimpleNamespace

from PIL import Image
from test_textlayer import BD, H, OLD_PATH, R, tiny_pdf

from providus_parse import textlayer
from providus_parse.pipeline import Block, ProvidusParser
from providus_parse.textlayer import looks_like_heading

BODY_LINES = [(72, 140 + 14 * i, R, 10, "The company reported higher sales in the year and it paid a small dividend") for i in range(4)]


def box(y, top=14, bottom=4, x1=0.9):
    return (0.1, (y - top) / H, x1, (y + bottom) / H)


def B(t, content, bbox):
    return Block(type=t, bbox=bbox, angle=0, content=content)


def run(spans, blocks, flags=None):
    """heading_class is off by default, so these tests turn it on unless they say otherwise."""
    return textlayer.apply(blocks, tiny_pdf(spans + BODY_LINES), {"heading_class": True} if flags is None else flags)[0]


def test_heading_class_is_off_by_default_and_on_with_the_environment_variable(monkeypatch):
    monkeypatch.delenv("PROVIDUS_FIX_HEADING_CLASS", raising=False)
    args = ([B("text", "Risk Factors", box(80))], tiny_pdf([(72, 80, BD, 10, "Risk Factors")] + BODY_LINES), None)
    assert textlayer.apply(*args)[0][0].type == "text"
    monkeypatch.setenv("PROVIDUS_FIX_HEADING_CLASS", "1")
    assert textlayer.apply(*args)[0][0].type == "section_header"


def test_a_short_all_bold_line_is_a_heading(make_chars):
    assert looks_like_heading("Risk Factors", make_chars("Risk Factors", bold=True, size=0.012), 0.012)


def test_a_short_large_line_is_a_heading(make_chars):
    assert looks_like_heading("Overview", make_chars("Overview", size=0.016), 0.012)


def test_a_sentence_is_not_a_heading(make_chars):
    text = "The company reported higher sales."
    assert not looks_like_heading(text, make_chars(text, bold=True, size=0.012), 0.012)


def test_a_long_bold_line_is_not_a_heading(make_chars):
    text = "This bold line has far too many words to be a section heading in any document"
    assert not looks_like_heading(text, make_chars(text, bold=True, size=0.012), 0.012)


def test_a_normal_size_regular_line_is_not_a_heading(make_chars):
    assert not looks_like_heading("Overview", make_chars("Overview", size=0.012), 0.012)


def test_a_colon_followed_by_text_is_not_a_heading(make_chars):
    assert not looks_like_heading("Note: see below", make_chars("Note: see below", bold=True), 12)
    assert looks_like_heading("Key terms:", make_chars("Key terms:", bold=True), 12)


def test_a_line_with_a_regular_letter_is_not_all_bold(make_chars):
    chars = make_chars("Risk", bold=True) + make_chars("Factors", bold=False)
    assert not looks_like_heading("Risk Factors", chars, 12)


def test_two_lines_are_not_a_heading(make_chars):
    chars = make_chars("Risk", bold=True, y=100) + make_chars("Factors", bold=True, y=130)
    assert not looks_like_heading("Risk\nFactors", chars, 12)


def test_a_bold_text_block_becomes_a_section_header_with_two_hashes():
    out = run([(72, 80, BD, 10, "Risk Factors")], [B("text", "Risk Factors", box(80))])
    assert (out[0].type, out[0].content) == ("section_header", "## Risk Factors")


def test_a_large_text_block_becomes_a_section_header():
    out = run([(72, 80, R, 14, "Overview")], [B("text", "Overview", box(80, 16))])
    assert (out[0].type, out[0].content) == ("section_header", "## Overview")


def test_a_section_header_under_a_larger_title_gets_two_hashes():
    spans = [(72, 50, BD, 24, "Annual Report"), (72, 100, BD, 15, "Overview")]
    blocks = [B("title", "Annual Report", box(50, 26)), B("text", "Overview", box(100, 17))]
    assert [b.content for b in run(spans, blocks)] == ["# Annual Report", "## Overview"]


def test_a_smaller_section_header_gets_the_next_level():
    spans = [(72, 50, BD, 24, "Annual Report"), (72, 100, BD, 16, "Overview"), (72, 120, BD, 13, "Scope")]
    blocks = [B("title", "Annual Report", box(50, 26)), B("text", "Overview", box(100, 18)),
              B("text", "Scope", box(120, 15, 2))]
    assert [b.content for b in run(spans, blocks)] == ["# Annual Report", "## Overview", "### Scope"]


def test_a_body_paragraph_stays_text():
    out = run([], [B("text", "The company reported higher sales in the year and it paid a small dividend", box(140))])
    assert out[0].type == "text"


def test_only_text_blocks_change():
    spans = [(72, 80, BD, 10, "Risk Factors")]
    for t in ("list", "image_caption", "table_caption", "header", "footer", "page_footnote"):
        assert run(spans, [B(t, "Risk Factors", box(80))])[0].type == t


def test_a_line_inside_a_table_stays_text():
    spans = [(72, 80, BD, 10, "Total")]
    blocks = [B("table", "<table><tr><td>Total</td></tr></table>", (0.05, 0.05, 0.95, 0.3)),
              B("text", "Total", box(80))]
    assert run(spans, blocks)[1].type == "text"


def test_a_line_with_math_stays_text():
    out = run([(72, 80, BD, 10, "Energy")], [B("text", r"Energy $E=mc^2$", box(80))])
    assert out[0].type == "text"


def test_a_page_of_bold_text_gets_no_bold_headings():
    spans = [(72, 80 + 14 * i, BD, 10, "Bold words fill this page from top to bottom of the page") for i in range(8)]
    out = textlayer.apply([B("text", "Bold words", box(80))], tiny_pdf(spans), None)[0]
    assert out[0].type == "text"


def test_flag_off_keeps_the_text_block():
    out = run([(72, 80, BD, 10, "Risk Factors")], [B("text", "Risk Factors", box(80))], {"heading_class": False})
    assert out[0].type == "text"


def test_a_failure_keeps_the_block(monkeypatch):
    monkeypatch.setattr(textlayer, "looks_like_heading", lambda *a, **k: 1 / 0)
    out = run([(72, 80, BD, 10, "Risk Factors")], [B("text", "Risk Factors", box(80))])
    assert out[0].type == "text"


def test_the_parser_writes_a_heading_and_a_section_header_item():
    class Client:
        def two_step_extract(self, image):
            return [SimpleNamespace(type="text", bbox=(0.1, 0.09, 0.9, 0.11), angle=0, content="Risk Factors")]

    spans = [(72, 80, BD, 10, "Risk Factors")]
    page = ProvidusParser("x", fixes={**OLD_PATH, "heading_class": True}, client=Client()).parse_page(Image.new("RGB", (100, 100)), tiny_pdf(spans + BODY_LINES))
    assert page.markdown == "## Risk Factors"
    assert (page.items[0].category, page.items[0].text) == ("Section-header", "Risk Factors")


@pytest.mark.parametrize("text", ["Total $5,000", "Page 3 of 10", "Q1 Q2 Q3 Q4", "12 March 2024", "(a)", "INVOICE #1234"])
def test_a_line_of_numbers_or_symbols_is_not_a_heading(make_chars, text):
    assert not looks_like_heading(text, make_chars(text, bold=True), 12)


@pytest.mark.parametrize("text", ["Risk Factors", "1. Introduction", "2.3 Results of Operations", "Part II Financial Data"])
def test_a_bold_line_with_a_number_prefix_is_a_heading(make_chars, text):
    assert looks_like_heading(text, make_chars(text, bold=True), 12)


@pytest.mark.parametrize("text", ["Name:", "Note", "Total", "Yes", "Signature"])
def test_a_single_bold_word_is_not_a_heading(make_chars, text):
    assert not looks_like_heading(text, make_chars(text, bold=True), 12)


def test_a_single_large_word_is_a_heading(make_chars):
    assert looks_like_heading("Overview", make_chars("Overview", size=15), 12)
    assert looks_like_heading("Summary:", make_chars("Summary:", size=15), 12)


def test_an_all_bold_header_loses_every_bold_mark():
    out = run([(72, 80, BD, 10, "Risk Factors")], [B("text", "**Risk** **Factors**", box(80))])
    assert out[0].content == "## Risk Factors"


def test_a_partly_bold_header_keeps_the_mark():
    out = run([(72, 80, R, 14, "Part "), (112, 80, BD, 14, "One")], [B("text", "Part **One**", box(80, 16))])
    assert out[0].content == "## Part **One**"


def test_an_all_italic_header_loses_the_italic_marks():
    out = run([(72, 80, "Times-Italic", 14, "Risk Factors")], [B("text", "*Risk Factors*", box(80, 16))])
    assert out[0].content == "## Risk Factors"


def test_without_titles_the_section_header_sizes_start_at_two_hashes():
    spans = [(72, 60, BD, 16, "Overview Part"), (72, 85, BD, 13, "Scope Part"), (72, 110, BD, 11.5, "Detail Part")]
    blocks = [B("text", "Overview Part", box(60, 18)), B("text", "Scope Part", box(85, 15, 2)),
              B("text", "Detail Part", box(110, 13, 2))]
    assert [b.content for b in run(spans, blocks)] == ["## Overview Part", "### Scope Part", "### Detail Part"]
