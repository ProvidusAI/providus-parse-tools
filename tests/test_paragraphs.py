import pytest

from providus_parse.pipeline import ProvidusParser
from providus_parse.textlayer import Char, paragraph_boxes
from test_textlayer import FILLER, OLD_PATH, tiny_pdf


def line(y, x0=0.1, x1=0.9, size=0.012):
    """One layer line of one character. The real Char has also `base` (the baseline y)."""
    return [Char(text="a", x0=x0, y0=y, x1=x1, y1=y + size, base=y + size, size=size, bold=False, italic=False,
                 hidden=False)]


def test_a_wide_gap_starts_a_new_paragraph():
    chars = line(0.10) + line(0.115) + line(0.16) + line(0.175)  # gap 0.033 between lines 2 and 3
    assert len(paragraph_boxes(chars)) == 2


def test_an_indented_first_line_starts_a_new_paragraph():
    chars = line(0.10) + line(0.115) + line(0.13, x0=0.14) + line(0.145)
    assert len(paragraph_boxes(chars)) == 2


def test_even_lines_stay_one_paragraph():
    chars = line(0.10) + line(0.115) + line(0.13)
    assert paragraph_boxes(chars) == [pytest.approx((0.1, 0.10, 0.9, 0.142))]


def test_a_hanging_indent_list_without_gaps_stays_one_paragraph():
    ys = [0.10 + 0.015 * i for i in range(9)]
    chars = [c for k, y in enumerate(ys) for c in line(y, x0=0.1 if k % 3 == 0 else 0.14)]  # 3 bullets x 3 lines
    assert len(paragraph_boxes(chars)) == 1


def test_a_hanging_indent_list_with_gaps_gives_one_paragraph_per_item():
    ys = [0.10 + 0.015 * (k % 3) + 0.06 * (k // 3) for k in range(9)]
    chars = [c for k, y in enumerate(ys) for c in line(y, x0=0.1 if k % 3 == 0 else 0.14)]
    assert len(paragraph_boxes(chars)) == 3


def test_a_reference_entry_with_a_hanging_indent_is_one_paragraph():
    chars = line(0.10) + line(0.115, x0=0.14) + line(0.13, x0=0.14) + line(0.145, x0=0.14)
    assert len(paragraph_boxes(chars)) == 1


def test_centred_text_is_one_paragraph():
    chars = line(0.10) + line(0.115, 0.25, 0.75) + line(0.13, 0.3, 0.7)
    assert len(paragraph_boxes(chars)) == 1


def test_a_block_quote_indented_as_a_whole_is_one_paragraph():
    chars = line(0.10, x0=0.14) + line(0.115, x0=0.14) + line(0.13, x0=0.14)
    assert len(paragraph_boxes(chars)) == 1


def test_an_indented_last_line_after_a_left_edge_line_starts_a_paragraph():
    assert len(paragraph_boxes(line(0.10) + line(0.115) + line(0.13, x0=0.14))) == 2


def test_no_chars_give_no_paragraph():
    assert paragraph_boxes([]) == []


BOX = (0.05, 0.07, 0.95, 0.2)
TEXT = "First paragraph line one\nfirst paragraph line two\nSecond paragraph line one\nsecond paragraph line two"
SPANS = [(72, 80, "Helvetica", 10, "First paragraph line one"), (72, 92, "Helvetica", 10, "first paragraph line two"),
         (72, 130, "Helvetica", 10, "Second paragraph line one"), (72, 142, "Helvetica", 10, "second paragraph line two")]


class _C:
    def __init__(self, text):
        self.text = text

    def two_step_extract(self, image):
        from types import SimpleNamespace
        return [SimpleNamespace(type="text", bbox=BOX, angle=0, content=self.text)]


def parse(text, flags=None, spans=SPANS):
    parser = ProvidusParser("u", fixes={**OLD_PATH, "para_split": True, **(flags or {})}, client=_C(text))
    return parser.parse_page(None, pdf=tiny_pdf(spans + FILLER))


def test_a_text_block_with_two_paragraphs_gives_two_items_and_the_same_markdown():
    r = parse(TEXT)
    off = parse(TEXT, {"para_split": False})
    assert [i.text for i in r.items] == [
        "First paragraph line one\nfirst paragraph line two", "Second paragraph line one\nsecond paragraph line two"]
    assert [i.category for i in r.items] == ["Text", "Text"]
    assert r.items[0].bbox[3] < r.items[1].bbox[1]  # the boxes are 0 to 1 and do not overlap
    assert all(0 <= v <= 1 for i in r.items for v in i.bbox)
    assert len(off.items) == 1 and off.items[0].bbox == BOX
    assert r.markdown == off.markdown


def test_the_text_splits_by_character_share_when_the_line_counts_differ():
    r = parse(TEXT.replace("\n", " "))
    assert [i.text for i in r.items] == [
        "First paragraph line one first paragraph line two", "Second paragraph line one second paragraph line two"]


def test_one_paragraph_gives_one_item():
    r = parse(TEXT, spans=[(x, y if y < 100 else y - 26, f, s, t) for x, y, f, s, t in SPANS])
    assert len(r.items) == 1 and r.items[0].bbox == BOX


def test_a_block_that_keeps_the_model_text_gives_one_item():
    assert len(parse("Totally different words about something else entirely").items) == 1


def test_a_failure_keeps_one_item(monkeypatch):
    from providus_parse import textlayer
    monkeypatch.setattr(textlayer, "_para_info", lambda *a: 1 / 0)
    assert len(parse(TEXT).items) == 1


def test_a_split_item_has_no_unbalanced_mark():
    from providus_parse.pipeline import _close_marks, _items, Block
    assert _close_marks("a **b c") == "a b c"
    assert _close_marks("a *b c") == "a b c"
    assert _close_marks("a <sup>2") == "a 2"
    assert _close_marks("**a** *b* <sup>2</sup>") == "**a** *b* <sup>2</sup>"
    box = (0.1, 0.1, 0.9, 0.2)
    b = Block("text", box, 0, "aaaa **bbbb cccc** dddd", paras=((box, 1, 10), (box, 1, 10)))
    assert [i.text for i in _items(b)] == ["aaaa bbbb", "cccc dddd"]


def test_para_split_is_off_by_default():
    parser = ProvidusParser("u", fixes=OLD_PATH, client=_C(TEXT))
    assert len(parser.parse_page(None, pdf=tiny_pdf(SPANS + FILLER)).items) == 1
