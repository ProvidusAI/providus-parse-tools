import pytest

from providus_parse import fixes
from providus_parse.fixes import add_table_header
from providus_parse.pipeline import Block


def T(*rows):
    return "<table>" + "".join(f"<tr>{r}</tr>" for r in rows) + "</table>"


def test_rowspan_in_first_row_makes_header_rows():
    html = T('<td rowspan="2">Item</td><td>2023</td>', "<td>Q1</td>", "<td>Rent</td><td>5</td>")
    assert add_table_header(html) == (
        "<table><thead><tr><th rowspan=\"2\">Item</th><th>2023</th></tr><tr><th>Q1</th></tr></thead>"
        "<tr><td>Rent</td><td>5</td></tr></table>")


def test_colspan_in_last_header_row_adds_one_row():
    html = T('<td rowspan="2">Item</td><td colspan="2">2023</td>', '<td colspan="2">Sales</td>',
             "<td>Q1</td><td>Q2</td>", "<td>Rent</td><td>5</td><td>6</td>")
    out = add_table_header(html)
    assert "<thead><tr><th rowspan=\"2\">Item</th><th colspan=\"2\">2023</th></tr><tr><th colspan=\"2\">Sales</th></tr><tr><th>Q1</th><th>Q2</th></tr></thead>" in out
    assert out.endswith("<tr><td>Rent</td><td>5</td><td>6</td></tr></table>")


def test_first_row_without_rowspan_is_one_header_row():
    html = T("<td>A</td><td>B</td>", "<td>1</td><td>2</td>", "<td>3</td><td>4</td>")
    assert add_table_header(html) == (
        "<table><thead><tr><th>A</th><th>B</th></tr></thead><tr><td>1</td><td>2</td></tr><tr><td>3</td><td>4</td></tr></table>")


def test_colspan_in_a_first_row_without_rowspan_adds_one_row():
    html = T('<td colspan="2">2023</td>', "<td>Q1</td><td>Q2</td>", "<td>1</td><td>2</td>")
    assert add_table_header(html).startswith(
        '<table><thead><tr><th colspan="2">2023</th></tr><tr><th>Q1</th><th>Q2</th></tr></thead>')


def test_without_the_first_row_flag_a_table_needs_a_rowspan_in_its_first_row():
    plain = T("<td>A</td><td>B</td>", "<td>1</td><td>2</td>", "<td>3</td><td>4</td>")
    assert add_table_header(plain, first_row=False) == plain
    spanned = T('<td rowspan="2">A</td>', "<td>B</td>", "<td>1</td>")
    assert "<thead>" in add_table_header(spanned, first_row=False)


def test_table_with_th_or_thead_is_not_changed():
    for html in (T('<th rowspan="2">A</th><td>B</td>', "<td>1</td>", "<td>x</td><td>y</td>"),
                 "<table><thead><tr><td rowspan=\"2\">A</td></tr></thead><tr><td>1</td></tr></table>"):
        assert add_table_header(html) == html


def head_rows(html):
    out = add_table_header(html)
    return out.split("</thead>")[0].count("<tr") if "<thead>" in out else 0


def test_two_row_table_with_a_first_row_colspan_has_one_header_row():
    html = T('<td colspan="2">Title</td>', "<td>1</td><td>2</td>")
    assert add_table_header(html) == (
        '<table><thead><tr><th colspan="2">Title</th></tr></thead><tr><td>1</td><td>2</td></tr></table>')


def test_rowspan_over_every_row_clamps_to_leave_one_body_row():
    two = T('<td rowspan="2">A</td>', "<td>B</td>")
    assert add_table_header(two) == '<table><thead><tr><th rowspan="2">A</th></tr></thead><tr><td>B</td></tr></table>'
    three = T('<td rowspan="3">A</td>', "<td>B</td>", "<td>C</td>")
    assert head_rows(three) == 2


def test_colspan_row_is_not_added_when_it_would_use_the_last_row():
    html = T('<td rowspan="2">A</td>', '<td colspan="2">B</td>', "<td>1</td>")
    assert head_rows(html) == 2
    assert add_table_header(html).endswith("</thead><tr><td>1</td></tr></table>")


def test_rowspan_closure_follows_rowspans_in_later_header_rows():
    html = T('<td rowspan="2">A</td>', '<td rowspan="2">B</td>', "<td>C</td>", "<td>1</td>")  # rows 0-1, then row 1 reaches row 2
    assert head_rows(html) == 3


def test_a_single_row_table_has_no_header():
    html = T("<td>A</td><td>B</td>")
    assert add_table_header(html) == html


def test_rowspan_below_one_counts_as_one():
    html = T('<td rowspan="0">A</td>', "<td>1</td>", "<td>2</td>")
    assert head_rows(html) == 1


def test_nested_or_tbody_tables_are_not_changed():
    nested = "<table><tr><td>" + T("<td>x</td>", "<td>y</td>") + "</td></tr><tr><td>b</td></tr></table>"
    assert add_table_header(nested) == nested
    tbody = "<table><tbody><tr><td>A</td></tr><tr><td>1</td></tr></tbody></table>"
    assert add_table_header(tbody) == tbody


def test_cell_attributes_and_text_survive():
    html = T("<td class='x' rowspan=\"2\">A &amp; B</td><td>C</td>", "<td>D</td>", "<td>1</td><td>2</td>")
    assert "<th class='x' rowspan=\"2\">A &amp; B</th>" in add_table_header(html)


def test_text_that_is_not_a_table_is_not_changed():
    assert add_table_header("plain") == "plain"
    assert add_table_header("") == ""


def test_apply_changes_table_blocks_only_and_obeys_the_flag(monkeypatch):
    html = T('<td rowspan="2">A</td><td>B</td>', "<td>C</td>", "<td>1</td><td>2</td>")
    blocks = [Block("text", (0, 0, 1, 1), 0, html), Block("table", (0, 0, 1, 1), 0, html)]
    out = fixes.apply(blocks, {})
    assert out[0].content == html and "<thead>" in out[1].content
    assert fixes.apply(blocks, {"table_header": False})[1].content == html
    plain = T("<td>A</td>", "<td>1</td>", "<td>2</td>")
    one = [Block("table", (0, 0, 1, 1), 0, plain)]
    assert "<thead>" in fixes.apply(one, {})[0].content
    assert fixes.apply(one, {"table_header_first_row": False})[0].content == plain
    monkeypatch.setenv("PROVIDUS_FIX_TABLE_HEADER", "0")
    assert fixes.apply(blocks, {})[1].content == html


def test_rowspan_larger_than_the_table_gives_no_error():
    html = T('<td rowspan="99">A</td>', "<td>B</td>", "<td>C</td>")
    assert head_rows(html) == 2


def test_a_fix_that_raises_leaves_the_block_unchanged(monkeypatch):
    html = T('<td rowspan="2">A</td>', "<td>B</td>", "<td>1</td>")

    def boom(*a, **k):
        raise RuntimeError("bad table")

    monkeypatch.setattr(fixes, "add_table_header", boom)
    blocks = [Block("table", (0, 0, 1, 1), 0, html), Block("text", (0, 0, 1, 1), 0, "t")]
    assert fixes.apply(blocks, {}) == blocks
