from providus_parse import fixes
from providus_parse.fixes import add_table_caption, caption_for_chart, md_table_to_html
from providus_parse.pipeline import Block

MD = "| Category | LDC |\n| --- | --- |\n| IF | 0.6875 |\n| CP | 0.4444 |"


def B(t, c):
    return Block(type=t, bbox=(0, 0, 1, 1), angle=0, content=c)


def test_caption_before_the_chart_is_used():
    blocks = [B("text", "Body."), B("image_caption", "Realised hourly rate the past 5 years*"),
              B("chart", "<table><tr><td>2024</td><td>5</td></tr></table>")]
    assert caption_for_chart(blocks, 2) == "Realised hourly rate the past 5 years*"


def test_caption_after_the_chart_is_used_when_none_before():
    blocks = [B("chart", "<table></table>"), B("image_caption", "Figure 3: Sales by region")]
    assert caption_for_chart(blocks, 0) == "Figure 3: Sales by region"


def test_body_text_is_not_a_caption():
    blocks = [B("text", "This paragraph explains the method in several long sentences that are not a title."),
              B("chart", "<table></table>")]
    assert caption_for_chart(blocks, 1) is None


def test_a_page_number_is_not_a_caption():
    assert caption_for_chart([B("text", "76"), B("chart", MD)], 1) is None


def test_second_chart_in_a_row_finds_the_caption_two_blocks_back():
    blocks = [B("image_caption", "Figure 2.33 Averages"), B("chart", MD), B("chart", MD)]
    assert caption_for_chart(blocks, 2) == "Figure 2.33 Averages"


def test_add_table_caption_inserts_once_and_escapes():
    html = add_table_caption("<table><tr><td>1</td></tr></table>", "A & B")
    assert html.startswith("<table><caption>A &amp; B</caption><tr>")
    assert add_table_caption(html, "A & B") == html


def test_markdown_chart_table_becomes_html_with_a_header_row():
    assert md_table_to_html(MD) == (
        "<table><thead><tr><th>Category</th><th>LDC</th></tr></thead>"
        "<tr><td>IF</td><td>0.6875</td></tr><tr><td>CP</td><td>0.4444</td></tr></table>")


def test_markdown_conversion_escapes_cells_and_keeps_br():
    assert "<td>a &amp; b<br>c</td>" in md_table_to_html("| H |\n| --- |\n| a & b<br>c |")


def test_not_a_markdown_table_gives_none():
    assert md_table_to_html("Just a line.") is None


def test_apply_puts_the_caption_into_a_markdown_chart():
    blocks = [B("image_caption", "Figure 2.33 Averages"), B("chart", MD)]
    out = fixes.apply(blocks, {})
    assert out[0].content == "Figure 2.33 Averages"  # the caption block stays
    assert out[1].content.startswith("<table><caption>Figure 2.33 Averages</caption><thead>")


def test_apply_leaves_a_chart_without_caption_and_a_flag_off_chart():
    assert fixes.apply([B("chart", MD)], {})[0].content == MD
    blocks = [B("image_caption", "Figure 1"), B("chart", MD)]
    assert fixes.apply(blocks, {"chart_caption": False})[1].content == MD


def test_a_failing_chart_block_stays_unchanged(monkeypatch):
    def boom(*a):
        raise ValueError("x")
    monkeypatch.setattr(fixes, "add_table_caption", boom)
    blocks = [B("image_caption", "Figure 1"), B("chart", MD)]
    assert fixes.apply(blocks, {})[1].content == MD


def test_bold_marks_are_removed_but_a_footnote_star_stays():
    assert caption_for_chart([B("title", "**Sales by region**"), B("chart", MD)], 1) == "Sales by region"
