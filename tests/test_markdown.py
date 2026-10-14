from providus_parse.markdown import CATEGORY, WRAPPER_TYPES, blocks_to_markdown
from providus_parse.pipeline import Block


def B(t, c, bbox=(0.1, 0.1, 0.9, 0.2)):
    return Block(type=t, bbox=bbox, angle=0, content=c)


def test_markdown_orders_and_formats_blocks():
    md = blocks_to_markdown([B("title", "Annual Report"), B("text", "Revenue rose."),
                             B("table", "<table><tr><td>1</td></tr></table>")])
    assert md == "# Annual Report\n\nRevenue rose.\n\n<table><tr><td>1</td></tr></table>"


def test_equation_is_display_math():
    assert blocks_to_markdown([B("equation", "E=mc^2")]) == "$$\nE=mc^2\n$$"


def test_every_mineru_type_has_a_layout_category():
    for t in ("text", "title", "section_header", "table", "image", "chart", "equation", "header", "footer", "page_number",
              "page_footnote", "image_caption", "table_caption", "list", "code",
              "image_footnote", "table_footnote", "aside_text"):
        assert t in CATEGORY
    assert "image_block" in WRAPPER_TYPES
    assert "image_block" not in CATEGORY


def test_a_section_header_without_a_marker_gets_two_hashes():
    assert blocks_to_markdown([B("section_header", "Overview"), B("section_header", "### Scope")]) == "## Overview\n\n### Scope"


def test_header_footer_not_in_body_markdown_when_flag_is_off():
    assert blocks_to_markdown([B("header", "ACME 2024"), B("text", "Body.")], paratext=False) == "Body."
