import pytest

from providus_parse import fixes
from providus_parse.markdown import blocks_to_markdown
from providus_parse.pipeline import Block, ProvidusParser


def B(t, c, bbox=(0.1, 0.1, 0.9, 0.2)):
    return Block(type=t, bbox=bbox, angle=0, content=c)


# --- flags ---

def test_flag_default_on(monkeypatch):
    monkeypatch.delenv("PROVIDUS_FIX_HEADER_FOOTER", raising=False)
    assert fixes.enabled("header_footer") is True


@pytest.mark.parametrize("value", ["0", "false", "off", "no", "FALSE"])
def test_env_turns_a_flag_off(monkeypatch, value):
    monkeypatch.setenv("PROVIDUS_FIX_HEADER_FOOTER", value)
    assert fixes.enabled("header_footer") is False


def test_explicit_flag_beats_env(monkeypatch):
    monkeypatch.setenv("PROVIDUS_FIX_HEADER_FOOTER", "0")
    assert fixes.enabled("header_footer", {"header_footer": True}) is True
    assert fixes.enabled("table_header", {"table_header": False}) is False


# --- header_footer ---

def test_header_footer_page_number_in_markdown_in_reading_order():
    blocks = [B("header", "ACME 2024"), B("text", "Body."), B("footer", "Confidential"), B("page_number", "7")]
    assert blocks_to_markdown(blocks) == "ACME 2024\n\nBody.\n\nConfidential\n\n7"


def test_markdown_can_leave_paratext_out():
    assert blocks_to_markdown([B("header", "ACME"), B("text", "Body.")], paratext=False) == "Body."


class _C:
    def __init__(self, blocks):
        self.blocks = blocks

    def two_step_extract(self, image):
        return self.blocks


def _blk(t, c):
    from types import SimpleNamespace
    return SimpleNamespace(type=t, bbox=(0.1, 0.1, 0.9, 0.2), angle=0, content=c)


@pytest.mark.parametrize("flag,expected", [(True, "ACME\n\nBody."), (False, "Body.")])
def test_parser_header_footer_flag(flag, expected):
    p = ProvidusParser("u", fixes={"header_footer": flag, "image_block": False, "text_request_form": False,
                              "chart_request_form": False},
                       client=_C([_blk("header", "ACME"), _blk("text", "Body.")]))
    r = p.parse_page(None)
    assert r.markdown == expected
    assert [i.category for i in r.items] == ["Page-header", "Text"]  # items keep the header either way
