"""The defaults are the settings of the ParseBench runs behind the providus-vl-mini scores: no environment variable is needed.

The three flags that are off by default are tested by what they do: heading_class in test_heading_class.py
(test_heading_class_is_off_by_default_...), para_split in test_paragraphs.py (test_para_split_is_off_by_default),
text_layer_chars here."""

import os

import pytest
from test_textlayer import FILLER, R, B, tiny_pdf

from providus_parse import fixes, textlayer

ON = ["header_footer", "table_header", "table_header_first_row", "image_block", "text_request_form", "text_prompt_newline",
      "text_resize", "text_no_penalty", "chart_request_form", "chart_resize", "chart_no_penalty", "chart_caption",
      "text_layer", "text_layer_marks", "text_layer_headings"]


@pytest.fixture(autouse=True)
def _no_flags(monkeypatch):
    for k in [k for k in os.environ if k.startswith("PROVIDUS_FIX_")]:
        monkeypatch.delenv(k)


def test_the_flags_that_are_on_by_default():
    assert all(fixes.enabled(n) for n in ON)


def test_text_layer_chars_is_off_by_default():
    spans = [(72, 100, R, 10, "The quick brown fox jumps over the dog.")]
    model = "The quick brown fox  jumps over the dog."  # two spaces: the layer has one
    pdf = tiny_pdf(spans + FILLER)
    (default,), _ = textlayer.apply([B("text", model)], pdf, None)
    (chars_on,), _ = textlayer.apply([B("text", model)], pdf, {"text_layer_chars": True})
    assert "fox  jumps" in default.content  # the model's characters stay
    assert "fox  jumps" not in chars_on.content
