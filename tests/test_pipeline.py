import base64
import io
import json
from types import SimpleNamespace

from PIL import Image

from providus_parse.pipeline import ProvidusParser
from providus_parse.server import predict


OLD_PATH = {"image_block": False, "text_request_form": False, "chart_request_form": False}  # FakeClient has two_step_extract only


def blk(t, c, bbox=(0.1, 0.1, 0.9, 0.2), angle=0):
    return SimpleNamespace(type=t, bbox=bbox, angle=angle, content=c)


class FakeClient:
    def two_step_extract(self, image):
        return [blk("title", "Report"), blk("text", "Body.", bbox=(0.1, 0.3, 0.9, 0.4), angle=None)]


def png_body(size=(40, 20)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, "white").save(buf, "PNG")
    return json.dumps({"image_base64": base64.b64encode(buf.getvalue()).decode()}).encode()


def test_parse_page_gives_markdown_and_items():
    r = ProvidusParser("http://unused", fixes=OLD_PATH, client=FakeClient()).parse_page(None)
    assert r.markdown == "# Report\n\nBody."
    assert [(i.category, i.text) for i in r.items] == [("Title", "Report"), ("Text", "Body.")]
    assert r.items[1].bbox == (0.1, 0.3, 0.9, 0.4)


def test_unknown_type_falls_back_to_text():
    class C:
        def two_step_extract(self, image):
            return [blk("mystery", "x")]

    assert ProvidusParser("u", fixes=OLD_PATH, client=C()).parse_page(None).items[0].category == "Text"


def test_image_block_wrapper_gives_no_item():
    class C:
        def two_step_extract(self, image):
            return [blk("image_block", "", bbox=(0.1, 0.1, 0.9, 0.8)), blk("image", "", bbox=(0.1, 0.1, 0.5, 0.8)),
                    blk("image", "", bbox=(0.5, 0.1, 0.9, 0.8))]

    r = ProvidusParser("u", fixes=OLD_PATH, client=C()).parse_page(None)
    assert [i.category for i in r.items] == ["Picture", "Picture"]


def test_predict_answers_with_markdown_items_and_size():
    out = predict(png_body(), ProvidusParser("u", fixes=OLD_PATH, client=FakeClient()))
    assert out["status"] == "success"
    assert out["markdown"] == "# Report\n\nBody."
    assert out["items"][0] == {"category": "Title", "bbox": [0.1, 0.1, 0.9, 0.2], "text": "Report", "type": "title"}
    assert (out["image_width"], out["image_height"]) == (40, 20)


def test_predict_bad_body_answers_error():
    out = predict(b"not json", ProvidusParser("u", fixes=OLD_PATH, client=FakeClient()))
    assert out["status"] == "error"
    assert out["error"]


def test_items_keep_the_block_type_so_a_page_number_differs_from_a_footer():
    class C:
        def two_step_extract(self, image):
            return [blk("footer", "ACME"), blk("page_number", "7")]

    items = ProvidusParser("u", fixes=OLD_PATH, client=C()).parse_page(None).items
    assert [(i.category, i.type) for i in items] == [("Page-footer", "footer"), ("Page-footer", "page_number")]


def test_item_text_has_no_heading_marker_but_the_markdown_keeps_it():
    class C:
        def two_step_extract(self, image):
            return [blk("title", "## Results"), blk("text", "# is a hash sign.")]

    r = ProvidusParser("u", fixes=OLD_PATH, client=C()).parse_page(None)
    assert [i.text for i in r.items] == ["Results", "# is a hash sign."]
    assert r.markdown == "## Results\n\n# is a hash sign."


def _one_block(t, c):
    class C:
        def two_step_extract(self, image):
            return [blk(t, c)]

    return ProvidusParser("u", fixes=OLD_PATH, client=C()).parse_page(None)


def test_chart_html_table_gives_plain_cell_text_in_the_item_only():
    html_table = "<table><caption>Revenue by year</caption><tr><th>Year</th><th>Rev</th></tr><tr><td>2024</td><td>5 &amp; 6</td></tr></table>"
    r = _one_block("chart", html_table)
    assert r.items[0].text == "Year Rev 2024 5 & 6"
    assert "<caption>Revenue by year</caption>" in r.markdown


def test_chart_pipe_table_item_text_does_not_change():
    r = _one_block("chart", "| a | b |\n| - | - |\n| 1 | 2 |")
    assert r.items[0].text == "| a | b |\n| - | - |\n| 1 | 2 |"


def test_text_block_with_a_table_tag_does_not_change():
    assert _one_block("text", "<table><tr><td>x</td></tr></table>").items[0].text == "<table><tr><td>x</td></tr></table>"
