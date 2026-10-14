import io
from types import SimpleNamespace

import pytest
from PIL import Image

from providus_parse import extract
from providus_parse.pipeline import ProvidusParser


def blk(t, bbox=(0.1, 0.1, 0.9, 0.2), content=None, angle=0):
    return SimpleNamespace(type=t, bbox=bbox, angle=angle, content=content)


def png(size) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, "white").save(buf, "PNG")
    return buf.getvalue()


SP = SimpleNamespace  # stands for mineru-vl-utils SamplingParams in these tests


class FakeClient:
    """Mirrors the calls of MinerUClient that providus_parse uses: layout_detect, helper, client."""

    def __init__(self, layout, crop_size=(100, 40)):
        self.layout, self.crop_size = layout, crop_size
        self.seen = {}
        outer = self

        class Helper:
            def prepare_for_extract(self, image, blocks, not_extract_list=None, image_analysis=None):
                outer.seen["layout"] = list(blocks)
                idx = [i for i, b in enumerate(blocks) if b.type not in {"image_block", "list"}]
                prompts = [{"table": "\nTable Recognition:", "chart": "\nImage Analysis:", "image": "\nImage Analysis:"}
                           .get(blocks[i].type, "\nText Recognition:") for i in idx]
                params = [SP(presence_penalty=1.0, frequency_penalty=0.05, no_repeat_ngram_size=100) for _ in idx]
                return [png(outer.crop_size) for _ in idx], prompts, params, idx

            def post_process(self, blocks):
                return blocks

        class Vlm:
            def batch_predict(self, images, prompts, params, priority=None):
                outer.seen.update(images=images, prompts=prompts, params=params)
                return [f"out{i}" for i in range(len(images))]

        self.helper, self.client = Helper(), Vlm()

    def layout_detect(self, image):
        return list(self.layout)

    def two_step_extract(self, image):
        self.seen["two_step"] = True
        return [blk("text", content="old")]


def only(*names):
    """Flags: everything off except the named ones."""
    all_names = ("header_footer", "table_header", "image_block", "text_request_form",
                 "text_prompt_newline", "text_resize", "text_no_penalty", "chart_request_form", "chart_resize",
                 "chart_no_penalty", "text_layer")
    return {n: n in names for n in all_names}


# --- image_block ---

def test_image_block_wrapper_is_removed_before_extraction_so_children_get_content():
    c = FakeClient([blk("image_block", (0.1, 0.1, 0.9, 0.8)), blk("chart", (0.1, 0.1, 0.5, 0.8)),
                    blk("image", (0.5, 0.1, 0.9, 0.8))])
    r = ProvidusParser("u", fixes=only("image_block"), client=c).parse_page(None)
    assert [b.type for b in c.seen["layout"]] == ["chart", "image"]
    assert [(i.category, i.text) for i in r.items] == [("Chart", "out0"), ("Picture", "out1")]


def test_image_block_flag_off_uses_the_client_two_step_path():
    c = FakeClient([blk("image_block")])
    r = ProvidusParser("u", fixes=only(), client=c).parse_page(None)
    assert c.seen.get("two_step") and r.markdown == "old"


# --- text_request_form ---

def test_text_resize_sizes():
    # Multiple of 28, at least 294,768 pixels.
    for size in [(300, 40), (100, 100), (28, 28), (2000, 30), (3000, 3000)]:
        w, h = extract.text_dimensions(*size)
        assert w % 28 == 0 and h % 28 == 0
        assert 294_768 <= w * h <= 2_822_400
    # Pinned sizes.
    pins = {(300, 40): (1512, 224), (100, 100): (560, 560), (2000, 30): (4452, 84), (3000, 3000): (1680, 1680),
            (1000, 1000): (1008, 1008), (50, 2000): (112, 3444)}
    assert {k: extract.text_dimensions(*k) for k in pins} == pins


def test_text_requests_get_the_text_prompt_size_and_no_penalty():
    c = FakeClient([blk("text"), blk("table")])
    ProvidusParser("u", fixes=only("text_request_form", "text_prompt_newline", "text_resize", "text_no_penalty"),
                   client=c).parse_page(None)
    text_p, table_p = c.seen["params"]
    assert c.seen["prompts"] == ["\nText Recognition:\n", "\nTable Recognition:"]
    assert (text_p.presence_penalty, text_p.frequency_penalty, text_p.no_repeat_ngram_size) == (0.0, 0.0, 100)
    assert (table_p.presence_penalty, table_p.frequency_penalty) == (1.0, 0.05)  # table request untouched
    w, h = Image.open(io.BytesIO(c.seen["images"][0])).size
    assert w % 28 == 0 and h % 28 == 0 and w * h >= 294_768
    assert Image.open(io.BytesIO(c.seen["images"][1])).size == (100, 40)


@pytest.mark.parametrize("sub", ["text_prompt_newline", "text_resize", "text_no_penalty"])
def test_each_text_part_works_alone(sub):
    c = FakeClient([blk("text")])
    ProvidusParser("u", fixes=only("text_request_form", sub), client=c).parse_page(None)
    prompt, p, size = c.seen["prompts"][0], c.seen["params"][0], Image.open(io.BytesIO(c.seen["images"][0])).size
    assert (prompt == "\nText Recognition:\n") == (sub == "text_prompt_newline")
    assert (p.presence_penalty == 0.0) == (sub == "text_no_penalty")
    assert (size != (100, 40)) == (sub == "text_resize")
    assert c.seen["layout"] and "two_step" not in c.seen


def test_text_request_form_off_turns_all_three_parts_off():
    c = FakeClient([blk("text")])
    flags = {**only("image_block", "text_prompt_newline", "text_resize", "text_no_penalty"), "text_request_form": False}
    ProvidusParser("u", fixes=flags, client=c).parse_page(None)
    assert c.seen["prompts"] == ["\nText Recognition:"] and c.seen["params"][0].presence_penalty == 1.0
    assert Image.open(io.BytesIO(c.seen["images"][0])).size == (100, 40)


def test_wrappers_give_no_item_when_the_image_block_fix_is_off_but_text_fix_is_on():
    c = FakeClient([blk("image_block"), blk("text")])
    r = ProvidusParser("u", fixes=only("text_request_form", "text_prompt_newline"), client=c).parse_page(None)
    assert [i.category for i in r.items] == ["Text"]


def test_shared_params_object_is_not_changed():
    shared = SP(presence_penalty=1.0, frequency_penalty=0.05)
    out = extract.text_params(shared)
    assert out is not shared and shared.presence_penalty == 1.0 and out.presence_penalty == 0.0
    assert extract.text_params(None) is None


def test_resize_keeps_a_pil_image_a_pil_image():
    out = extract.resize_text_crop(Image.new("RGB", (100, 40)))
    assert isinstance(out, Image.Image) and out.size[0] % 28 == 0



# --- chart_request_form ---

def _sizes(images):
    return [Image.open(io.BytesIO(i)).size for i in images]


def test_chart_and_image_requests_get_the_resize_and_no_penalty_but_keep_their_prompt():
    c = FakeClient([blk("chart"), blk("image"), blk("text"), blk("table")])
    ProvidusParser("u", fixes=only("chart_request_form", "chart_resize", "chart_no_penalty"), client=c).parse_page(None)
    assert c.seen["prompts"] == ["\nImage Analysis:", "\nImage Analysis:", "\nText Recognition:", "\nTable Recognition:"]
    pens = [(p.presence_penalty, p.frequency_penalty) for p in c.seen["params"]]
    assert pens == [(0.0, 0.0), (0.0, 0.0), (1.0, 0.05), (1.0, 0.05)]
    assert c.seen["params"][0].no_repeat_ngram_size == 100
    chart, image, text, table = _sizes(c.seen["images"])
    assert chart == image == extract.text_dimensions(100, 40) and text == table == (100, 40)


@pytest.mark.parametrize("sub", ["chart_resize", "chart_no_penalty"])
def test_each_chart_part_works_alone(sub):
    c = FakeClient([blk("chart")])
    ProvidusParser("u", fixes=only("chart_request_form", sub), client=c).parse_page(None)
    assert (c.seen["params"][0].presence_penalty == 0.0) == (sub == "chart_no_penalty")
    assert (_sizes(c.seen["images"])[0] != (100, 40)) == (sub == "chart_resize")


def test_chart_request_form_off_turns_both_parts_off():
    c = FakeClient([blk("chart")])
    flags = {**only("image_block", "chart_resize", "chart_no_penalty"), "chart_request_form": False}
    ProvidusParser("u", fixes=flags, client=c).parse_page(None)
    assert c.seen["params"][0].presence_penalty == 1.0 and _sizes(c.seen["images"]) == [(100, 40)]


# --- the same fix against the real mineru-vl-utils steps (skipped if the package is not installed) ---

LAYOUT = ("<|box_start|>100 100 900 800<|box_end|><|ref_start|>image_block<|ref_end|><|rotate_up|>"
          "<|box_start|>100 100 500 800<|box_end|><|ref_start|>chart<|ref_end|><|rotate_up|>"
          "<|box_start|>500 100 900 800<|box_end|><|ref_start|>text<|ref_end|><|rotate_up|>")


CHART = ("<|class_start|>chart<|class_end|><|sub_class_start|>bar<|sub_class_end|>"
         "<|content_start|>| a | b |\n|---|---|\n| 1 | 2 |<|content_end|>")


class StubVlm:
    def __init__(self):
        self.requests = []

    def predict(self, image, prompt="", sampling_params=None, priority=None):
        return LAYOUT

    def batch_predict(self, images, prompts="", sampling_params=None, priority=None):
        self.requests += list(zip(prompts, sampling_params))
        return [CHART if "Image Analysis" in p else "Words." for p in prompts]


def real_client():
    mvu = pytest.importorskip("mineru_vl_utils")
    c = mvu.MinerUClient(backend="http-client", server_url="http://unused",
                         model_name="stub", skip_model_name_checking=True, image_analysis=True, use_tqdm=False)
    c.client = StubVlm()
    return c


def test_real_two_step_skips_the_chart_inside_an_image_block_and_the_fix_reads_it():
    c = real_client()
    page = Image.new("RGB", (400, 400), "white")
    before = [b.type for b in c.two_step_extract(page)]
    assert [p for p, _ in c.client.requests] == ["\nText Recognition:"]  # the chart gets no request
    c.client.requests.clear()
    r = ProvidusParser("u", fixes=only("image_block"), client=c).parse_page(page)
    assert [p for p, _ in c.client.requests] == ["\nImage Analysis:", "\nText Recognition:"]
    assert "image_block" in before
    assert [(i.category, i.text) for i in r.items] == [("Chart", "| a | b |\n|---|---|\n| 1 | 2 |"), ("Text", "Words.")]


def test_real_text_request_has_the_text_form_and_the_chart_request_is_unchanged():
    c = real_client()
    page = Image.new("RGB", (400, 400), "white")
    sent = []
    c.client.batch_predict = lambda images, prompts="", sampling_params=None, priority=None: (
        sent.extend(zip(images, prompts, sampling_params)) or [CHART if "Image" in p else "Words." for p in prompts])
    ProvidusParser("u", fixes=only("image_block", "text_request_form", "text_prompt_newline", "text_resize",
                                   "text_no_penalty"), client=c).parse_page(page)
    (chart_img, chart_prompt, chart_p), (text_img, text_prompt, text_p) = sent
    assert (chart_prompt, chart_p.presence_penalty, chart_p.frequency_penalty) == ("\nImage Analysis:", 1.0, 0.05)
    assert (text_prompt, text_p.presence_penalty, text_p.frequency_penalty) == ("\nText Recognition:\n", 0.0, 0.0)
    assert text_p.no_repeat_ngram_size == 100
    w, h = Image.open(io.BytesIO(text_img)).size
    assert w % 28 == 0 and h % 28 == 0 and w * h >= 294_768
    assert Image.open(io.BytesIO(chart_img)).size == (160, 280)  # chart crop: not resized


def test_real_chart_request_gets_the_resize_and_no_penalty():
    c = real_client()
    page = Image.new("RGB", (400, 400), "white")
    sent = []
    c.client.batch_predict = lambda images, prompts="", sampling_params=None, priority=None: (
        sent.extend(zip(images, prompts, sampling_params)) or [CHART if "Image" in p else "Words." for p in prompts])
    r = ProvidusParser("u", fixes=only("image_block", "chart_request_form", "chart_resize", "chart_no_penalty"),
                       client=c).parse_page(page)
    (chart_img, chart_prompt, chart_p), (text_img, text_prompt, text_p) = sent
    assert (chart_prompt, chart_p.presence_penalty, chart_p.frequency_penalty) == ("\nImage Analysis:", 0.0, 0.0)
    assert (text_prompt, text_p.presence_penalty) == ("\nText Recognition:", 1.0)
    assert Image.open(io.BytesIO(chart_img)).size == extract.text_dimensions(160, 280)
    assert Image.open(io.BytesIO(text_img)).size == (160, 280)
    assert r.items[0].category == "Chart"
