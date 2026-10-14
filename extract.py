"""Page extraction: layout detection, then one request for each region.

It uses the same steps as MinerUClient.two_step_extract (mineru-vl-utils 2.0.7, mineru_client.py:992),
so that the fixes can change a step. If no fix needs it, the plain two_step_extract runs.
"""

from __future__ import annotations

import copy
import io
import math

from PIL import Image

from providus_parse import fixes
from providus_parse.markdown import WRAPPER_TYPES

# The text request form (flag text_request_form).
# MinerUClient drops the newline, sends the raw crop, and sets presence penalty 1.0 and frequency penalty 0.05.
TEXT_PROMPT = "\nText Recognition:\n"
TEXT_PARTS = ("text_prompt_newline", "text_resize", "text_no_penalty")
CHART_PARTS = ("chart_resize", "chart_no_penalty")  # the same resize and penalties for image analysis requests
NON_TEXT_TYPES = {"table", "equation", "image", "chart"}  # these types have their own prompt in MinerUClient
IMAGE_TYPES = {"image", "chart"}  # the image analysis requests; the prompt stays as MinerUClient sends it
_FACTOR, _MIN_PIXELS, _MAX_PIXELS, _MAX_RATIO = 28, 294_768, 2_822_400, 200


def text_dimensions(w: int, h: int) -> tuple[int, int]:
    """Multiples of 28 px, between 294,768 and 2,822,400 pixels. Aspect ratio at most 200."""
    if max(w, h) / min(w, h) > _MAX_RATIO:
        raise ValueError(f"aspect ratio over {_MAX_RATIO}: {w}x{h}")

    tw, th = (max(_FACTOR, round(v / _FACTOR) * _FACTOR) for v in (w, h))
    if tw * th > _MAX_PIXELS:
        s = math.sqrt(w * h / _MAX_PIXELS)
        tw, th = (max(_FACTOR, math.floor(int(v / s) / _FACTOR) * _FACTOR) for v in (w, h))
    elif tw * th < _MIN_PIXELS:
        s = math.sqrt(_MIN_PIXELS / (w * h))
        tw, th = (math.ceil(int(v * s) / _FACTOR) * _FACTOR for v in (w, h))
        if tw * th > _MAX_PIXELS:  # a very long strip can overshoot when it grows
            s = math.sqrt(tw * th / _MAX_PIXELS)
            tw, th = (max(_FACTOR, math.floor(int(v / s) / _FACTOR) * _FACTOR) for v in (tw, th))
    return tw, th


def resize_text_crop(crop):
    """Resize one crop (PNG bytes or a PIL image) to text_dimensions. Same type out."""
    as_bytes = isinstance(crop, (bytes, bytearray))
    img = Image.open(io.BytesIO(crop)) if as_bytes else crop
    try:
        size = text_dimensions(*img.size)
    except ValueError:
        return crop
    if size == img.size:
        return crop
    out = img.convert("RGB").resize(size, Image.Resampling.BICUBIC)
    if not as_bytes:
        return out
    buf = io.BytesIO()
    out.save(buf, "PNG")
    return buf.getvalue()


def text_params(params):
    """A copy of the sampling params without presence and frequency penalty. The input stays as it is."""
    if params is None:
        return None
    out = copy.copy(params)
    out.presence_penalty = out.frequency_penalty = 0.0
    return out


def extract_blocks(client, image, flags: dict[str, bool] | None = None):
    drop_wrappers = fixes.enabled("image_block", flags)
    parts = [n for n in TEXT_PARTS if fixes.enabled("text_request_form", flags) and fixes.enabled(n, flags)]
    parts += [n for n in CHART_PARTS if fixes.enabled("chart_request_form", flags) and fixes.enabled(n, flags)]
    if not drop_wrappers and not parts:
        return client.two_step_extract(image)
    layout = list(client.layout_detect(image))
    if drop_wrappers:
        # An image_block wrapper makes mineru-vl-utils skip the charts and images inside it
        # (mineru_client.py:330-333, 363). Without the wrapper, each child is a normal region.
        layout = [b for b in layout if b.type not in WRAPPER_TYPES]
    images, prompts, params, indices = client.helper.prepare_for_extract(image, layout)
    for k, i in enumerate(indices):
        if layout[i].type in IMAGE_TYPES:
            if "chart_resize" in parts:
                images[k] = resize_text_crop(images[k])
            if "chart_no_penalty" in parts:
                params[k] = text_params(params[k])
            continue
        if layout[i].type in NON_TEXT_TYPES:
            continue
        if "text_prompt_newline" in parts:
            prompts[k] = TEXT_PROMPT
        if "text_resize" in parts:
            images[k] = resize_text_crop(images[k])
        if "text_no_penalty" in parts:
            params[k] = text_params(params[k])
    texts = client.client.batch_predict(images, prompts, params) if images else []
    for i, text in zip(indices, texts):
        layout[i].content = text
    return client.helper.post_process(layout)
