"""HTTP server. POST /predict with {"image_base64", "pdf_base64" (optional)} answers markdown, items and the image size."""

from __future__ import annotations

import argparse
import base64
import binascii
import io
import json
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from PIL import Image

from providus_parse.pipeline import ProvidusParser


def _pdf(data) -> bytes | None:
    """The one-page PDF of the page, or None. A bad PDF never fails the page: the image path runs."""
    try:
        return base64.b64decode(data) if isinstance(data, str) and data else None
    except (binascii.Error, ValueError):
        return None


def predict(raw: bytes, parser: ProvidusParser) -> dict:
    # Decoding is inside the try. A bad body must answer status=error, not drop the connection.
    try:
        body = json.loads(raw)
        img = Image.open(io.BytesIO(base64.b64decode(body["image_base64"]))).convert("RGB")
        result = parser.parse_page(img, pdf=_pdf(body.get("pdf_base64")))
        items = [{**asdict(i), "bbox": list(i.bbox)} for i in result.items]
        return {"markdown": result.markdown or " ", "items": items, "text_layer": result.text_layer,
                "image_width": img.width, "image_height": img.height, "status": "success"}
    except Exception as e:
        return {"status": "error", "error": f"{type(e).__name__}: {e}"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8100)
    ap.add_argument("--vllm", required=True, help="URL of the router (it sends text crops to the merged model and the rest to the base), for example http://127.0.0.1:8000")
    a = ap.parse_args()
    parser = ProvidusParser(a.vllm)

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            data = predict(self.rfile.read(int(self.headers.get("Content-Length") or 0)), parser)
            if data.get("text_layer"):  # the server log shows how often the layer is used
                t = data["text_layer"]
                print(f"text_layer used={t['used']} kept_model={t['kept_model']} bold_share={t.get('bold_share', '-')}",
                      flush=True)
            raw = json.dumps(data).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def log_message(self, *args):
            pass

    ThreadingHTTPServer(("127.0.0.1", a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
