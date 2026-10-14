# providus-parse-tools

providus-parse-tools is the document pipeline of the model `providus-vl-mini`.
It turns a page image into markdown and layout items.
providus-parse-tools extends mineru-vl-utils by OpenDataLab. The MinerU client finds the layout regions and reads each region.
This package adds the output format (markdown and layout items), output fixes and an HTTP server.
providus-vl-mini is a fine-tune of MinerU2.5-Pro by OpenDataLab (Apache-2.0). We did not pretrain a model.

## Install

```
pip install providus-parse-tools
```

The distribution name is `providus-parse-tools`. The Python import name is `providus_parse`.

```python
from providus_parse.pipeline import ProvidusParser
```

To parse one page (the router address is the one from "Run the server" below):

```python
from PIL import Image

parser = ProvidusParser("http://127.0.0.1:8000")
page = parser.parse_page(Image.open("page.png").convert("RGB"), pdf=open("page.pdf", "rb").read())  # pdf is optional
print(page.markdown)
```

## Serve the model

The step-by-step guide is [SERVING.md](https://github.com/ProvidusAI/providus-parse-tools/blob/v1/SERVING.md) in this repo.
The model repo has the same steps in [serve/README.md](https://huggingface.co/ProvidusAI/providus-vl-mini/blob/main/serve/README.md).

`providus-vl-mini` is a fine-tune of MinerU2.5-Pro-2605-1.2B. Serve it and the base model with vLLM.
Send the text crops to providus-vl-mini. Send all other requests (layout, tables, formulas, charts) to the base model.
Then start the server in the next section and point `--vllm` at the router (port 8000).

## Run the server

Start the vLLM servers and the router first (see the guide above). The router listens on port 8000: it sends the text crops to the merged model and all other requests to the base model.
Point the parse server at the router, not at a single vLLM server. Then run:

```
python -m providus_parse.server --port 8100 --vllm http://127.0.0.1:8000
```

The ParseBench runs of the scores rendered the pages at 144 DPI. Send page images with the same size if you want comparable results.

Send `POST /predict` with the body `{"image_base64": "<base64 of the page image>"}`.
For a PDF page, also send `"pdf_base64"`: a one-page PDF of the same page. The server then uses its text layer.
The answer has `markdown`, `items`, `text_layer`, `image_width`, `image_height` and `status`.
`text_layer` gives the count of text blocks that use the layer text (`used`) and that keep the model's text (`kept_model`).
If the page has a usable layer, `bold_share` gives the share of bold layer characters. The server log has one line per page with the three values.
Each item has `category`, `bbox` (0 to 1 page coordinates), `text` and `type` (the mineru-vl-utils block type).
The text of a `Title` item has no `#` marker. The markdown keeps the marker.
If an error occurs, `status` is `error` and `error` gives the reason.

## Output fixes

Each fix has a flag. Set `PROVIDUS_FIX_<NAME>=0` to turn one off, or `PROVIDUS_FIX_<NAME>=1` to turn one on.
The defaults are the settings of the ParseBench runs (the scores of providus-vl-mini): you need no environment variable to repeat them.
The `fixes` argument of `ProvidusParser` has priority over the environment.

| Flag | Default | Effect |
|---|---|---|
| `header_footer` | on | Header, footer and page number text goes into the page markdown. |
| `table_header` | on | Table HTML gets `<thead>` and `<th>` rows. |
| `table_header_first_row` | on | A table without a rowspan in its first row also gets a header row. |
| `image_block` | on | The charts and images inside an `image_block` get their own request. |
| `text_request_form` | on | Text-region requests use the providus-vl-mini text form. Off turns off the three parts below. |
| `text_prompt_newline` | on | The text prompt keeps its trailing newline. |
| `text_resize` | on | The text crop is resized before the request. |
| `text_no_penalty` | on | The text request has no presence penalty and no frequency penalty. |
| `chart_request_form` | on | Chart and image requests get the resize and no penalties of the text form. The prompt does not change. Off turns off the two parts below. |
| `chart_resize` | on | The chart or image crop is resized as a text crop. |
| `chart_no_penalty` | on | The chart or image request has no presence penalty and no frequency penalty. |
| `chart_caption` | on | A chart gets its title or caption as `<caption>` in its table. The layout item of a chart has the plain cell text only. |
| `text_layer` | on | For a PDF page, text blocks use the PDF text layer when it agrees with the model's text. Off turns off the three parts below. |
| `text_layer_marks` | on | The layer fonts give the bold, italic and superscript marks. |
| `text_layer_chars` | off | On: the block gets the layer characters and marks. Off: only the marks move onto the model's words. |
| `text_layer_headings` | on | The layer font sizes of the titles give the heading levels (`#`, `##`, `###`). |
| `heading_class` | off | On: a short bold or large `text` line becomes a `section_header`. |
| `para_split` | off | On: a `text` block that uses the layer text gives one layout item per paragraph. The markdown does not change. |

## Text layer

`textlayer.py` reads the PDF text layer with `pypdfium2` 5.13.0 (Apache-2.0 or BSD-3-Clause; PDFium is BSD-3-Clause).
Code, formula and table blocks keep the model's text. If an error occurs in the text layer path, the block keeps the model's text. The page does not fail.

## Results

Scores are the mean of 4 full ParseBench runs (± standard deviation).
Scorer: ParseBench main b9ba246 (2026-10-12). Each run has 2,078 pages. No rule in our pipeline contains ParseBench documents, file names or answers. We used benchmark results to find common error types, then wrote general rules and checked them on our own data.

| Overall | Tables | Charts | Content faithfulness | Semantic formatting | Visual grounding |
|---|---|---|---|---|---|
| 78.96 ± 0.04 | 86.33 ± 0.02 | 71.14 ± 0.10 | 87.71 ± 0.10 | 74.41 ± 0.12 | 75.19 ± 0.03 |

These scores are for the whole v1 system, not for the adapter alone: the base model, the text LoRA adapter on the text route, and providus-parse-tools 1.0.0, which also reads the PDF text layer for formatting marks (bold, italic, headings). A different pipeline will give different scores.

The score of providus-vl-mini (78.96) would place 7th overall and first among open-weight models on leaderboard.csv (b9ba246, 2026-10-12). The 7th place is 0.04 above the next row (Anthropic Fable 5.1, 78.92), which is within our run spread. providus-vl-mini is not listed in that file yet.
The next open-weight row, rakedoc-nano (77.23), is licensed AGPL-3.0. providus-vl-mini is Apache-2.0, so companies can use it in closed-source products and services.

## License and attribution

This package is Apache-2.0 (see `LICENSE`). Keep the `NOTICE` file when you redistribute it or a derivative.
If you offer an online service with it and providus-vl-mini, please state in its interface or public documentation that it
uses MinerU by OpenDataLab and providus-vl-mini by Providus AI (MinerU's project licence asks online services to say so; the credit to Providus AI is our request).
`providus-vl-mini` is a fine-tune of MinerU2.5-Pro-2605-1.2B by OpenDataLab (Apache-2.0). We did not pretrain a model.
To cite this work, use `CITATION.cff`. Please also cite the MinerU papers below.

## Acknowledgements

- OpenDataLab and the MinerU team: the MinerU2.5-Pro model and `mineru-vl-utils`. This work is built on them.
- The ParseBench authors: the benchmark we measure with.
- The vLLM project: model serving.
- Hugging Face PEFT.

## BibTeX

```bibtex
@article{wang2026mineru25pro,
  title   = {MinerU2.5-Pro: Pushing the Limits of Data-Centric Document Parsing at Scale},
  author  = {Wang, Bin and others},
  journal = {arXiv preprint arXiv:2604.04771},
  year    = {2026}
}

@article{niu2025mineru2,
  title   = {MinerU2.5: A Decoupled Vision-Language Model for Efficient High-Resolution Document Parsing},
  author  = {Niu, Junbo and others},
  journal = {arXiv preprint arXiv:2509.22186},
  year    = {2025}
}

@article{zhang2026parsebench,
  title   = {ParseBench: A Document Parsing Benchmark for AI Agents},
  author  = {Zhang, Boyang and others},
  journal = {arXiv preprint arXiv:2604.08538},
  year    = {2026}
}

```
