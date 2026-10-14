# Serving providus-vl-mini for ParseBench

This guide is for the ParseBench team. It shows how to start the servers behind the `providus_vlm_mini` pipeline
and how to run a smoke test. The full model-side guide is
[serve/README.md](https://huggingface.co/ProvidusAI/providus-vl-mini/blob/main/serve/README.md) in the model repo. The commands below are the same commands, shortened.

## About

providus-vl-mini is a fine-tune of MinerU2.5-Pro by OpenDataLab (Apache-2.0). We did not pretrain a model.
providus-parse-tools extends mineru-vl-utils by OpenDataLab.
Scores are the mean of 4 full ParseBench runs (± standard deviation).
No rule in our pipeline contains ParseBench documents, file names or answers. We used benchmark results to find common error types, then wrote general rules and checked them on our own data.

## What you need

- One GPU. The v1 runs used one H100 80 GB (each of the two vLLM servers takes 25% of the GPU memory). We have not measured smaller cards yet (see "Speed by GPU" below).
- Python 3.12 (tested), and vLLM 0.28.0.
- Access to the gated repo [ProvidusAI/providus-vl-mini](https://huggingface.co/ProvidusAI/providus-vl-mini): accept the terms on the page, then `hf auth login`. Access is approved automatically.

## 1. Install and download

Run the commands from one working folder.

```
pip install vllm==0.28.0 pyyaml transformers==5.19.0 peft==0.21.2 providus-parse-tools==1.0.0
hf download opendatalab/MinerU2.5-Pro-2605-1.2B --revision bff20d4ae2bf202df9f45284b4d43681555a97ed --local-dir ./base
hf download ProvidusAI/providus-vl-mini --local-dir ./providus-vl-mini
```

We tested these steps with Python 3.12 in one new environment. Use Python 3.12.
`transformers==5.19.0` is the version that vLLM 0.28.0 brought in our test (vLLM needs 5.5.3 or newer). `torchvision` and
`pillow` come with vLLM, and the optional merge below uses them too. `peft` is needed only for that merge.

The first `hf download` line gets the base model from OpenDataLab. The second gets the model repo: the merged model is at its root
(the base with the text LoRA merged in, bf16), and the adapter alone is in `adapter/`. There is no merge step.

Optional: to make the merged model yourself from the base and `adapter/`, run this once. It needs a few GB of memory and no GPU.
It writes `./merged`; to serve it, use `./merged` in place of `./providus-vl-mini` in the command of terminal 2.

```python
import torch
from peft import PeftModel
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

model = Qwen2VLForConditionalGeneration.from_pretrained("./base", dtype=torch.bfloat16)
model = PeftModel.from_pretrained(model, "./providus-vl-mini/adapter").merge_and_unload()
model.save_pretrained("./merged")
AutoProcessor.from_pretrained("./base").save_pretrained("./merged")
```

## 2. Start the model servers

Use one terminal for each command. Start them one at a time, and start the next only when the one before answers
(`curl -sf http://127.0.0.1:8001/v1/models`). If you start them together, each vLLM counts the memory of the others as its own.

Terminal 1, the base model:

```
export VLLM_USE_FLASHINFER_SAMPLER=0
vllm serve ./base --served-model-name providus-vl-mini --port 8001 --max-model-len 8192 --gpu-memory-utilization 0.25 --max-num-seqs 64 --limit-mm-per-prompt '{"image":1}'
```

Terminal 2, providus-vl-mini (the merged model):

```
export VLLM_USE_FLASHINFER_SAMPLER=0
vllm serve ./providus-vl-mini --served-model-name providus-vl-mini --port 8002 --max-model-len 8192 --gpu-memory-utilization 0.25 --max-num-seqs 64 --limit-mm-per-prompt '{"image":1}'
```

Terminal 3, the router. It listens on port 8000. It sends a request to providus-vl-mini when the prompt contains
`Text Recognition`, and all other requests to the base model:

```
python providus-vl-mini/serve/router.py providus-vl-mini/serve/router.yaml
```

## 3. Start the parse server

Terminal 4:

```
python -m providus_parse.server --port 8100 --vllm http://127.0.0.1:8000
```

The server listens on port 8100. The provider posts each page to `POST /predict` as a PNG image
(`image_base64`) and, for a PDF source, also as a one-page PDF (`pdf_base64`). The server uses the PDF text layer for bold, italic and heading marks.

## 4. Point ParseBench at it

The provider reads one environment variable, `PROVIDUS_PARSE_URL` (the address of the parse server, without `/predict`).
You can also set `server_url` in the pipeline config.

```
export PROVIDUS_PARSE_URL=http://127.0.0.1:8100
```

## 5. Smoke test

From a ParseBench checkout that has the `providus_vlm_mini` provider. `--test` runs the small dataset (3 files per category).
With `--group table` it should be about 3 documents:

```
export PROVIDUS_PARSE_URL=http://127.0.0.1:8100
uv sync --extra runners
uv run parse-bench run providus_vlm_mini --test --group table --max_concurrent 4 --open_report=False
```

If a page fails, the provider records it as a failed page (`failed_pages` in the raw output). It does not stop the run.
If the server is not reachable, ParseBench retries the document, then records it as failed (the error names the page and the file) and goes on with the next document.

## 6. Full run

```
uv run parse-bench run providus_vlm_mini --max_concurrent 24
```

The full set has 2,078 pages. In the v1 runs (one H100 80 GB, `--max_concurrent 24`), a run took 43.4 to 44.9 minutes
from start to scores: inference and scoring, with the servers already running (mean of 4 runs: 44.4 minutes). Over a full run that is about 1.28 s per page. Zero per-page fees: providus-vl-mini is free to use under Apache-2.0. Your only cost is your own hardware.
The scores depend on the whole setup above (the base model for layout, tables, formulas and charts, providus-vl-mini for text crops, the router, and providus-parse-tools 1.0.0).
A different setup gives different scores.

## Speed by GPU

Measured on 200 ParseBench pages (page 1 of each document), one rented GPU, 2026-10-13. Batch: 24 pages at a time. One user: 50 other pages, one at a time.

| GPU | Memory | s/page (24 at a time) | s/page (one at a time) | Pages/hour |
|---|---:|---:|---:|---:|
| NVIDIA H100 80GB HBM3 | 79.6 GiB | 0.61 | 3.87 | 5,893 |

Smaller GPUs are not measured yet.

## Our numbers

Mean of 4 full ParseBench runs (± standard deviation), scorer ParseBench main b9ba246 (2026-10-12):

| Overall | Tables | Charts | Content faithfulness | Semantic formatting | Visual grounding |
|---|---|---|---|---|---|
| 78.96 ± 0.04 | 86.33 ± 0.02 | 71.14 ± 0.10 | 87.71 ± 0.10 | 74.41 ± 0.12 | 75.19 ± 0.03 |

These are a suggestion. Please run it yourselves.
