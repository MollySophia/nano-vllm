# OpenAI-Compatible API

This repository now includes a minimal OpenAI-compatible HTTP server built on top of `nanovllm.LLM`.

## Start the Server

```bash
python -m nanovllm.entrypoints.openai.api_server \
  --model /models/rwkv7-g1e-1.5b-20260309-ctx8192.pth \
  --served-model-name rwkv7-1p5b
```

On POSIX platforms, this now defaults to the highest-throughput API path:

- `2` frontend worker processes
- queue-backed single backend model process
- `aiohttp` as the frontend HTTP stack
- `rwkv_prefill_chunk_size=256` to keep long-prompt prefill from monopolizing a whole step

To force the older single-process path for debugging or profiling, pass:

```bash
python -m nanovllm.entrypoints.openai.api_server \
  --model /models/rwkv7-g1e-1.5b-20260309-ctx8192.pth \
  --served-model-name rwkv7-1p5b \
  --frontend-workers 1
```

Optional authentication:

```bash
python -m nanovllm.entrypoints.openai.api_server \
  --model /models/rwkv7-g1e-1.5b-20260309-ctx8192.pth \
  --served-model-name rwkv7-1p5b \
  --api-key local-dev-key
```

## Supported Endpoints

- `GET /health`
- `GET /v1/models`
- `GET /v1/models/{model_id}`
- `POST /v1/completions`
- `POST /v1/chat/completions`

## Supported Request Fields

Shared subset:

- `model`
- `temperature`
- `max_tokens`
- `max_completion_tokens` for chat completions

Accepted only at their current default / no-op values:

- `n=1`
- `top_p=1`
- `presence_penalty=0`
- `frequency_penalty=0`

## Unsupported Fields

The server currently returns `400` for unsupported behavior instead of silently ignoring it. Not supported yet:

- stop sequences
- tools / tool_choice
- response_format
- logprobs / top_logprobs
- echo
- seed
- batched prompt arrays beyond a single prompt

## Prompt Rendering

- RWKV chat requests now use a native `apply_chat_template()` modeled on `rwkv-mobile` defaults:
  `System:` / `User:` / `Assistant:` role labels, `\n\n` turn separators, and an `Assistant:` generation prompt after a trailing user turn.
- Other tokenizer-backed chat models with `apply_chat_template` use the tokenizer's native template.
- Tokenizers without any chat template still fall back to a simple plain-text transcript ending with `Assistant:`.

## Performance Metadata

The server keeps the standard OpenAI-style response body shape intact:

- token usage stays in the JSON body `usage`
- response bodies do not add non-standard speed fields such as `tokens_per_second`

Instead, timing and throughput data are exposed through headers:

- `openai-processing-ms`
- `x-request-id`
- `x-nanovllm-streaming`
- `x-nanovllm-metrics-scope`
- `x-nanovllm-queue-wait-ms`
- `x-nanovllm-prompt-tokens`
- `x-nanovllm-ttft-ms`

Sync responses also include final generation metrics:

- `x-nanovllm-completion-tokens`
- `x-nanovllm-generation-ms`
- `x-nanovllm-output-tokens-per-second`
- `x-nanovllm-decode-tokens-per-second`
- `x-nanovllm-total-ms`

Streaming responses only expose partial metrics in headers because headers are sent before the stream finishes:

- `openai-processing-ms` means time until the stream is ready to start
- `x-nanovllm-metrics-scope=partial`
- final decode throughput is not sent in headers for streaming requests

## Example Requests

Completion:

```bash
curl http://127.0.0.1:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "rwkv7-1p5b",
    "prompt": "The quick brown fox",
    "max_tokens": 32,
    "temperature": 0
  }'
```

Streaming completion:

```bash
curl -N http://127.0.0.1:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "rwkv7-1p5b",
    "prompt": "The quick brown fox",
    "max_tokens": 32,
    "temperature": 0,
    "stream": true
  }'
```

Chat completion:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "rwkv7-1p5b",
    "messages": [
      {"role": "user", "content": "请用中文简单解释什么是线性注意力。"}
    ],
    "max_tokens": 64,
    "temperature": 0
  }'
```

Show response headers:

```bash
curl -i http://127.0.0.1:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "rwkv7-1p5b",
    "prompt": "The quick brown fox",
    "max_tokens": 32,
    "temperature": 0
  }'
```

## Load Testing

Use [benchmark_openai_api.py](/home/molly/nano-vllm/benchmark_openai_api.py) to simulate many concurrent users against the OpenAI-compatible API.

Fixed request count:

```bash
python benchmark_openai_api.py \
  --base-url http://127.0.0.1:8000 \
  --model rwkv7-1p5b \
  --endpoint chat \
  --users 128 \
  --total-requests 2000 \
  --max-tokens 64
```

Duration-based streaming run:

```bash
python benchmark_openai_api.py \
  --base-url http://127.0.0.1:8000 \
  --model rwkv7-1p5b \
  --endpoint chat \
  --users 128 \
  --duration 60 \
  --stream \
  --max-tokens 64
```

Concurrency sweep:

```bash
python benchmark_openai_api.py \
  --base-url http://127.0.0.1:8000 \
  --model rwkv7-1p5b \
  --endpoint chat \
  --users-sweep 1 2 4 8 16 32 64 128 \
  --total-requests 512 \
  --max-tokens 64 \
  --output-json sweep_summary.json
```

Custom prompt corpus:

```bash
python benchmark_openai_api.py \
  --base-url http://127.0.0.1:8000 \
  --model rwkv7-1p5b \
  --prompt-file prompts.txt \
  --users 64 \
  --total-requests 1000 \
  --output-json load_test.json \
  --details-jsonl load_test.jsonl \
  --details-csv load_test.csv
```

The benchmark reports:

- request throughput and success rate
- client-observed latency percentiles
- client TTFT percentiles for streaming requests
- server-reported `openai-processing-ms`
- server-reported queue wait / TTFT / generation time when exposed in headers
- token throughput when the API returns token counts
- status-code breakdown and sample errors

When `--users-sweep` is used:

- each concurrency point is run separately
- the terminal output includes a compact sweep table
- `--output-json` writes a `runs` array instead of a single summary object
- `--details-jsonl` / `--details-csv` aggregate per-request rows across all sweep points

## Plotting Sweep Results

Use [plot_benchmark_openai_api.py](/home/molly/nano-vllm/plot_benchmark_openai_api.py) to turn benchmark output into PNG charts.

Summary-only plots:

```bash
python plot_benchmark_openai_api.py \
  --summary-json sweep_summary.json
```

Summary plus request-level detail plots:

```bash
python plot_benchmark_openai_api.py \
  --summary-json sweep_summary.json \
  --details-file sweep_details.csv \
  --output-dir sweep_plots
```

This writes:

- `summary.png`
  shows requests/s, latency, TTFT, queue wait, token throughput, and error rate versus concurrency
- `details.png`
  shows request-level boxplots for latency, TTFT, and queue wait across concurrency points

## Notes

- The default POSIX server mode uses queue-backed multi-frontend ingress. The backend still owns a single shared `LLM` instance and forms batches there.
- `--frontend-workers 1` falls back to the older single-process API path.
- Streaming uses SSE with `data: ...` chunks and a final `data: [DONE]`.
- Streamed chat completions intentionally omit a terminal `finish_reason="length"` marker so `openai` Python SDK `chat.completions.stream().get_final_completion()` can return the aggregated text instead of raising `LengthFinishReasonError`. Non-stream chat/completion responses still report the accurate finish reason.
