<p align="center">
<img width="300" src="assets/logo.png">
</p>

# Nano-vLLM RWKV

A lightweight RWKV7 inference server based on nano-vLLM. This fork focuses on
RWKV `.pth` checkpoints, OpenAI-compatible serving, rwkv_lightning-style private
batch APIs, CUDA graph decode, state-cache experiments, and local fp16/int8
benchmarking.

## Install

```bash
uv venv
source .venv/bin/activate
```

Use `uv sync --extra ...` to select the Torch build for your platform:

```bash
uv sync --extra torch-cu130
uv sync --extra torch-cu126
uv sync --extra torch-cpu
uv sync --extra torch-rocm
```

If you just want an editable install without syncing extras:

```bash
uv pip install -e .
```

Legacy pip path:

```bash
pip install -e .
```
> [!TIP]
> On Windows, the recommended setup is to run nano-vLLM inside WSL2.

Use the local `nanovllm` conda environment when running on Molly's machine:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate nanovllm
```

## Start Server

OpenAI-compatible server:

```bash
python -m nanovllm.entrypoints.openai.api_server \
  --model /models/rwkv7-g1e-7.2b-20260301-ctx8192.pth \
  --served-model-name rwkv7-7.2b \
  --host 0.0.0.0 \
  --port 8000 \
  --gpu-memory-utilization 0.97 \
  --rwkv-prefill-token-budget 2048 \
  --rwkv-prefill-max-batch-size 128 \
  --rwkv-prefill-chunk-size 256
```

Optional int8 path:

```bash
python -m nanovllm.entrypoints.openai.api_server \
  --model /models/rwkv7-g1e-7.2b-20260301-ctx8192.pth \
  --served-model-name rwkv7-7.2b-int8 \
  --gpu-memory-utilization 0.97 \
  --rwkv-quant-int8
```

## API Examples

Standard OpenAI chat:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "rwkv7-7.2b",
    "messages": [{"role": "user", "content": "请简单介绍 RWKV。"}],
    "max_tokens": 128,
    "temperature": 0.8
  }'
```

rwkv_lightning-compatible private batch form:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "rwkv7-7.2b",
    "contents": [
      "User: Give me one short idea for dinner.\n\nAssistant:",
      "User: Give me one short idea for a weekend project.\n\nAssistant:"
    ],
    "max_tokens": 128,
    "stop_tokens": ["\nUser:"],
    "temperature": 0.8,
    "top_k": 50,
    "top_p": 0.6,
    "alpha_presence": 1.0,
    "alpha_frequency": 0.1,
    "alpha_decay": 0.996,
    "stream": false
  }'
```

Private compatibility endpoints also include:

- `/v2/chat/completions`
- `/state/chat/completions`
- `/state/status`
- `/state/delete`
- `/translate/v1/batch-translate`
- `/FIM/v1/batch-FIM`
- `/openai/v1/chat/completions`

## Idealized Benchmark

Direct model benchmark, not HTTP serving:

```bash
python benchmark_rwkv.py \
  --model-pth /models/rwkv7-g1e-7.2b-20260301-ctx8192.pth \
  --concurrency 960 \
  --prompt-length 4 \
  --decode-steps 128 \
  --seed 0 \
  --gpu-memory-utilization 0.97 \
  --rwkv-prefill-token-budget 2048 \
  --rwkv-prefill-max-batch-size 128
```

Representative idealized local results on RTX 5090 32GB:

| concurrency | fp16 prefill_tps | fp16 decode_tps | int8 prefill_tps | int8 decode_tps |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 146.64 | 109.54 | 177.58 | 127.67 |
| 32 | 4776.64 | 2009.35 | 4719.33 | 2122.24 |
| 128 | 11457.66 | 6534.00 | 12099.90 | 6536.43 |
| 256 | 11475.95 | 7897.44 | 12145.78 | 8070.27 |
| 320 | 10938.44 | 8899.59 | 11439.41 | 8628.95 |
| 512 | 11484.48 | 8752.59 | 12192.23 | 9252.35 |
| 768 | 11482.42 | 9442.70 | 12159.68 | 9756.08 |
| 960 | 11292.10 | 9815.77 | 11890.15 | 9923.79 |

Recent spot rerun for fp16 `concurrency=960` produced `prefill_tps=8909.41`
and `decode_tps=9787.51` while GPU util was effectively saturated during
active decode.

Resident-concurrency probes:

| mode | resident_blocks | decode_tps |
| --- | ---: | ---: |
| fp16 | 991 | 9108.78 |
| int8 | 1350 | 10010.96 |

`int8` uses the default Marlin lm_head path. These are direct model benchmark
numbers and should not be compared with HTTP multi-user serving latency.

More benchmark details are in [performance.md](performance.md).

## Accuracy

Accuracy notes and evaluation tables are in [accuracy.md](accuracy.md).

## Tests

```bash
python scripts/run_tests.py --skip-prepare-test-env
```

Use the full flow when the external comparison checkout needs to be refreshed:

```bash
python scripts/run_tests.py
```
