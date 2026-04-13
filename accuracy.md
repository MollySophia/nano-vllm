# Accuracy Notes

Relative comparison only. These scores were not tuned for prompt, sampling, stop rules, or extraction. They do not represent the model's true capability.

## Setup

- Date: `2026-04-13`
- Model: `/models/rwkv7-g1e-1.5b-20260309-ctx8192.pth`
- All runs are full-set comparisons on the same machine.

## GSM8K

- Dataset: official `gsm8k` `test`, `1319` examples
- Prompt: current `legacy` prompt in [eval_rwkv_gsm8k.py](/home/molly/nano-vllm/eval_rwkv_gsm8k.py:62)
- Generation: `temperature=0`, `max_tokens=768`

| backend | mode | acc | extract_rate | output_tps |
| --- | --- | ---: | ---: | ---: |
| `nano-vllm fp16` | `bs1 cudagraph` | `49.13` | `69.52` | `305.11` |
| `Albatross fp16` | `bs1 cudagraph` | `48.75` | `70.74` | `336.26` |
| `nano-vllm fp16` | `bs16` | `48.22` | `69.45` | `1013.31` |
| `Albatross fp16` | `bs16` | `47.54` | `68.01` | `2360.45` |

## LAMBADA

- Dataset: `nanovllm/eval_data/lambada_test.jsonl`, `5153` examples
- Metric: exact last-word teacher-forcing, `pad_eod=1`

| backend | mode | ppl | acc | target_tps |
| --- | --- | ---: | ---: | ---: |
| `nano-vllm fp16` | `bs1` | `4.6240` | `67.07` | `77.09` |
| `Albatross fp16` | `bs1` | `4.6225` | `67.09` | `159.71` |
| `nano-vllm fp16` | `bs16` | `4.6241` | `67.07` | `84.06` |
| `Albatross fp16` | `bs16` | `4.6217` | `67.11` | `486.28` |

## MMLU

- Dataset: `nanovllm/eval_data/mmlu_test_dataset`, `14042` examples
- Metric: prompt-only 4-choice logit selection

| backend | mode | acc | prompt_tps | examples_per_s |
| --- | --- | ---: | ---: | ---: |
| `nano-vllm fp16` | `bs1` | `50.62` | `8725.26` | `66.94` |
| `Albatross fp16` | `bs1` | `50.61` | `15988.55` | `122.67` |
| `nano-vllm fp16` | `bs16` | `50.56` | `30414.64` | `233.35` |
| `Albatross fp16` | `bs16` | `50.57` | `33581.61` | `257.65` |

## Files

- [nano_gsm8k_full_bs1_cudagraph_1p5b.jsonl](</tmp/nano_gsm8k_full_bs1_cudagraph_1p5b.jsonl>)
- [alba_gsm8k_full_bs1_cudagraph_1p5b.jsonl](</tmp/alba_gsm8k_full_bs1_cudagraph_1p5b.jsonl>)
- [nano_gsm8k_full_batch16_1p5b.jsonl](</tmp/nano_gsm8k_full_batch16_1p5b.jsonl>)
- [alba_gsm8k_full_batch16_1p5b.jsonl](</tmp/alba_gsm8k_full_batch16_1p5b.jsonl>)
