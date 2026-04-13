# Performance Notes

- Date: `2026-04-13`
- Model: `/models/rwkv7-g1e-7.2b-20260301-ctx8192.pth`
- Concurrency: `1 32 128 256 320 512 768 960`
- Settings: `prompt_length=4`, `decode_steps=128`, `seed=0`
- `nano-vllm int8` uses the default `int8_marlin_lm_head` path.
- `nano-vllm` was measured with [benchmark_rwkv.py](/home/molly/nano-vllm/benchmark_rwkv.py).
- `Albatross fp16` was measured with a local warmup harness against `reference.rwkv7`, because the current `Albatross` repo no longer ships the newer CLI benchmark entrypoint.

## bs=1

`nano-vllm` reports `steady_decode_tps`; `Albatross` reports `cudagraph_decode_tps`.

| backend | prefill_tps | decode_tps | steady / cudagraph_decode_tps | decode vs Albatross | steady vs Albatross graph |
| --- | ---: | ---: | ---: | ---: | ---: |
| `Albatross fp16` | `59.00` | `96.29` | `116.89` | `100.0%` | `100.0%` |
| `nano-vllm fp16` | `146.64` | `109.54` | `112.12` | `113.8%` | `95.9%` |
| `nano-vllm int8` | `177.58` | `127.67` | `133.72` | `132.6%` | `114.4%` |

## Decode

| concurrency | `Albatross fp16` | `nano-vllm fp16` | `nano-vllm int8` | `fp16 / Albatross` | `int8 / Albatross` |
| --- | ---: | ---: | ---: | ---: | ---: |
| `32` | `2274.53` | `2009.35` | `2122.24` | `88.3%` | `93.3%` |
| `128` | `7157.91` | `6534.00` | `6536.43` | `91.3%` | `91.3%` |
| `256` | `8301.67` | `7897.44` | `8070.27` | `95.1%` | `97.2%` |
| `320` | `9382.43` | `8899.59` | `8628.95` | `94.9%` | `92.0%` |
| `512` | `9032.69` | `8752.59` | `9252.35` | `96.9%` | `102.4%` |
| `768` | `9590.23` | `9442.70` | `9756.08` | `98.5%` | `101.7%` |
| `960` | `9891.51` | `9815.77` | `9923.79` | `99.2%` | `100.3%` |

## Prefill

For `Albatross`, the `bs=1` prefill number comes from its raw single-sequence path and is less representative. The multi-batch points are the more useful comparison.

| concurrency | `Albatross fp16` | `nano-vllm fp16` | `nano-vllm int8` | `fp16 / Albatross` | `int8 / Albatross` |
| --- | ---: | ---: | ---: | ---: | ---: |
| `32` | `9551.42` | `4776.64` | `4719.33` | `50.0%` | `49.4%` |
| `128` | `13881.84` | `11457.66` | `12099.90` | `82.5%` | `87.2%` |
| `256` | `14539.44` | `11475.95` | `12145.78` | `78.9%` | `83.5%` |
| `320` | `15965.06` | `10938.44` | `11439.41` | `68.5%` | `71.7%` |
| `512` | `15124.69` | `11484.48` | `12192.23` | `75.9%` | `80.6%` |
| `768` | `15610.93` | `11482.42` | `12159.68` | `73.6%` | `77.9%` |
| `960` | `15946.52` | `11292.10` | `11890.15` | `70.8%` | `74.6%` |

## Takeaways

- `bs=1` decode: `nano-vllm fp16` is faster than `Albatross fp16` on normal decode, but still slightly below `Albatross` CUDAGraph steady speed. `nano-vllm int8` is the fastest single-request path.
- Fixed-concurrency decode: `nano-vllm fp16` is still behind at `32` and `128`, recovers to `95%+` from `256`, and is essentially tied by `960`. `nano-vllm int8` overtakes `Albatross fp16` at `512`, `768`, and `960`.
- Fixed-concurrency prefill: `Albatross fp16` remains clearly ahead across all tested multi-batch points. `nano-vllm int8` is consistently faster than `nano-vllm fp16`, but neither catches `Albatross` on prefill.
