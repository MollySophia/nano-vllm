#!/usr/bin/env python3
import argparse
import os
import time

import torch

from nanovllm import LLM, SamplingParams


def ensure_model_dir(model_pth: str) -> str:
    model_name = os.path.basename(model_pth).replace(".pth", "")
    model_dir = os.path.join("/tmp", f"{model_name}_nanovllm")
    os.makedirs(model_dir, exist_ok=True)
    link = os.path.join(model_dir, "model.pth")
    if not (os.path.islink(link) and os.path.realpath(link) == model_pth):
        if os.path.exists(link) or os.path.islink(link):
            os.remove(link)
        os.symlink(model_pth, link)
    return model_dir


def run_decode_only(
    model_pth: str,
    concurrency: int,
    prompt_tokens: list[int],
    decode_steps: int,
    gpu_memory_utilization: float,
    rwkv_prefill_token_budget: int,
    rwkv_prefill_max_batch_size: int,
    rwkv_quant_int8: bool,
):
    model_dir = ensure_model_dir(model_pth)
    # Prefill consumes the first sampled token, so request one extra token to leave
    # exactly `decode_steps` decode iterations after prefill.
    sampling_params = SamplingParams(temperature=1e-4, ignore_eos=True, max_tokens=decode_steps + 1)
    requested_max_num_seqs = concurrency if concurrency != -1 else 4096
    llm = LLM(
        model_dir,
        enforce_eager=True,
        tensor_parallel_size=1,
        max_num_seqs=requested_max_num_seqs,
        max_num_batched_tokens=max(16384, requested_max_num_seqs * len(prompt_tokens)),
        max_model_len=8192,
        gpu_memory_utilization=gpu_memory_utilization,
        rwkv_prefill_token_budget=rwkv_prefill_token_budget,
        rwkv_prefill_max_batch_size=rwkv_prefill_max_batch_size,
        rwkv_quant_int8=rwkv_quant_int8,
    )
    if concurrency == -1:
        concurrency = llm.model_runner.config.num_state_blocks
    llm.model_runner.sampler.forward = lambda logits, temperatures: logits.argmax(dim=-1)
    for _ in range(concurrency):
        llm.add_request(prompt_tokens, sampling_params)

    while llm.scheduler.waiting:
        outputs, _ = llm.step()  # prefill only
        assert len(outputs) == 0
    seqs = list(llm.scheduler.running)
    assert len(seqs) == concurrency, f"len(seqs) = {len(seqs)}, concurrency = {concurrency}, specified concurrency exceeded calculated memory limit"

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    steps = 0
    while seqs:
        token_ids = llm.model_runner.call("run", seqs, False)
        llm.scheduler.postprocess(seqs, token_ids)
        steps += 1
        seqs = list(llm.scheduler.running)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    resident_blocks = llm.model_runner.config.num_state_blocks
    llm.exit()
    decode_tps = concurrency * steps / dt
    return concurrency, resident_blocks, steps, dt, decode_tps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-pth", required=True)
    parser.add_argument("--concurrency", type=int, nargs="+", default=[512, 768])
    parser.add_argument("--prompt-tokens", type=int, nargs="+", default=[3645, 6579, 10737, 15388])
    parser.add_argument("--decode-steps", type=int, default=128)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.95)
    parser.add_argument("--rwkv-prefill-token-budget", type=int, default=2048)
    parser.add_argument("--rwkv-prefill-max-batch-size", type=int, default=128)
    parser.add_argument("--rwkv-quant-int8", action="store_true")
    args = parser.parse_args()

    for n in args.concurrency:
        torch.cuda.empty_cache()
        actual_n, resident_blocks, steps, dt, decode_tps = run_decode_only(
            args.model_pth,
            n,
            args.prompt_tokens,
            args.decode_steps,
            args.gpu_memory_utilization,
            args.rwkv_prefill_token_budget,
            args.rwkv_prefill_max_batch_size,
            args.rwkv_quant_int8,
        )
        print(
            f"gpu_memory_utilization={args.gpu_memory_utilization:.2f},"
            f"rwkv_prefill_token_budget={args.rwkv_prefill_token_budget},"
            f"rwkv_prefill_max_batch_size={args.rwkv_prefill_max_batch_size},"
            f"rwkv_quant_int8={int(args.rwkv_quant_int8)},"
            f"n={actual_n},resident_blocks={resident_blocks},decode_steps={steps},"
            f"time_s={dt:.4f},decode_tps={decode_tps:.2f}"
        )
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
