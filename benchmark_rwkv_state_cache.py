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


def build_partial_prompts(base_prompt: list[int], extra_len: int, trials: int, vocab_size: int) -> list[list[int]]:
    prompts: list[list[int]] = []
    for i in range(trials):
        suffix = [((50000 + i) % max(vocab_size, 1))]
        while len(suffix) < extra_len:
            suffix.append(((51000 + i * 17 + len(suffix)) % max(vocab_size, 1)))
        prompts.append(base_prompt + suffix)
    return prompts


def build_miss_prompts(prompt_len: int, trials: int, vocab_size: int, seed: int) -> list[list[int]]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    prompts = []
    for _ in range(trials):
        prompt = torch.randint(0, vocab_size, (prompt_len,), generator=generator, dtype=torch.int64).tolist()
        prompts.append(prompt)
    return prompts


def run_request(llm: LLM, prompt: list[int], max_tokens: int) -> tuple[list[int], dict[str, int | bool | None], float, float]:
    sp = SamplingParams(temperature=0.0, ignore_eos=True, max_tokens=max_tokens)
    llm.add_request(prompt, sp)
    seq = llm.scheduler.waiting[-1]
    meta = {}
    t0 = time.perf_counter()
    first_token_dt = None
    while not seq.is_finished:
        seqs, is_prefill = llm.scheduler.schedule()
        assert len(seqs) == 1
        if not meta:
            scheduled = seqs[0]
            meta = {
                "is_prefill": is_prefill,
                "exact": bool(getattr(scheduled, "exact_cache_hit", False)),
                "cached_prefix_len": int(getattr(scheduled, "cached_prefix_len", 0)),
                "prompt_cache_slot": getattr(scheduled, "prompt_cache_slot", None),
                "state_slot": getattr(scheduled, "state_slot", None),
                "cache_hit_slot": getattr(scheduled, "cache_hit_slot", None),
            }
        token_ids = llm.model_runner.call("run", seqs, is_prefill)
        llm.scheduler.postprocess(seqs, token_ids)
        if first_token_dt is None:
            first_token_dt = time.perf_counter() - t0
    total_dt = time.perf_counter() - t0
    return list(seq.completion_token_ids), meta, first_token_dt or total_dt, total_dt


def avg(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def run_suite(
    model_dir: str,
    cache_enable: bool,
    warmup_prompt: list[int],
    base_prompt: list[int],
    miss_prompts: list[list[int]],
    partial_prompts: list[list[int]],
    max_tokens: int,
    gpu_memory_utilization: float,
    max_state_slots: int,
    rwkv_state_cache_safety_reserve_slots: int,
) -> dict[str, list[dict]]:
    llm = LLM(
        model_dir,
        enforce_eager=True,
        tensor_parallel_size=1,
        max_num_seqs=4,
        max_num_batched_tokens=max(16384, 4 * max(len(base_prompt), *(len(p) for p in partial_prompts), *(len(p) for p in miss_prompts))),
        max_model_len=8192,
        gpu_memory_utilization=gpu_memory_utilization,
        max_state_slots=max_state_slots,
        rwkv_state_cache_safety_reserve_slots=rwkv_state_cache_safety_reserve_slots,
        rwkv_state_cache_enable=cache_enable,
    )
    results = {"miss": [], "exact": [], "partial": []}

    # Compile / allocator warmup on an unrelated prompt so measured scenarios
    # start from a steady engine state.
    run_request(llm, warmup_prompt, 1)

    for prompt in miss_prompts:
        output, meta, first_dt, total_dt = run_request(llm, prompt, max_tokens)
        results["miss"].append(
            {
                "prompt": prompt,
                "output": output,
                "meta": meta,
                "first_ms": first_dt * 1000,
                "total_ms": total_dt * 1000,
            }
        )

    seed_output, seed_meta, seed_first_dt, seed_total_dt = run_request(llm, base_prompt, max_tokens)
    results["miss"].append(
        {
            "prompt": base_prompt,
            "output": seed_output,
            "meta": seed_meta,
            "first_ms": seed_first_dt * 1000,
            "total_ms": seed_total_dt * 1000,
        }
    )

    for _ in range(len(partial_prompts)):
        output, meta, first_dt, total_dt = run_request(llm, base_prompt, max_tokens)
        results["exact"].append(
            {
                "prompt": base_prompt,
                "output": output,
                "meta": meta,
                "first_ms": first_dt * 1000,
                "total_ms": total_dt * 1000,
            }
        )

    for prompt in partial_prompts:
        output, meta, first_dt, total_dt = run_request(llm, prompt, max_tokens)
        results["partial"].append(
            {
                "prompt": prompt,
                "output": output,
                "meta": meta,
                "first_ms": first_dt * 1000,
                "total_ms": total_dt * 1000,
            }
        )

    llm.exit()
    return results


def summarize(case: str, off_rows: list[dict], on_rows: list[dict]) -> str:
    same_count = sum(int(off["output"] == on["output"]) for off, on in zip(off_rows, on_rows))
    exact_rate = sum(int(bool(row["meta"].get("exact"))) for row in on_rows) / len(on_rows) if on_rows else 0.0
    avg_cached_prefix = avg([float(row["meta"].get("cached_prefix_len", 0)) for row in on_rows])
    return (
        f"case={case},trials={len(on_rows)},"
        f"off_first_ms={avg([row['first_ms'] for row in off_rows]):.2f},"
        f"on_first_ms={avg([row['first_ms'] for row in on_rows]):.2f},"
        f"off_total_ms={avg([row['total_ms'] for row in off_rows]):.2f},"
        f"on_total_ms={avg([row['total_ms'] for row in on_rows]):.2f},"
        f"same_outputs={same_count}/{len(on_rows)},"
        f"on_exact_rate={exact_rate:.2f},"
        f"on_avg_cached_prefix_len={avg_cached_prefix:.2f}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-pth", required=True)
    parser.add_argument("--prompt-length", type=int, default=64)
    parser.add_argument("--partial-extra-length", type=int, default=32)
    parser.add_argument("--trials", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.95)
    parser.add_argument("--max-state-slots", type=int, default=-1)
    parser.add_argument("--rwkv-state-cache-safety-reserve-slots", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    model_dir = ensure_model_dir(args.model_pth)
    probe_llm = LLM(
        model_dir,
        enforce_eager=True,
        tensor_parallel_size=1,
        max_num_seqs=1,
        max_num_batched_tokens=max(16384, args.prompt_length + args.partial_extra_length),
        max_model_len=8192,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_state_slots=args.max_state_slots,
        rwkv_state_cache_safety_reserve_slots=args.rwkv_state_cache_safety_reserve_slots,
    )
    vocab_size = int(probe_llm.model_runner.config.model_config.vocab_size)
    probe_llm.exit()

    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed)
    base_prompt = torch.randint(0, vocab_size, (args.prompt_length,), generator=generator, dtype=torch.int64).tolist()
    warmup_prompt = torch.randint(0, vocab_size, (args.prompt_length,), generator=generator, dtype=torch.int64).tolist()
    miss_prompts = build_miss_prompts(args.prompt_length, args.trials, vocab_size, args.seed + 1)
    partial_prompts = build_partial_prompts(base_prompt, args.partial_extra_length, args.trials, vocab_size)

    off = run_suite(
        model_dir,
        False,
        warmup_prompt,
        base_prompt,
        miss_prompts,
        partial_prompts,
        args.max_tokens,
        args.gpu_memory_utilization,
        args.max_state_slots,
        args.rwkv_state_cache_safety_reserve_slots,
    )
    on = run_suite(
        model_dir,
        True,
        warmup_prompt,
        base_prompt,
        miss_prompts,
        partial_prompts,
        args.max_tokens,
        args.gpu_memory_utilization,
        args.max_state_slots,
        args.rwkv_state_cache_safety_reserve_slots,
    )

    print(
        f"model={args.model_pth},prompt_length={args.prompt_length},partial_extra_length={args.partial_extra_length},"
        f"trials={args.trials},max_tokens={args.max_tokens},gpu_memory_utilization={args.gpu_memory_utilization:.2f},"
        f"max_state_slots={args.max_state_slots},"
        f"rwkv_state_cache_safety_reserve_slots={args.rwkv_state_cache_safety_reserve_slots}"
    )
    print(summarize("miss", off["miss"][:args.trials], on["miss"][:args.trials]))
    print(summarize("exact", off["exact"], on["exact"]))
    print(summarize("partial", off["partial"], on["partial"]))


if __name__ == "__main__":
    main()
