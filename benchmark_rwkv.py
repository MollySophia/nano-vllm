#!/usr/bin/env python3
import argparse
import os
import time

import torch

from nanovllm import LLM, SamplingParams
from nanovllm.utils.rwkv_int8 import (
    add_rwkv_int8_cli_args,
    describe_rwkv_int8_mode,
    resolve_rwkv_int8_lm_head_flags,
)
from nanovllm.utils.context import reset_context


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


def run_benchmark(
    model_pth: str,
    concurrency: int,
    prompt_length: int,
    decode_steps: int,
    gpu_memory_utilization: float,
    max_state_slots: int,
    rwkv_state_cache_safety_reserve_slots: int,
    rwkv_prefill_token_budget: int,
    rwkv_prefill_max_batch_size: int,
    rwkv_prefill_chunk_size: int,
    rwkv_state_cache_enable: bool,
    rwkv_quant_int8: bool,
    rwkv_int8_fp16_lm_head: bool = False,
    enforce_eager: bool = False,
    seed: int = 0,
) -> tuple[int, int, int, int, float, float, float, float | None]:
    (
        rwkv_quant_int8_lm_head,
        rwkv_quant_int8_lm_head_marlin,
    ) = resolve_rwkv_int8_lm_head_flags(
        rwkv_quant_int8=rwkv_quant_int8,
        rwkv_int8_fp16_lm_head=rwkv_int8_fp16_lm_head,
    )
    model_dir = ensure_model_dir(model_pth)
    # Prefill consumes the first sampled token, so request one extra token to leave
    # exactly `decode_steps` decode iterations after prefill.
    sampling_params = SamplingParams(temperature=0.0, ignore_eos=True, max_tokens=decode_steps + 1)
    requested_max_num_seqs = concurrency if concurrency != -1 else 4096
    llm = LLM(
        model_dir,
        enforce_eager=enforce_eager,
        tensor_parallel_size=1,
        max_num_seqs=requested_max_num_seqs,
        max_num_batched_tokens=max(16384, requested_max_num_seqs * prompt_length),
        max_model_len=8192,
        gpu_memory_utilization=gpu_memory_utilization,
        max_state_slots=max_state_slots,
        rwkv_state_cache_safety_reserve_slots=rwkv_state_cache_safety_reserve_slots,
        rwkv_prefill_token_budget=rwkv_prefill_token_budget,
        rwkv_prefill_max_batch_size=rwkv_prefill_max_batch_size,
        rwkv_prefill_chunk_size=rwkv_prefill_chunk_size,
        rwkv_state_cache_enable=rwkv_state_cache_enable,
        rwkv_quant_int8=rwkv_quant_int8,
        rwkv_int8_fp16_lm_head=rwkv_int8_fp16_lm_head,
    )
    vocab_size = int(llm.model_runner.config.model_config.vocab_size)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    prompt_tokens = torch.randint(0, vocab_size, (prompt_length,), generator=generator, dtype=torch.int64).tolist()
    if concurrency == -1:
        concurrency = llm.model_runner.config.num_state_blocks
    llm.model_runner.sampler.forward = lambda logits, temperatures: logits.argmax(dim=-1)
    for _ in range(concurrency):
        llm.add_request(prompt_tokens, sampling_params)

    torch.cuda.synchronize()
    prefill_t0 = time.perf_counter()
    prefill_tokens = 0
    while llm.scheduler.waiting or any(seq.num_prefill_tokens_remaining > 0 for seq in llm.scheduler.running):
        outputs, num_tokens = llm.step()
        assert len(outputs) == 0
        if num_tokens > 0:
            prefill_tokens += num_tokens
    torch.cuda.synchronize()
    prefill_dt = time.perf_counter() - prefill_t0
    prefill_tps = prefill_tokens / prefill_dt if prefill_dt > 0 else 0.0

    seqs = list(llm.scheduler.running)
    assert len(seqs) == concurrency, f"len(seqs) = {len(seqs)}, concurrency = {concurrency}, specified concurrency exceeded calculated memory limit"

    if concurrency == 1:
        seq = seqs[0]
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        input_ids, positions, temperatures = llm.model_runner.prepare_decode_single(seq)
        next_token = torch.tensor([seq.last_token], device=input_ids.device, dtype=input_ids.dtype)
        context_lens = llm.model_runner._bs1_decode_tensors["context_lens"]
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        steps = 0
        while steps < decode_steps:
            input_ids[0] = next_token[0]
            next_token = llm.model_runner.decode_single_step(seq, input_ids, positions, temperatures, record_sequence=False)
            positions.add_(1)
            context_lens.add_(1)
            steps += 1
        torch.cuda.synchronize()
        loop_dt = time.perf_counter() - t1
        reset_context()
        decode_dt = time.perf_counter() - t0
        resident_blocks = llm.model_runner.config.num_state_blocks
        llm.model_runner.call("exit")
        decode_tps = concurrency * steps / decode_dt
        steady_decode_tps = concurrency * steps / loop_dt
        return concurrency, resident_blocks, prefill_tokens, steps, prefill_dt, prefill_tps, decode_tps, steady_decode_tps

    torch.cuda.synchronize()
    decode_t0 = time.perf_counter()
    steps = 0
    while seqs:
        token_ids = llm.model_runner.call("run", seqs, False)
        llm.scheduler.postprocess(seqs, token_ids)
        steps += 1
        seqs = list(llm.scheduler.running)
    torch.cuda.synchronize()
    decode_dt = time.perf_counter() - decode_t0
    resident_blocks = llm.model_runner.config.num_state_blocks
    llm.exit()
    decode_tps = concurrency * steps / decode_dt
    return concurrency, resident_blocks, prefill_tokens, steps, prefill_dt, prefill_tps, decode_tps, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-pth", required=True)
    parser.add_argument("--concurrency", type=int, nargs="+", default=[512, 768])
    parser.add_argument("--prompt-length", type=int, default=4)
    parser.add_argument("--decode-steps", type=int, default=128)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.95)
    parser.add_argument("--max-state-slots", type=int, default=-1)
    parser.add_argument("--rwkv-state-cache-safety-reserve-slots", type=int, default=0)
    parser.add_argument("--rwkv-prefill-token-budget", type=int, default=2048)
    parser.add_argument("--rwkv-prefill-max-batch-size", type=int, default=128)
    parser.add_argument("--rwkv-prefill-chunk-size", type=int, default=-1)
    parser.add_argument("--rwkv-state-cache-enable", action="store_true")
    add_rwkv_int8_cli_args(parser)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    (
        rwkv_quant_int8_lm_head,
        rwkv_quant_int8_lm_head_marlin,
    ) = resolve_rwkv_int8_lm_head_flags(
        rwkv_quant_int8=args.rwkv_quant_int8,
        rwkv_int8_fp16_lm_head=args.rwkv_int8_fp16_lm_head,
    )
    mode_name = describe_rwkv_int8_mode(
        rwkv_quant_int8=args.rwkv_quant_int8,
        rwkv_quant_int8_lm_head=rwkv_quant_int8_lm_head,
        rwkv_quant_int8_lm_head_marlin=rwkv_quant_int8_lm_head_marlin,
    )

    for n in args.concurrency:
        torch.cuda.empty_cache()
        (
            actual_n,
            resident_blocks,
            prefill_tokens,
            steps,
            prefill_dt,
            prefill_tps,
            decode_tps,
            steady_decode_tps,
        ) = run_benchmark(
            args.model_pth,
            n,
            args.prompt_length,
            args.decode_steps,
            args.gpu_memory_utilization,
            args.max_state_slots,
            args.rwkv_state_cache_safety_reserve_slots,
            args.rwkv_prefill_token_budget,
            args.rwkv_prefill_max_batch_size,
            args.rwkv_prefill_chunk_size,
            args.rwkv_state_cache_enable,
            args.rwkv_quant_int8,
            args.rwkv_int8_fp16_lm_head,
            args.enforce_eager,
            args.seed,
        )
        summary = (
            f"gpu_memory_utilization={args.gpu_memory_utilization:.2f},"
            f"max_state_slots={args.max_state_slots},"
            f"rwkv_state_cache_safety_reserve_slots={args.rwkv_state_cache_safety_reserve_slots},"
            f"rwkv_prefill_token_budget={args.rwkv_prefill_token_budget},"
            f"rwkv_prefill_max_batch_size={args.rwkv_prefill_max_batch_size},"
            f"rwkv_prefill_chunk_size={args.rwkv_prefill_chunk_size},"
            f"rwkv_state_cache_enable={int(args.rwkv_state_cache_enable)},"
            f"rwkv_quant_int8={int(args.rwkv_quant_int8)},"
            f"rwkv_int8_fp16_lm_head={int(args.rwkv_int8_fp16_lm_head)},"
            f"rwkv_mode={mode_name},"
            f"prompt_length={args.prompt_length},seed={args.seed},"
            f"n={actual_n},resident_blocks={resident_blocks},"
            f"prefill_tokens={prefill_tokens},prefill_time_s={prefill_dt:.4f},prefill_tps={prefill_tps:.2f},"
            f"decode_steps={steps},decode_tps={decode_tps:.2f}"
        )
        if steady_decode_tps is not None:
            summary += f",steady_decode_tps={steady_decode_tps:.2f}"
        print(summary)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
