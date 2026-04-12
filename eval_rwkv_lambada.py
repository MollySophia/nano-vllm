#!/usr/bin/env python3
import argparse
import json
import math
import os
import time
from dataclasses import dataclass

import torch

ROOT = os.path.dirname(__file__)

from nanovllm import LLM, SamplingParams  # noqa: E402
from nanovllm.tokenizers import RWKVTokenizer, get_rwkv_tokenizer
from nanovllm.utils.rwkv_int8 import (  # noqa: E402
    add_rwkv_int8_cli_args,
    describe_rwkv_int8_mode,
    resolve_rwkv_int8_lm_head_flags,
)


DEFAULT_LAMBADA = os.path.join(ROOT, "Albatross-better", "eval", "lambada_test.jsonl")


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


@dataclass
class SampleState:
    prefix_tokens: list[int]
    target_tokens: list[int]
    logprob_sum: float = 0.0
    correct: bool = True


def load_lambada(tokenizer: RWKVTokenizer, path: str, limit: int, pad_eod: bool) -> list[SampleState]:
    samples: list[SampleState] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)
            prefix, last_word = doc["text"].rsplit(" ", 1)
            prefix_tokens = tokenizer.encode(prefix)
            if pad_eod:
                prefix_tokens = [0] + prefix_tokens
            target_tokens = tokenizer.encode(" " + last_word)
            samples.append(
                SampleState(
                    prefix_tokens=prefix_tokens,
                    target_tokens=target_tokens,
                )
            )
            if limit > 0 and len(samples) >= limit:
                break
    return samples


def score_logits(logits: torch.Tensor, seqs, sample_map: dict[int, SampleState], target_idx: int) -> list[int]:
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    argmax = logits.argmax(dim=-1)
    next_tokens: list[int] = []
    for row, seq in enumerate(seqs):
        sample = sample_map[seq.seq_id]
        gold = sample.target_tokens[target_idx]
        sample.logprob_sum += float(log_probs[row, gold].item())
        if int(argmax[row].item()) != gold:
            sample.correct = False
        next_tokens.append(gold)
    return next_tokens


def eval_chunk_decode_only(llm: LLM, runner, chunk: list[SampleState]) -> tuple[int, int]:
    sample_map: dict[int, SampleState] = {}
    total_target_tokens = sum(len(s.target_tokens) for s in chunk)

    for sample in chunk:
        full_tokens = sample.prefix_tokens + sample.target_tokens
        sampling = SamplingParams(temperature=1e-4, ignore_eos=True, max_tokens=len(full_tokens) - 1)
        llm.add_request([full_tokens[0]], sampling)

    while llm.scheduler.waiting:
        seqs, is_prefill = llm.scheduler.schedule()
        assert is_prefill
        for seq in seqs:
            sample_map[seq.seq_id] = chunk[len(sample_map)]
        input_ids, positions = runner.prepare_prefill(seqs)
        logits = runner.run_model(input_ids, positions, True)
        gold_tokens = score_logits(logits, seqs, sample_map, 0)
        llm.scheduler.postprocess(seqs, gold_tokens)

    while llm.scheduler.running:
        seqs, is_prefill = llm.scheduler.schedule()
        assert not is_prefill
        input_ids, positions = runner.prepare_decode(seqs)
        logits = runner.run_model(input_ids, positions, False)
        target_idx = seqs[0].num_completion_tokens
        gold_tokens = score_logits(logits, seqs, sample_map, target_idx)
        llm.scheduler.postprocess(seqs, gold_tokens)

    return len(chunk), total_target_tokens


def eval_chunk_prefill_then_decode(llm: LLM, runner, chunk: list[SampleState]) -> tuple[int, int]:
    sample_map: dict[int, SampleState] = {}
    total_target_tokens = sum(len(s.target_tokens) for s in chunk)

    for sample in chunk:
        sampling = SamplingParams(temperature=1e-4, ignore_eos=True, max_tokens=len(sample.target_tokens))
        llm.add_request(sample.prefix_tokens, sampling)
        seqs, is_prefill = llm.scheduler.schedule()
        assert is_prefill and len(seqs) == 1
        seq = seqs[0]
        sample_map[seq.seq_id] = sample
        input_ids, positions = runner.prepare_prefill(seqs)
        logits = runner.run_model(input_ids, positions, True)
        gold_tokens = score_logits(logits, seqs, sample_map, 0)
        llm.scheduler.postprocess(seqs, gold_tokens)

    while llm.scheduler.running:
        seqs, is_prefill = llm.scheduler.schedule()
        assert not is_prefill
        input_ids, positions = runner.prepare_decode(seqs)
        logits = runner.run_model(input_ids, positions, False)
        target_idx = seqs[0].num_completion_tokens
        gold_tokens = score_logits(logits, seqs, sample_map, target_idx)
        llm.scheduler.postprocess(seqs, gold_tokens)

    return len(chunk), total_target_tokens


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-pth", required=True)
    parser.add_argument("--lambada-path", default=DEFAULT_LAMBADA)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--pad-eod", action="store_true")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.97)
    parser.add_argument("--rwkv-prefill-token-budget", type=int, default=2048)
    parser.add_argument("--rwkv-prefill-max-batch-size", type=int, default=128)
    add_rwkv_int8_cli_args(parser)
    parser.add_argument("--print-interval", type=int, default=1000)
    parser.add_argument(
        "--mode",
        choices=["decode_only", "prefill_then_decode"],
        default="prefill_then_decode",
        help="prefill_then_decode matches the original Lambada-style scoring; decode_only is an optional decode-heavy teacher-forcing mode.",
    )
    args = parser.parse_args()
    try:
        (
            rwkv_quant_int8_lm_head,
            rwkv_quant_int8_lm_head_marlin,
        ) = resolve_rwkv_int8_lm_head_flags(
            rwkv_quant_int8=args.rwkv_quant_int8,
            rwkv_int8_fp16_lm_head=args.rwkv_int8_fp16_lm_head,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    rwkv_mode = describe_rwkv_int8_mode(
        rwkv_quant_int8=args.rwkv_quant_int8,
        rwkv_quant_int8_lm_head=rwkv_quant_int8_lm_head,
        rwkv_quant_int8_lm_head_marlin=rwkv_quant_int8_lm_head_marlin,
    )

    tokenizer = get_rwkv_tokenizer()
    samples = load_lambada(tokenizer, args.lambada_path, args.limit, args.pad_eod)
    model_dir = ensure_model_dir(args.model_pth)
    llm = LLM(
        model_dir,
        enforce_eager=True,
        tensor_parallel_size=1,
        max_num_seqs=max(args.batch_size, 8),
        max_num_batched_tokens=max(16384, args.batch_size * 32),
        max_model_len=8192,
        gpu_memory_utilization=args.gpu_memory_utilization,
        rwkv_prefill_token_budget=args.rwkv_prefill_token_budget,
        rwkv_prefill_max_batch_size=args.rwkv_prefill_max_batch_size,
        rwkv_quant_int8=args.rwkv_quant_int8,
        rwkv_int8_fp16_lm_head=args.rwkv_int8_fp16_lm_head,
    )
    runner = llm.model_runner

    total_examples = 0
    total_target_tokens = 0
    total_logprob = 0.0
    total_correct = 0

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for start in range(0, len(samples), args.batch_size):
        chunk = samples[start:start + args.batch_size]
        if args.mode == "decode_only":
            n_examples, n_target_tokens = eval_chunk_decode_only(llm, runner, chunk)
        else:
            n_examples, n_target_tokens = eval_chunk_prefill_then_decode(llm, runner, chunk)
        total_examples += n_examples
        total_target_tokens += n_target_tokens
        total_logprob += sum(s.logprob_sum for s in chunk)
        total_correct += sum(1 for s in chunk if s.correct)

        if total_examples % args.print_interval == 0 or total_examples == len(samples):
            ppl = math.exp(-total_logprob / total_examples)
            acc = total_correct / total_examples * 100
            print(
                f"done={total_examples},ppl={ppl:.4f},acc={acc:.2f},"
                f"target_tokens={total_target_tokens}"
            )
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0

    ppl = math.exp(-total_logprob / total_examples)
    acc = total_correct / total_examples * 100
    target_tps = total_target_tokens / dt
    print(
        f"final_examples={total_examples},ppl={ppl:.4f},acc={acc:.2f},"
        f"target_tokens={total_target_tokens},time_s={dt:.4f},target_tps={target_tps:.2f},"
        f"rwkv_quant_int8={int(args.rwkv_quant_int8)},rwkv_mode={rwkv_mode},mode={args.mode}"
    )
    llm.exit()


if __name__ == "__main__":
    main()
