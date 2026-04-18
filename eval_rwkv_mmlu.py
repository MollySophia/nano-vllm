#!/usr/bin/env python3
import argparse
import json
import os
import random
import time
from collections import deque
from dataclasses import dataclass

import torch

from nanovllm import LLM, SamplingParams  # noqa: E402
from nanovllm.tokenizers import get_rwkv_tokenizer  # noqa: E402
from nanovllm.utils.rwkv_int8 import (  # noqa: E402
    add_rwkv_int8_cli_args,
    describe_rwkv_int8_mode,
    resolve_rwkv_int8_lm_head_flags,
)

try:
    from datasets import load_from_disk
except ImportError as exc:  # pragma: no cover - exercised only when dependency is missing.
    load_from_disk = None
    DATASETS_IMPORT_ERROR = exc
else:
    DATASETS_IMPORT_ERROR = None


DEFAULT_MMLU = os.path.join(
    os.path.dirname(__file__),
    "nanovllm",
    "eval_data",
    "mmlu_test_dataset",
)
TEMPLATE = """User: You are a very talented expert in <SUBJECT>. Answer this question:
<Q>
A. <|A|>
B. <|B|>
C. <|C|>
D. <|D|>

Assistant: The answer is"""
CHOICES = [" A", " B", " C", " D"]


@dataclass
class Sample:
    question: str
    choices: list[str]
    subject: str
    answer: int
    prompt: str
    prompt_token_ids: list[int]
    prompt_tokens: int
    predicted: int | None = None
    is_correct: bool = False


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


def resolve_mmlu_path(path: str) -> str:
    if os.path.isdir(path):
        return path
    raise SystemExit(f"MMLU dataset path does not exist: {path}")


def require_datasets() -> None:
    if load_from_disk is not None:
        return
    raise SystemExit(
        "eval_rwkv_mmlu.py requires the HuggingFace `datasets` package because it "
        "loads a local on-disk MMLU dataset snapshot. "
        "Install it in the active environment and rerun.\n"
        f"Original import error: {DATASETS_IMPORT_ERROR}"
    )


def build_prompt(question: str, choices: list[str], subject: str) -> str:
    return (
        TEMPLATE.replace("<Q>", question)
        .replace("<|A|>", choices[0])
        .replace("<|B|>", choices[1])
        .replace("<|C|>", choices[2])
        .replace("<|D|>", choices[3])
        .replace("<SUBJECT>", subject.replace("_", " "))
    )


def load_mmlu(
    path: str,
    tokenizer,
    limit: int,
    shuffle_choices: bool,
    seed: int,
    subjects: set[str] | None,
) -> list[Sample]:
    require_datasets()
    ds = load_from_disk(resolve_mmlu_path(path))
    rng = random.Random(seed)
    samples: list[Sample] = []
    for row in ds:
        subject = row["subject"]
        if subjects is not None and subject not in subjects:
            continue
        choices = list(row["choices"])
        gt = int(row["answer"])
        if shuffle_choices and not any("Both" in choice for choice in choices):
            gt_text = choices[gt]
            rng.shuffle(choices)
            gt = choices.index(gt_text)
        prompt = build_prompt(row["question"], choices, subject)
        prompt = prompt.replace("\r\n", "\n").strip()
        prompt_token_ids = [0] + tokenizer.encode(prompt)
        samples.append(
            Sample(
                question=row["question"],
                choices=choices,
                subject=subject,
                answer=gt,
                prompt=prompt,
                prompt_token_ids=prompt_token_ids,
                prompt_tokens=len(prompt_token_ids),
            )
        )
        if limit > 0 and len(samples) >= limit:
            break
    return samples


def write_predictions(path: str, samples: list[Sample]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for sample in samples:
            row = {
                "subject": sample.subject,
                "question": sample.question,
                "choices": sample.choices,
                "gold_answer": sample.answer,
                "predicted_answer": sample.predicted,
                "is_correct": sample.is_correct,
                "prompt_tokens": sample.prompt_tokens,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def evaluate_chunk(
    llm: LLM,
    runner,
    chunk: list[Sample],
    choice_token_ids: torch.Tensor,
) -> tuple[int, int]:
    total_prompt_tokens = sum(sample.prompt_tokens for sample in chunk)
    sampling = SamplingParams(temperature=1e-4, ignore_eos=True, max_tokens=1)
    pending = deque(chunk)

    for sample in chunk:
        llm.add_request(sample.prompt_token_ids, sampling)

    while llm.scheduler.waiting:
        seqs, is_prefill = llm.scheduler.schedule()
        assert is_prefill
        batch_samples = [pending.popleft() for _ in seqs]
        logits = runner.call("run_logits", seqs, is_prefill)
        batch_choice_token_ids = choice_token_ids.to(device=logits.device)
        choice_logits = logits.index_select(1, batch_choice_token_ids)
        predicted = choice_logits.argmax(dim=-1)
        dummy_tokens = logits.argmax(dim=-1).tolist()

        for row, sample in enumerate(batch_samples):
            sample.predicted = int(predicted[row].item())
            sample.is_correct = sample.predicted == sample.answer

        runner.call("prepare_postprocess", seqs, dummy_tokens)
        llm.scheduler.postprocess(seqs, dummy_tokens)

    assert not llm.scheduler.running
    return len(chunk), total_prompt_tokens


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-pth", required=True)
    parser.add_argument("--mmlu-path", default=DEFAULT_MMLU)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--shuffle-choices", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--subject", action="append", default=None)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.97)
    parser.add_argument("--max-state-slots", type=int, default=-1)
    parser.add_argument("--rwkv-state-cache-safety-reserve-slots", type=int, default=0)
    parser.add_argument("--rwkv-prefill-token-budget", type=int, default=2048)
    parser.add_argument("--rwkv-prefill-max-batch-size", type=int, default=128)
    parser.add_argument("--rwkv-state-cache-enable", action="store_true")
    add_rwkv_int8_cli_args(parser)
    parser.add_argument(
        "--enforce-eager",
        dest="enforce_eager",
        action="store_true",
        help="Disable CUDA graph capture in the underlying LLM. Default: enabled.",
    )
    parser.add_argument(
        "--no-enforce-eager",
        dest="enforce_eager",
        action="store_false",
        help="Allow CUDA graph capture in the underlying LLM when available.",
    )
    parser.add_argument("--print-interval", type=int, default=512)
    parser.add_argument("--predictions-path", default=None)
    parser.set_defaults(enforce_eager=True)
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
    subjects = set(args.subject) if args.subject else None
    samples = load_mmlu(
        args.mmlu_path,
        tokenizer,
        args.limit,
        args.shuffle_choices,
        args.seed,
        subjects,
    )
    if not samples:
        raise SystemExit("No MMLU samples selected.")

    choice_token_ids = [tokenizer.encode(choice) for choice in CHOICES]
    if not all(len(token_ids) == 1 for token_ids in choice_token_ids):
        raise SystemExit(f"Expected single-token choices, got: {choice_token_ids}")
    choice_token_ids_tensor = torch.tensor(
        [token_ids[0] for token_ids in choice_token_ids],
        dtype=torch.int64,
        device="cuda",
    )

    print("format_example:")
    print("-" * 80)
    print(samples[0].prompt)
    print("-" * 80)

    max_prompt_tokens = max(sample.prompt_tokens for sample in samples)
    model_dir = ensure_model_dir(args.model_pth)
    llm = LLM(
        model_dir,
        enforce_eager=args.enforce_eager,
        tensor_parallel_size=1,
        max_num_seqs=max(args.batch_size, 8),
        max_num_batched_tokens=max(16384, args.batch_size * max_prompt_tokens),
        max_model_len=8192,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_state_slots=args.max_state_slots,
        rwkv_state_cache_safety_reserve_slots=args.rwkv_state_cache_safety_reserve_slots,
        rwkv_prefill_token_budget=args.rwkv_prefill_token_budget,
        rwkv_prefill_max_batch_size=args.rwkv_prefill_max_batch_size,
        rwkv_state_cache_enable=args.rwkv_state_cache_enable,
        rwkv_quant_int8=args.rwkv_quant_int8,
        rwkv_int8_fp16_lm_head=args.rwkv_int8_fp16_lm_head,
    )
    runner = llm.model_runner

    total_examples = 0
    total_prompt_tokens = 0
    total_correct = 0

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for start in range(0, len(samples), args.batch_size):
        chunk = samples[start:start + args.batch_size]
        n_examples, n_prompt_tokens = evaluate_chunk(llm, runner, chunk, choice_token_ids_tensor)
        total_examples += n_examples
        total_prompt_tokens += n_prompt_tokens
        total_correct += sum(1 for sample in chunk if sample.is_correct)

        if total_examples % args.print_interval == 0 or total_examples == len(samples):
            acc = total_correct / total_examples * 100
            print(
                f"done={total_examples},acc={acc:.2f},"
                f"prompt_tokens={total_prompt_tokens}"
            )
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0

    if args.predictions_path:
        write_predictions(args.predictions_path, samples)

    acc = total_correct / total_examples * 100
    prompt_tps = total_prompt_tokens / dt
    examples_per_s = total_examples / dt
    print(
        f"final_examples={total_examples},acc={acc:.2f},"
        f"prompt_tokens={total_prompt_tokens},time_s={dt:.4f},"
        f"prompt_tps={prompt_tps:.2f},examples_per_s={examples_per_s:.2f},"
        f"batch_size={args.batch_size},enforce_eager={int(args.enforce_eager)},"
        f"rwkv_quant_int8={int(args.rwkv_quant_int8)},rwkv_mode={rwkv_mode},"
        f"shuffle_choices={int(args.shuffle_choices)}"
    )
    llm.exit()


if __name__ == "__main__":
    main()
