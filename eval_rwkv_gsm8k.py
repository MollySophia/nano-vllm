#!/usr/bin/env python3
import argparse
import json
import os
import re
import time
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from fractions import Fraction

import torch

from nanovllm import LLM, SamplingParams  # noqa: E402
from nanovllm.tokenizers import get_rwkv_tokenizer  # noqa: E402
from nanovllm.utils.rwkv_int8 import (  # noqa: E402
    add_rwkv_int8_cli_args,
    describe_rwkv_int8_mode,
    resolve_rwkv_int8_lm_head_flags,
)


DEFAULT_GSM8K = os.path.join(
    os.path.dirname(__file__),
    "nanovllm",
    "eval_data",
    "GSM8K_100sample.jsonl",
)
BOXED_RE = re.compile(r"\\boxed\{([^{}]+)\}")
NUMBER_RE = re.compile(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?")
COT_MARKER = "<|completions_of_cot|>"
FINAL_ANSWER_MARKER = "<|final_answer|>"
ANSWER_STOP_MARKERS = ["}\\).", "}\\)", "}\\\\"]


@dataclass
class Sample:
    problem: str
    gold_answer: str
    prompt: str
    prompt_tokens: int
    expected_context: str | None = None
    cot_text: str = ""
    prediction_text: str = ""
    extracted_answer: str | None = None
    is_correct: bool = False
    output_tokens: int = 0


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


def build_legacy_prompt(problem: str) -> str:
    return (
        "User: Solve the following grade school math problem carefully.\n"
        "Keep the solution concise.\n"
        'End with a final line exactly in the form "#### <number>".\n'
        f"Problem: {problem}\n\n"
        "Assistant: <think>"
    )


def build_rwkv_rs_expected_context(problem: str, subject: str = "gsm8k") -> str:
    user_part = (
        f"You are a very talented expert in {subject}.\n"
        "Solve the problem and output the final answer in \\boxed{}.\n"
        f"Problem: {problem}"
    )
    assistant_part = (
        f"<think>{COT_MARKER}</think>\n"
        f"Therefore, the answer is \\(\\boxed{{{FINAL_ANSWER_MARKER}}}\\)."
    )
    return f"User: {user_part}\n\nAssistant: {assistant_part}"


def get_prompt_for_cot(expected_context: str) -> str:
    return expected_context.split(COT_MARKER, 1)[0]


def get_prompt_for_final_answer(expected_context: str, cot_text: str) -> str:
    return expected_context.replace(COT_MARKER, cot_text).split(FINAL_ANSWER_MARKER, 1)[0]


def render_full_prediction(expected_context: str, cot_text: str, final_answer_text: str) -> str:
    return (
        expected_context
        .replace(COT_MARKER, cot_text)
        .replace(FINAL_ANSWER_MARKER, final_answer_text)
    )


def trim_on_markers(text: str, markers: list[str]) -> str:
    end = None
    for marker in markers:
        idx = text.find(marker)
        if idx == -1:
            continue
        if end is None or idx < end:
            end = idx
    if end is None:
        return text
    return text[:end]


def normalize_number(value: str) -> str | None:
    text = value.strip()
    if not text:
        return None
    text = text.replace("\u2212", "-")
    text = text.replace("$", "")
    text = text.replace(",", "")
    text = text.rstrip(".")
    text = re.sub(r"(?i)\b(?:dollars?|usd|cents?)\b", "", text).strip()
    if not text:
        return None

    try:
        if re.fullmatch(r"[-+]?\d+\s*/\s*\d+", text):
            frac = Fraction(text.replace(" ", ""))
            if frac.denominator == 1:
                return str(frac.numerator)
            dec = Decimal(frac.numerator) / Decimal(frac.denominator)
        else:
            dec = Decimal(text)
    except (ValueError, ZeroDivisionError, InvalidOperation):
        return None

    normalized = format(dec, "f")
    if "." in normalized:
        normalized = normalized.rstrip("0").rstrip(".")
    if normalized in {"", "-0"}:
        normalized = "0"
    return normalized


def extract_number_from_snippet(snippet: str) -> str | None:
    text = snippet.strip()
    if not text:
        return None
    normalized = normalize_number(text)
    if normalized is not None:
        return normalized
    matches = NUMBER_RE.findall(text)
    if not matches:
        return None
    return normalize_number(matches[-1])


def extract_final_answer(text: str) -> str | None:
    for match in reversed(re.findall(r"####\s*([^\n\r]+)", text)):
        answer = extract_number_from_snippet(match)
        if answer is not None:
            return answer

    boxed = BOXED_RE.findall(text)
    for match in reversed(boxed):
        answer = extract_number_from_snippet(match)
        if answer is not None:
            return answer

    answer_markers = re.findall(
        r"(?is)(?:final answer|answer)\s*(?:is|=|:)?\s*([^\n\r]+)",
        text,
    )
    for match in reversed(answer_markers):
        answer = extract_number_from_snippet(match)
        if answer is not None:
            return answer

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return None
    last_line = lines[-1]
    if re.search(r"(?i)(?:final answer|answer)\s*(?:is|=|:)?", last_line):
        return extract_number_from_snippet(last_line)

    numeric_only = re.sub(r"[-+*/().,\s]", "", last_line)
    if numeric_only.isdigit():
        return extract_number_from_snippet(last_line)
    return None


def normalize_gold_answer(raw_answer: str) -> str:
    normalized = normalize_number(raw_answer)
    if normalized is not None:
        return normalized
    extracted = extract_final_answer(raw_answer)
    if extracted is not None:
        return extracted
    return raw_answer.strip()


def load_gsm8k(path: str, tokenizer, limit: int, prompt_style: str) -> list[Sample]:
    samples: list[Sample] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if prompt_style == "rwkv_rs_two_stage":
                expected_context = build_rwkv_rs_expected_context(row["problem"])
                prompt = get_prompt_for_cot(expected_context)
            else:
                expected_context = None
                prompt = build_legacy_prompt(row["problem"])
            samples.append(
                Sample(
                    problem=row["problem"],
                    gold_answer=normalize_gold_answer(row["answer"]),
                    prompt=prompt,
                    prompt_tokens=len(tokenizer.encode(prompt)),
                    expected_context=expected_context,
                )
            )
            if limit > 0 and len(samples) >= limit:
                break
    return samples


def load_gsm8k_hf(split: str, tokenizer, limit: int, prompt_style: str) -> list[Sample]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            "datasets is required for --hf-gsm8k. Install it or provide --gsm8k-path."
        ) from exc

    dataset = load_dataset("gsm8k", "main", split=split)
    samples: list[Sample] = []
    for row in dataset:
        problem = row["question"]
        answer = row["answer"]
        if prompt_style == "rwkv_rs_two_stage":
            expected_context = build_rwkv_rs_expected_context(problem)
            prompt = get_prompt_for_cot(expected_context)
        else:
            expected_context = None
            prompt = build_legacy_prompt(problem)
        samples.append(
            Sample(
                problem=problem,
                gold_answer=normalize_gold_answer(answer),
                prompt=prompt,
                prompt_tokens=len(tokenizer.encode(prompt)),
                expected_context=expected_context,
            )
        )
        if limit > 0 and len(samples) >= limit:
            break
    return samples


def write_predictions(path: str, samples: list[Sample]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for sample in samples:
            row = {
                "problem": sample.problem,
                "gold_answer": sample.gold_answer,
                "cot_text": sample.cot_text,
                "extracted_answer": sample.extracted_answer,
                "is_correct": sample.is_correct,
                "prompt_tokens": sample.prompt_tokens,
                "output_tokens": sample.output_tokens,
                "prediction_text": sample.prediction_text,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def evaluate_chunk(
    llm: LLM,
    chunk: list[Sample],
    tokenizer,
    prompt_style: str,
    sampling_params: SamplingParams,
    cot_max_tokens: int,
    answer_max_tokens: int,
) -> tuple[int, int]:
    total_prompt_tokens = sum(sample.prompt_tokens for sample in chunk)
    if prompt_style != "rwkv_rs_two_stage":
        outputs = llm.generate([sample.prompt for sample in chunk], sampling_params, use_tqdm=False)
        total_output_tokens = 0
        for sample, output in zip(chunk, outputs):
            sample.prediction_text = output["text"]
            sample.extracted_answer = extract_final_answer(sample.prediction_text)
            sample.is_correct = sample.extracted_answer == sample.gold_answer
            sample.output_tokens = len(output["token_ids"])
            total_output_tokens += sample.output_tokens
        return total_prompt_tokens, total_output_tokens

    cot_sampling_params = SamplingParams(
        temperature=sampling_params.temperature,
        ignore_eos=sampling_params.ignore_eos,
        max_tokens=cot_max_tokens,
    )
    answer_sampling_params = SamplingParams(
        temperature=sampling_params.temperature,
        ignore_eos=sampling_params.ignore_eos,
        max_tokens=answer_max_tokens,
    )

    cot_outputs = llm.generate([sample.prompt for sample in chunk], cot_sampling_params, use_tqdm=False)
    total_output_tokens = 0
    answer_prompts: list[str] = []
    for sample, output in zip(chunk, cot_outputs):
        sample.cot_text = trim_on_markers(output["text"], ["</think>"]).strip()
        sample.output_tokens = len(output["token_ids"])
        total_output_tokens += sample.output_tokens
        answer_prompt = get_prompt_for_final_answer(sample.expected_context, sample.cot_text)
        total_prompt_tokens += len(tokenizer.encode(answer_prompt))
        answer_prompts.append(answer_prompt)

    answer_outputs = llm.generate(answer_prompts, answer_sampling_params, use_tqdm=False)
    for sample, output in zip(chunk, answer_outputs):
        final_answer_text = trim_on_markers(output["text"], ANSWER_STOP_MARKERS).strip()
        sample.prediction_text = render_full_prediction(
            sample.expected_context,
            sample.cot_text,
            final_answer_text,
        )
        sample.extracted_answer = extract_final_answer(sample.prediction_text)
        sample.is_correct = sample.extracted_answer == sample.gold_answer
        sample.output_tokens += len(output["token_ids"])
        total_output_tokens += len(output["token_ids"])

    return total_prompt_tokens, total_output_tokens


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-pth", required=True)
    parser.add_argument("--gsm8k-path", default=DEFAULT_GSM8K)
    parser.add_argument(
        "--hf-gsm8k",
        action="store_true",
        help="Load the official gsm8k dataset via datasets instead of a local jsonl file.",
    )
    parser.add_argument(
        "--hf-split",
        default="test",
        help="datasets split to use with --hf-gsm8k. Default: test",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--prompt-style",
        choices=["legacy", "rwkv_rs_two_stage"],
        default="legacy",
    )
    parser.add_argument("--max-tokens", type=int, default=768)
    parser.add_argument("--cot-max-tokens", type=int, default=512)
    parser.add_argument("--answer-max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.97)
    parser.add_argument("--max-state-slots", type=int, default=-1)
    parser.add_argument("--rwkv-state-cache-safety-reserve-slots", type=int, default=0)
    parser.add_argument("--rwkv-prefill-token-budget", type=int, default=2048)
    parser.add_argument("--rwkv-prefill-max-batch-size", type=int, default=128)
    parser.add_argument("--rwkv-state-cache-enable", action="store_true")
    add_rwkv_int8_cli_args(parser)
    parser.add_argument("--max-num-seqs", type=int, default=32)
    parser.add_argument("--max-num-batched-tokens", type=int, default=32768)
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
    parser.add_argument("--print-interval", type=int, default=10)
    parser.add_argument("--show-samples", type=int, default=3)
    parser.add_argument("--predictions-out")
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
    if args.hf_gsm8k:
        samples = load_gsm8k_hf(args.hf_split, tokenizer, args.limit, args.prompt_style)
    else:
        samples = load_gsm8k(args.gsm8k_path, tokenizer, args.limit, args.prompt_style)
    if not samples:
        raise SystemExit("no GSM8K samples loaded")

    model_dir = ensure_model_dir(args.model_pth)
    llm = LLM(
        model_dir,
        enforce_eager=args.enforce_eager,
        tensor_parallel_size=1,
        max_num_seqs=max(args.max_num_seqs, args.batch_size),
        max_num_batched_tokens=max(args.max_num_batched_tokens, args.batch_size * 1024),
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
    sampling_params = SamplingParams(
        temperature=args.temperature,
        ignore_eos=False,
        max_tokens=args.max_tokens,
    )

    total_correct = 0
    total_output_tokens = 0
    total_prompt_tokens = 0
    total_extracted = 0

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for start in range(0, len(samples), args.batch_size):
        chunk = samples[start:start + args.batch_size]
        prompt_tokens, output_tokens = evaluate_chunk(
            llm,
            chunk,
            tokenizer,
            args.prompt_style,
            sampling_params,
            args.cot_max_tokens,
            args.answer_max_tokens,
        )
        total_output_tokens += output_tokens
        total_prompt_tokens += prompt_tokens
        total_correct += sum(1 for sample in chunk if sample.is_correct)
        total_extracted += sum(1 for sample in chunk if sample.extracted_answer is not None)
        done = start + len(chunk)
        if done % args.print_interval == 0 or done == len(samples):
            acc = total_correct / done * 100
            extract_rate = total_extracted / done * 100
            print(
                f"done={done},acc={acc:.2f},extract_rate={extract_rate:.2f},"
                f"prompt_tokens={total_prompt_tokens},output_tokens={total_output_tokens}"
            )
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0

    if args.predictions_out:
        write_predictions(args.predictions_out, samples)

    for idx, sample in enumerate(samples[:args.show_samples]):
        print(
            f"sample={idx},gold={sample.gold_answer},pred={sample.extracted_answer},"
            f"correct={int(sample.is_correct)}"
        )
        print(f"problem={sample.problem}")
        print(f"prediction={sample.prediction_text!r}")

    acc = total_correct / len(samples) * 100
    extract_rate = total_extracted / len(samples) * 100
    prompt_tps = total_prompt_tokens / dt
    output_tps = total_output_tokens / dt
    print(
        f"final_examples={len(samples)},acc={acc:.2f},extract_rate={extract_rate:.2f},"
        f"prompt_tokens={total_prompt_tokens},output_tokens={total_output_tokens},"
        f"time_s={dt:.4f},prompt_tps={prompt_tps:.2f},output_tps={output_tps:.2f},"
        f"batch_size={args.batch_size},enforce_eager={int(args.enforce_eager)},"
        f"rwkv_quant_int8={int(args.rwkv_quant_int8)},rwkv_mode={rwkv_mode},"
        f"prompt_style={args.prompt_style},hf_gsm8k={int(args.hf_gsm8k)}"
    )
    llm.exit()


if __name__ == "__main__":
    main()
