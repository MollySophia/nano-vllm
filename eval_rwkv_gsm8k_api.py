#!/usr/bin/env python3
import argparse
import asyncio
import json
import os
import time
from dataclasses import dataclass
from typing import Any

import httpx

from eval_rwkv_gsm8k import (
    ANSWER_STOP_MARKERS,
    DEFAULT_GSM8K,
    build_legacy_prompt,
    build_rwkv_rs_expected_context,
    extract_final_answer,
    get_prompt_for_cot,
    get_prompt_for_final_answer,
    normalize_gold_answer,
    render_full_prediction,
    trim_on_markers,
)

try:
    from datasets import load_dataset
except ImportError as exc:  # pragma: no cover - exercised only when dependency is missing.
    load_dataset = None
    DATASETS_IMPORT_ERROR = exc
else:
    DATASETS_IMPORT_ERROR = None


@dataclass
class Sample:
    problem: str
    gold_answer: str
    prompt: str
    expected_context: str | None = None
    cot_text: str = ""
    prediction_text: str = ""
    extracted_answer: str | None = None
    is_correct: bool = False
    prompt_tokens: int = 0
    output_tokens: int = 0


def require_datasets() -> None:
    if load_dataset is not None:
        return
    raise SystemExit(
        "eval_rwkv_gsm8k_api.py requires the HuggingFace `datasets` package "
        "when --hf-gsm8k is used.\n"
        f"Original import error: {DATASETS_IMPORT_ERROR}"
    )


def request_headers(api_key: str | None) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


class GSM8KAPIClient:
    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str | None,
        timeout_s: float,
        max_connections: int,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.headers = request_headers(api_key)
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout_s),
            limits=httpx.Limits(
                max_connections=max_connections,
                max_keepalive_connections=max_connections,
            ),
        )

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self._client.aclose()

    async def complete(
        self,
        *,
        prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> dict[str, Any]:
        response = await self._client.post(
            f"{self.base_url}/v1/completions",
            headers=self.headers,
            json={
                "model": self.model,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
        )
        if response.status_code != 200:
            raise RuntimeError(
                f"/v1/completions returned {response.status_code}: {response.text[:500]}"
            )
        return response.json()


def load_gsm8k_text(path: str, limit: int, prompt_style: str) -> list[Sample]:
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
                    expected_context=expected_context,
                )
            )
            if limit > 0 and len(samples) >= limit:
                break
    return samples


def load_gsm8k_hf_text(split: str, limit: int, prompt_style: str) -> list[Sample]:
    require_datasets()
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
                expected_context=expected_context,
            )
        )
        if limit > 0 and len(samples) >= limit:
            break
    return samples


def completion_text_and_usage(response_body: dict[str, Any]) -> tuple[str, int, int]:
    choices = response_body.get("choices") or []
    usage = response_body.get("usage") or {}
    if len(choices) != 1:
        raise RuntimeError("Expected exactly one completion choice.")
    prompt_tokens = int(usage.get("prompt_tokens", 0))
    completion_tokens = int(usage.get("completion_tokens", 0))
    return str(choices[0].get("text") or ""), prompt_tokens, completion_tokens


def apply_single_stage_response(sample: Sample, response_body: dict[str, Any]) -> tuple[int, int]:
    prediction_text, prompt_tokens, completion_tokens = completion_text_and_usage(response_body)
    sample.prediction_text = prediction_text
    sample.extracted_answer = extract_final_answer(sample.prediction_text)
    sample.is_correct = sample.extracted_answer == sample.gold_answer
    sample.prompt_tokens += prompt_tokens
    sample.output_tokens += completion_tokens
    return prompt_tokens, completion_tokens


def apply_two_stage_responses(
    sample: Sample,
    cot_response: dict[str, Any],
    answer_response: dict[str, Any],
) -> tuple[int, int]:
    cot_text, cot_prompt_tokens, cot_completion_tokens = completion_text_and_usage(cot_response)
    sample.cot_text = trim_on_markers(cot_text, ["</think>"]).strip()

    answer_text, answer_prompt_tokens, answer_completion_tokens = completion_text_and_usage(
        answer_response
    )
    final_answer_text = trim_on_markers(answer_text, ANSWER_STOP_MARKERS).strip()
    sample.prediction_text = render_full_prediction(
        sample.expected_context,
        sample.cot_text,
        final_answer_text,
    )
    sample.extracted_answer = extract_final_answer(sample.prediction_text)
    sample.is_correct = sample.extracted_answer == sample.gold_answer
    sample.prompt_tokens += cot_prompt_tokens + answer_prompt_tokens
    sample.output_tokens += cot_completion_tokens + answer_completion_tokens
    return cot_prompt_tokens + answer_prompt_tokens, cot_completion_tokens + answer_completion_tokens


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


async def evaluate_chunk(
    api: GSM8KAPIClient,
    chunk: list[Sample],
    *,
    prompt_style: str,
    temperature: float,
    max_tokens: int,
    cot_max_tokens: int,
    answer_max_tokens: int,
) -> tuple[int, int]:
    if prompt_style != "rwkv_rs_two_stage":
        responses = await asyncio.gather(
            *[
                api.complete(
                    prompt=sample.prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                for sample in chunk
            ]
        )
        total_prompt_tokens = 0
        total_output_tokens = 0
        for sample, response_body in zip(chunk, responses, strict=True):
            prompt_tokens, output_tokens = apply_single_stage_response(sample, response_body)
            total_prompt_tokens += prompt_tokens
            total_output_tokens += output_tokens
        return total_prompt_tokens, total_output_tokens

    cot_responses = await asyncio.gather(
        *[
            api.complete(
                prompt=sample.prompt,
                max_tokens=cot_max_tokens,
                temperature=temperature,
            )
            for sample in chunk
        ]
    )
    answer_prompts: list[str] = []
    for sample, cot_response in zip(chunk, cot_responses, strict=True):
        cot_text, _, _ = completion_text_and_usage(cot_response)
        sample.cot_text = trim_on_markers(cot_text, ["</think>"]).strip()
        answer_prompts.append(get_prompt_for_final_answer(sample.expected_context, sample.cot_text))

    answer_responses = await asyncio.gather(
        *[
            api.complete(
                prompt=answer_prompt,
                max_tokens=answer_max_tokens,
                temperature=temperature,
            )
            for answer_prompt in answer_prompts
        ]
    )

    total_prompt_tokens = 0
    total_output_tokens = 0
    for sample, cot_response, answer_response in zip(
        chunk,
        cot_responses,
        answer_responses,
        strict=True,
    ):
        prompt_tokens, output_tokens = apply_two_stage_responses(
            sample,
            cot_response,
            answer_response,
        )
        total_prompt_tokens += prompt_tokens
        total_output_tokens += output_tokens
    return total_prompt_tokens, total_output_tokens


async def evaluate(args) -> None:
    if args.hf_gsm8k:
        samples = load_gsm8k_hf_text(args.hf_split, args.limit, args.prompt_style)
    else:
        samples = load_gsm8k_text(args.gsm8k_path, args.limit, args.prompt_style)
    if not samples:
        raise SystemExit("no GSM8K samples loaded")

    total_correct = 0
    total_output_tokens = 0
    total_prompt_tokens = 0
    total_extracted = 0

    async with GSM8KAPIClient(
        base_url=args.base_url,
        model=args.model,
        api_key=args.api_key,
        timeout_s=args.request_timeout,
        max_connections=max(args.max_connections, args.batch_size * 4),
    ) as api:
        t0 = time.perf_counter()
        for start in range(0, len(samples), args.batch_size):
            chunk = samples[start:start + args.batch_size]
            prompt_tokens, output_tokens = await evaluate_chunk(
                api,
                chunk,
                prompt_style=args.prompt_style,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                cot_max_tokens=args.cot_max_tokens,
                answer_max_tokens=args.answer_max_tokens,
            )
            total_output_tokens += output_tokens
            total_prompt_tokens += prompt_tokens
            total_correct += sum(1 for sample in chunk if sample.is_correct)
            total_extracted += sum(1 for sample in chunk if sample.extracted_answer is not None)
            done = start + len(chunk)
            if done % args.print_interval == 0 or done == len(samples):
                acc = total_correct / done * 100.0
                extract_rate = total_extracted / done * 100.0
                print(
                    f"done={done},acc={acc:.2f},extract_rate={extract_rate:.2f},"
                    f"prompt_tokens={total_prompt_tokens},output_tokens={total_output_tokens}",
                    flush=True,
                )
        dt = time.perf_counter() - t0

    if args.predictions_out:
        write_predictions(args.predictions_out, samples)

    for idx, sample in enumerate(samples[:args.show_samples]):
        print(
            f"sample={idx},gold={sample.gold_answer},pred={sample.extracted_answer},"
            f"correct={int(sample.is_correct)}",
            flush=True,
        )
        print(f"problem={sample.problem}", flush=True)
        print(f"prediction={sample.prediction_text!r}", flush=True)

    acc = total_correct / len(samples) * 100.0
    extract_rate = total_extracted / len(samples) * 100.0
    prompt_tps = total_prompt_tokens / dt
    output_tps = total_output_tokens / dt
    print(
        f"final_examples={len(samples)},acc={acc:.2f},extract_rate={extract_rate:.2f},"
        f"prompt_tokens={total_prompt_tokens},output_tokens={total_output_tokens},"
        f"time_s={dt:.4f},prompt_tps={prompt_tps:.2f},output_tps={output_tps:.2f},"
        f"batch_size={args.batch_size},prompt_style={args.prompt_style},"
        f"hf_gsm8k={int(args.hf_gsm8k)},base_url={args.base_url},model={args.model}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key")
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
    parser.add_argument("--print-interval", type=int, default=10)
    parser.add_argument("--show-samples", type=int, default=3)
    parser.add_argument("--predictions-out")
    parser.add_argument("--request-timeout", type=float, default=300.0)
    parser.add_argument("--max-connections", type=int, default=512)
    args = parser.parse_args()
    asyncio.run(evaluate(args))


if __name__ == "__main__":
    main()
