#!/usr/bin/env python3
import argparse
import asyncio
import json
import os
import random
import time
from dataclasses import dataclass
from typing import Any

import httpx

from eval_rwkv_lambada_api import RWKV_EOD_TEXT
from eval_rwkv_mmlu import CHOICES, DEFAULT_MMLU, build_prompt

try:
    from datasets import load_from_disk
except ImportError as exc:  # pragma: no cover - exercised only when dependency is missing.
    load_from_disk = None
    DATASETS_IMPORT_ERROR = exc
else:
    DATASETS_IMPORT_ERROR = None


@dataclass
class Sample:
    question: str
    choices: list[str]
    subject: str
    answer: int
    prompt: str
    prompt_tokens: int = 0
    predicted: int | None = None
    is_correct: bool = False


def resolve_mmlu_path(path: str) -> str:
    if os.path.isdir(path):
        return path
    raise SystemExit(f"MMLU dataset path does not exist: {path}")


def require_datasets() -> None:
    if load_from_disk is not None:
        return
    raise SystemExit(
        "eval_rwkv_mmlu_api.py requires the HuggingFace `datasets` package because it "
        "loads a local on-disk MMLU dataset snapshot.\n"
        f"Original import error: {DATASETS_IMPORT_ERROR}"
    )


def request_headers(api_key: str | None) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


class MMLUAPIClient:
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

    async def tokenize(self, text: str) -> dict[str, Any]:
        response = await self._client.post(
            f"{self.base_url}/v1/tokenize",
            headers=self.headers,
            json={
                "model": self.model,
                "text": text,
            },
        )
        if response.status_code != 200:
            raise RuntimeError(
                f"/v1/tokenize returned {response.status_code}: {response.text[:500]}"
            )
        return response.json()

    async def score_text(self, prompt: str) -> dict[str, Any]:
        response = await self._client.post(
            f"{self.base_url}/v1/completions",
            headers=self.headers,
            json={
                "model": self.model,
                "prompt": prompt,
                "max_tokens": 0,
                "echo": True,
                "logprobs": 1,
                "temperature": 0.0,
            },
        )
        if response.status_code != 200:
            raise RuntimeError(
                f"/v1/completions returned {response.status_code}: {response.text[:500]}"
            )
        return response.json()


def load_mmlu_text_samples(
    path: str,
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
        samples.append(
            Sample(
                question=row["question"],
                choices=choices,
                subject=subject,
                answer=gt,
                prompt=prompt,
            )
        )
        if limit > 0 and len(samples) >= limit:
            break
    return samples


def build_scoring_prompt(prompt: str, choice_text: str) -> str:
    return f"{RWKV_EOD_TEXT}{prompt}{choice_text}"


def extract_choice_logprob_and_prompt_tokens(response_body: dict[str, Any]) -> tuple[float, int]:
    choices = response_body.get("choices") or []
    usage = response_body.get("usage") or {}
    if len(choices) != 1:
        raise RuntimeError("Expected exactly one completion choice.")
    logprobs = choices[0].get("logprobs")
    if not isinstance(logprobs, dict):
        raise RuntimeError("Completion response did not include logprobs.")
    token_logprobs = list(logprobs.get("token_logprobs") or [])
    if not token_logprobs or token_logprobs[-1] is None:
        raise RuntimeError("Final choice token logprob is missing from completion response.")
    return float(token_logprobs[-1]), int(usage.get("prompt_tokens", 0))


def apply_choice_scores(
    sample: Sample,
    *,
    choice_scores: list[float],
    logical_prompt_tokens: int,
) -> None:
    if len(choice_scores) != len(CHOICES):
        raise RuntimeError(f"Expected {len(CHOICES)} choice scores, got {len(choice_scores)}")
    sample.prompt_tokens = logical_prompt_tokens
    sample.predicted = max(range(len(choice_scores)), key=lambda idx: choice_scores[idx])
    sample.is_correct = sample.predicted == sample.answer


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


async def verify_choice_tokenization(api: MMLUAPIClient) -> list[int]:
    responses = await asyncio.gather(*[api.tokenize(choice_text) for choice_text in CHOICES])
    counts = [int(response["count"]) for response in responses]
    if not all(count == 1 for count in counts):
        raise SystemExit(f"Expected single-token MMLU choices, got counts: {counts}")
    return counts


async def evaluate_chunk(
    api: MMLUAPIClient,
    chunk: list[Sample],
    *,
    choice_token_counts: list[int],
) -> tuple[int, int, int]:
    total_prompt_tokens = 0
    total_api_prompt_tokens = 0
    requests = []
    for sample in chunk:
        for choice_text in CHOICES:
            requests.append(api.score_text(build_scoring_prompt(sample.prompt, choice_text)))
    responses = await asyncio.gather(*requests)

    offset = 0
    for sample in chunk:
        choice_scores: list[float] = []
        prompt_tokens_with_choice: list[int] = []
        for choice_idx, choice_text in enumerate(CHOICES):
            response_body = responses[offset]
            offset += 1
            logprob, api_prompt_tokens = extract_choice_logprob_and_prompt_tokens(response_body)
            choice_scores.append(logprob)
            prompt_tokens_with_choice.append(api_prompt_tokens)
            total_api_prompt_tokens += api_prompt_tokens
        logical_prompt_tokens = prompt_tokens_with_choice[0] - choice_token_counts[0]
        apply_choice_scores(
            sample,
            choice_scores=choice_scores,
            logical_prompt_tokens=logical_prompt_tokens,
        )
        total_prompt_tokens += logical_prompt_tokens
    return len(chunk), total_prompt_tokens, total_api_prompt_tokens


async def evaluate(args) -> None:
    subjects = set(args.subject) if args.subject else None
    samples = load_mmlu_text_samples(
        args.mmlu_path,
        args.limit,
        args.shuffle_choices,
        args.seed,
        subjects,
    )
    if not samples:
        raise SystemExit("No MMLU samples selected.")

    print("format_example:", flush=True)
    print("-" * 80, flush=True)
    print(samples[0].prompt, flush=True)
    print("-" * 80, flush=True)

    total_examples = 0
    total_prompt_tokens = 0
    total_api_prompt_tokens = 0
    total_correct = 0

    async with MMLUAPIClient(
        base_url=args.base_url,
        model=args.model,
        api_key=args.api_key,
        timeout_s=args.request_timeout,
        max_connections=max(args.max_connections, args.batch_size * len(CHOICES)),
    ) as api:
        choice_token_counts = await verify_choice_tokenization(api)
        t0 = time.perf_counter()
        for start in range(0, len(samples), args.batch_size):
            chunk = samples[start:start + args.batch_size]
            n_examples, n_prompt_tokens, n_api_prompt_tokens = await evaluate_chunk(
                api,
                chunk,
                choice_token_counts=choice_token_counts,
            )
            total_examples += n_examples
            total_prompt_tokens += n_prompt_tokens
            total_api_prompt_tokens += n_api_prompt_tokens
            total_correct += sum(1 for sample in chunk if sample.is_correct)

            if total_examples % args.print_interval == 0 or total_examples == len(samples):
                acc = total_correct / total_examples * 100.0
                print(
                    f"done={total_examples},acc={acc:.2f},"
                    f"prompt_tokens={total_prompt_tokens},api_prompt_tokens={total_api_prompt_tokens}",
                    flush=True,
                )
        dt = time.perf_counter() - t0

    if args.predictions_path:
        write_predictions(args.predictions_path, samples)

    acc = total_correct / total_examples * 100.0
    prompt_tps = total_prompt_tokens / dt
    api_prompt_tps = total_api_prompt_tokens / dt
    examples_per_s = total_examples / dt
    print(
        f"final_examples={total_examples},acc={acc:.2f},"
        f"prompt_tokens={total_prompt_tokens},api_prompt_tokens={total_api_prompt_tokens},"
        f"time_s={dt:.4f},prompt_tps={prompt_tps:.2f},api_prompt_tps={api_prompt_tps:.2f},"
        f"examples_per_s={examples_per_s:.2f},batch_size={args.batch_size},"
        f"shuffle_choices={int(args.shuffle_choices)},base_url={args.base_url},model={args.model}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key")
    parser.add_argument("--mmlu-path", default=DEFAULT_MMLU)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--shuffle-choices", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--subject", action="append", default=None)
    parser.add_argument("--print-interval", type=int, default=512)
    parser.add_argument("--predictions-path", default=None)
    parser.add_argument("--request-timeout", type=float, default=300.0)
    parser.add_argument("--max-connections", type=int, default=2048)
    args = parser.parse_args()
    asyncio.run(evaluate(args))


if __name__ == "__main__":
    main()
