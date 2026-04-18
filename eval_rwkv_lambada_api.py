#!/usr/bin/env python3
import argparse
import asyncio
import json
import math
import os
import time
from dataclasses import dataclass
from typing import Any

import httpx


DEFAULT_LAMBADA = os.path.join(
    os.path.dirname(__file__),
    "nanovllm",
    "eval_data",
    "lambada_test.jsonl",
)
RWKV_EOD_TEXT = "<|rwkv_end_of_text|>"


@dataclass
class RawSample:
    prefix_text: str
    target_text: str


@dataclass
class TokenizedSample:
    prefix_text: str
    target_text: str
    prefix_token_ids: list[int]
    target_token_ids: list[int]
    target_tokens: list[str]
    logprob_sum: float = 0.0
    correct: bool = True


def load_lambada_texts(path: str, limit: int, pad_eod: bool) -> list[RawSample]:
    samples: list[RawSample] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)
            prefix, last_word = doc["text"].rsplit(" ", 1)
            prefix_text = f"{RWKV_EOD_TEXT}{prefix}" if pad_eod else prefix
            samples.append(
                RawSample(
                    prefix_text=prefix_text,
                    target_text=" " + last_word,
                )
            )
            if limit > 0 and len(samples) >= limit:
                break
    return samples


def request_headers(api_key: str | None) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


class LambadaAPIClient:
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

    async def _post_json(self, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        response = await self._client.post(
            f"{self.base_url}{endpoint}",
            headers=self.headers,
            json=payload,
        )
        if response.status_code != 200:
            raise RuntimeError(
                f"{endpoint} returned {response.status_code}: {response.text[:500]}"
            )
        return response.json()

    async def tokenize(self, text: str) -> dict[str, Any]:
        return await self._post_json(
            "/v1/tokenize",
            {
                "model": self.model,
                "text": text,
            },
        )

    async def score_prompt(self, prompt_token_ids: list[int]) -> dict[str, Any]:
        return await self._post_json(
            "/v1/completions",
            {
                "model": self.model,
                "prompt_token_ids": prompt_token_ids,
                "max_tokens": 0,
                "echo": True,
                "logprobs": 1,
                "temperature": 0.0,
            },
        )


async def prepare_tokenized_sample(api: LambadaAPIClient, raw: RawSample) -> TokenizedSample:
    prefix_body, target_body = await asyncio.gather(
        api.tokenize(raw.prefix_text),
        api.tokenize(raw.target_text),
    )
    prefix_token_ids = list(prefix_body["token_ids"])
    target_token_ids = list(target_body["token_ids"])
    if not prefix_token_ids:
        raise RuntimeError("LAMBADA prefix tokenized to an empty input.")
    if not target_token_ids:
        raise RuntimeError("LAMBADA target tokenized to an empty input.")
    return TokenizedSample(
        prefix_text=raw.prefix_text,
        target_text=raw.target_text,
        prefix_token_ids=prefix_token_ids,
        target_token_ids=target_token_ids,
        target_tokens=list(target_body["tokens"]),
    )


def score_sample_from_completion(
    sample: TokenizedSample,
    response_body: dict[str, Any],
) -> tuple[float, bool]:
    choices = response_body.get("choices") or []
    if len(choices) != 1:
        raise RuntimeError("Expected exactly one completion choice.")
    choice = choices[0]
    logprobs = choice.get("logprobs")
    if not isinstance(logprobs, dict):
        raise RuntimeError("Completion response did not include logprobs.")

    token_logprobs = list(logprobs.get("token_logprobs") or [])
    top_logprobs = list(logprobs.get("top_logprobs") or [])
    expected_len = len(sample.prefix_token_ids) + len(sample.target_token_ids)
    if len(token_logprobs) != expected_len or len(top_logprobs) != expected_len:
        raise RuntimeError(
            "Completion logprobs length mismatch: "
            f"expected {expected_len}, got token_logprobs={len(token_logprobs)}, "
            f"top_logprobs={len(top_logprobs)}"
        )

    target_count = len(sample.target_token_ids)
    target_logprobs = token_logprobs[-target_count:]
    target_top_logprobs = top_logprobs[-target_count:]

    logprob_sum = 0.0
    correct = True
    for gold_piece, token_logprob, top_entry in zip(
        sample.target_tokens,
        target_logprobs,
        target_top_logprobs,
        strict=True,
    ):
        if token_logprob is None:
            raise RuntimeError("Target token logprob is missing from completion response.")
        logprob_sum += float(token_logprob)
        if not isinstance(top_entry, dict) or not top_entry:
            correct = False
            continue
        predicted_piece = next(iter(top_entry.keys()))
        if predicted_piece != gold_piece:
            correct = False
    return logprob_sum, correct


async def gather_with_progress(
    coroutines: list[Any],
) -> list[Any]:
    return await asyncio.gather(*coroutines)


async def evaluate(args) -> None:
    raw_samples = load_lambada_texts(args.lambada_path, args.limit, args.pad_eod)
    if not raw_samples:
        raise SystemExit("No Lambada samples loaded.")

    total_examples = 0
    total_target_tokens = 0
    total_logprob = 0.0
    total_correct = 0

    async with LambadaAPIClient(
        base_url=args.base_url,
        model=args.model,
        api_key=args.api_key,
        timeout_s=args.request_timeout,
        max_connections=max(args.max_connections, args.batch_size * 3),
    ) as api:
        t0 = time.perf_counter()
        for start in range(0, len(raw_samples), args.batch_size):
            raw_chunk = raw_samples[start:start + args.batch_size]
            chunk = await gather_with_progress(
                [prepare_tokenized_sample(api, raw) for raw in raw_chunk]
            )
            responses = await gather_with_progress(
                [
                    api.score_prompt(sample.prefix_token_ids + sample.target_token_ids)
                    for sample in chunk
                ]
            )

            for sample, response_body in zip(chunk, responses, strict=True):
                sample.logprob_sum, sample.correct = score_sample_from_completion(
                    sample,
                    response_body,
                )

            total_examples += len(chunk)
            total_target_tokens += sum(len(sample.target_token_ids) for sample in chunk)
            total_logprob += sum(sample.logprob_sum for sample in chunk)
            total_correct += sum(1 for sample in chunk if sample.correct)

            if total_examples % args.print_interval == 0 or total_examples == len(raw_samples):
                ppl = math.exp(-total_logprob / total_examples)
                acc = total_correct / total_examples * 100.0
                print(
                    f"done={total_examples},ppl={ppl:.4f},acc={acc:.2f},"
                    f"target_tokens={total_target_tokens}"
                , flush=True)

        dt = time.perf_counter() - t0

    ppl = math.exp(-total_logprob / total_examples)
    acc = total_correct / total_examples * 100.0
    target_tps = total_target_tokens / dt
    print(
        f"final_examples={total_examples},ppl={ppl:.4f},acc={acc:.2f},"
        f"target_tokens={total_target_tokens},time_s={dt:.4f},target_tps={target_tps:.2f},"
        f"batch_size={args.batch_size},pad_eod={int(args.pad_eod)},"
        f"base_url={args.base_url},model={args.model}"
    , flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key")
    parser.add_argument("--lambada-path", default=DEFAULT_LAMBADA)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--pad-eod",
        dest="pad_eod",
        action="store_true",
        help="Prepend <|rwkv_end_of_text|> to each prefix before API tokenization. Default: enabled.",
    )
    parser.add_argument(
        "--no-pad-eod",
        dest="pad_eod",
        action="store_false",
        help="Disable the leading end-of-text prefix.",
    )
    parser.add_argument("--print-interval", type=int, default=1000)
    parser.add_argument("--request-timeout", type=float, default=120.0)
    parser.add_argument("--max-connections", type=int, default=512)
    parser.set_defaults(pad_eod=True)
    args = parser.parse_args()
    asyncio.run(evaluate(args))


if __name__ == "__main__":
    main()
