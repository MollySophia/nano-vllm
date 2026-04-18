#!/usr/bin/env python3
import argparse
import asyncio
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import httpx

import benchmark_openai_api as api_bench
from eval_rwkv_gsm8k import (
    DEFAULT_GSM8K,
    build_legacy_prompt,
    extract_final_answer,
    normalize_gold_answer,
)

try:
    from datasets import load_dataset
except ImportError as exc:  # pragma: no cover - exercised only when dependency is missing.
    load_dataset = None
    DATASETS_IMPORT_ERROR = exc
else:
    DATASETS_IMPORT_ERROR = None


@dataclass
class GSM8KSample:
    sample_index: int
    problem: str
    gold_answer: str
    prompt: str


@dataclass(kw_only=True)
class GSM8KRequestMetrics(api_bench.RequestMetrics):
    sample_index: int
    gold_answer: str | None
    extracted_answer: str | None
    is_correct: bool | None
    prompt_preview: str | None = None


class SampleSource:
    def __init__(self, samples: list[GSM8KSample], seed: int, random_sample: bool):
        if not samples:
            raise ValueError("No GSM8K samples available for the benchmark.")
        self.samples = list(samples)
        self.random_sample = random_sample
        self.rng = __import__("random").Random(seed)

    def sample(self, request_index: int) -> GSM8KSample:
        if self.random_sample:
            return self.samples[self.rng.randrange(len(self.samples))]
        return self.samples[request_index % len(self.samples)]


def require_datasets() -> None:
    if load_dataset is not None:
        return
    raise SystemExit(
        "benchmark_gsm8k_api.py requires the HuggingFace `datasets` package "
        "when --hf-gsm8k is used.\n"
        f"Original import error: {DATASETS_IMPORT_ERROR}"
    )


def load_gsm8k_samples(path: str, limit: int) -> list[GSM8KSample]:
    samples: list[GSM8KSample] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            samples.append(
                GSM8KSample(
                    sample_index=len(samples),
                    problem=row["problem"],
                    gold_answer=normalize_gold_answer(row["answer"]),
                    prompt=build_legacy_prompt(row["problem"]),
                )
            )
            if limit > 0 and len(samples) >= limit:
                break
    return samples


def load_gsm8k_hf_samples(split: str, limit: int) -> list[GSM8KSample]:
    require_datasets()
    dataset = load_dataset("gsm8k", "main", split=split)
    samples: list[GSM8KSample] = []
    for row in dataset:
        samples.append(
            GSM8KSample(
                sample_index=len(samples),
                problem=row["question"],
                gold_answer=normalize_gold_answer(row["answer"]),
                prompt=build_legacy_prompt(row["question"]),
            )
        )
        if limit > 0 and len(samples) >= limit:
            break
    return samples


def build_payload(
    *,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    stream: bool,
) -> dict[str, Any]:
    return {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": stream,
    }


def score_response_text(text: str, gold_answer: str) -> tuple[str | None, bool]:
    extracted = extract_final_answer(text)
    return extracted, extracted == gold_answer


async def run_sync_request(
    *,
    client: httpx.AsyncClient,
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    request_index: int,
    worker_id: int,
    sample: GSM8KSample,
) -> GSM8KRequestMetrics:
    started_at = time.perf_counter()
    try:
        response = await client.post(url, headers=headers, json=payload)
        latency_s = time.perf_counter() - started_at
        body_text = response.text
        response_bytes = len(body_text.encode("utf-8"))
        request_id = response.headers.get("x-request-id")
        response_headers = response.headers
        if response.status_code != 200:
            return GSM8KRequestMetrics(
                request_index=request_index,
                worker_id=worker_id,
                ok=False,
                status_code=response.status_code,
                error=body_text[:500],
                latency_s=latency_s,
                ttft_s=None,
                response_bytes=response_bytes,
                response_chars=len(body_text),
                request_id=request_id,
                finish_reason=None,
                prompt_tokens=api_bench.parse_int_header(response_headers, "x-nanovllm-prompt-tokens"),
                completion_tokens=api_bench.parse_int_header(response_headers, "x-nanovllm-completion-tokens"),
                total_tokens=None,
                server_processing_ms=api_bench.parse_float_header(response_headers, "openai-processing-ms"),
                server_queue_wait_ms=api_bench.parse_float_header(response_headers, "x-nanovllm-queue-wait-ms"),
                server_ttft_ms=api_bench.parse_float_header(response_headers, "x-nanovllm-ttft-ms"),
                server_generation_ms=api_bench.parse_float_header(response_headers, "x-nanovllm-generation-ms"),
                server_total_ms=api_bench.parse_float_header(response_headers, "x-nanovllm-total-ms"),
                server_output_tps=api_bench.parse_float_header(response_headers, "x-nanovllm-output-tokens-per-second"),
                server_decode_tps=api_bench.parse_float_header(response_headers, "x-nanovllm-decode-tokens-per-second"),
                server_request_parse_ms=api_bench.parse_float_header(response_headers, "x-nanovllm-request-parse-ms"),
                server_request_setup_ms=api_bench.parse_float_header(response_headers, "x-nanovllm-request-setup-ms"),
                server_response_build_ms=api_bench.parse_float_header(response_headers, "x-nanovllm-response-build-ms"),
                server_app_ms=api_bench.parse_float_header(response_headers, "x-nanovllm-server-app-ms"),
                sample_index=sample.sample_index,
                gold_answer=sample.gold_answer,
                extracted_answer=None,
                is_correct=None,
                prompt_preview=sample.problem[:120],
            )
        body = response.json()
        _, text, finish_reason = api_bench.parse_sync_body("completions", body)
        usage = body.get("usage") or {}
        prompt_tokens = usage.get("prompt_tokens") or api_bench.parse_int_header(response_headers, "x-nanovllm-prompt-tokens")
        completion_tokens = usage.get("completion_tokens") or api_bench.parse_int_header(response_headers, "x-nanovllm-completion-tokens")
        total_tokens = usage.get("total_tokens")
        if total_tokens is None and prompt_tokens is not None and completion_tokens is not None:
            total_tokens = prompt_tokens + completion_tokens
        extracted_answer, is_correct = score_response_text(text, sample.gold_answer)
        return GSM8KRequestMetrics(
            request_index=request_index,
            worker_id=worker_id,
            ok=True,
            status_code=response.status_code,
            error=None,
            latency_s=latency_s,
            ttft_s=None,
            response_bytes=response_bytes,
            response_chars=len(text),
            request_id=request_id,
            finish_reason=finish_reason,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            server_processing_ms=api_bench.parse_float_header(response_headers, "openai-processing-ms"),
            server_queue_wait_ms=api_bench.parse_float_header(response_headers, "x-nanovllm-queue-wait-ms"),
            server_ttft_ms=api_bench.parse_float_header(response_headers, "x-nanovllm-ttft-ms"),
            server_generation_ms=api_bench.parse_float_header(response_headers, "x-nanovllm-generation-ms"),
            server_total_ms=api_bench.parse_float_header(response_headers, "x-nanovllm-total-ms"),
            server_output_tps=api_bench.parse_float_header(response_headers, "x-nanovllm-output-tokens-per-second"),
            server_decode_tps=api_bench.parse_float_header(response_headers, "x-nanovllm-decode-tokens-per-second"),
            server_request_parse_ms=api_bench.parse_float_header(response_headers, "x-nanovllm-request-parse-ms"),
            server_request_setup_ms=api_bench.parse_float_header(response_headers, "x-nanovllm-request-setup-ms"),
            server_response_build_ms=api_bench.parse_float_header(response_headers, "x-nanovllm-response-build-ms"),
            server_app_ms=api_bench.parse_float_header(response_headers, "x-nanovllm-server-app-ms"),
            sample_index=sample.sample_index,
            gold_answer=sample.gold_answer,
            extracted_answer=extracted_answer,
            is_correct=is_correct,
            prompt_preview=sample.problem[:120],
        )
    except Exception as exc:
        latency_s = time.perf_counter() - started_at
        return GSM8KRequestMetrics(
            request_index=request_index,
            worker_id=worker_id,
            ok=False,
            status_code=None,
            error=f"{type(exc).__name__}: {exc}",
            latency_s=latency_s,
            ttft_s=None,
            response_bytes=0,
            response_chars=0,
            request_id=None,
            finish_reason=None,
            prompt_tokens=None,
            completion_tokens=None,
            total_tokens=None,
            server_processing_ms=None,
            server_queue_wait_ms=None,
            server_ttft_ms=None,
            server_generation_ms=None,
            server_total_ms=None,
            server_output_tps=None,
            server_decode_tps=None,
            server_request_parse_ms=None,
            server_request_setup_ms=None,
            server_response_build_ms=None,
            server_app_ms=None,
            sample_index=sample.sample_index,
            gold_answer=sample.gold_answer,
            extracted_answer=None,
            is_correct=None,
            prompt_preview=sample.problem[:120],
        )


async def run_stream_request(
    *,
    client: httpx.AsyncClient,
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    request_index: int,
    worker_id: int,
    sample: GSM8KSample,
) -> GSM8KRequestMetrics:
    started_at = time.perf_counter()
    ttft_s: float | None = None
    response_bytes = 0
    request_id: str | None = None
    finish_reason: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    server_processing_ms = None
    server_queue_wait_ms = None
    server_ttft_ms = None
    server_generation_ms = None
    server_total_ms = None
    server_output_tps = None
    server_decode_tps = None
    server_request_parse_ms = None
    server_request_setup_ms = None
    server_response_build_ms = None
    server_app_ms = None
    chunks: list[str] = []
    try:
        async with client.stream("POST", url, headers=headers, json=payload) as response:
            response_headers = response.headers
            request_id = response_headers.get("x-request-id")
            server_processing_ms = api_bench.parse_float_header(response_headers, "openai-processing-ms")
            server_queue_wait_ms = api_bench.parse_float_header(response_headers, "x-nanovllm-queue-wait-ms")
            server_ttft_ms = api_bench.parse_float_header(response_headers, "x-nanovllm-ttft-ms")
            server_generation_ms = api_bench.parse_float_header(response_headers, "x-nanovllm-generation-ms")
            server_total_ms = api_bench.parse_float_header(response_headers, "x-nanovllm-total-ms")
            server_output_tps = api_bench.parse_float_header(response_headers, "x-nanovllm-output-tokens-per-second")
            server_decode_tps = api_bench.parse_float_header(response_headers, "x-nanovllm-decode-tokens-per-second")
            server_request_parse_ms = api_bench.parse_float_header(response_headers, "x-nanovllm-request-parse-ms")
            server_request_setup_ms = api_bench.parse_float_header(response_headers, "x-nanovllm-request-setup-ms")
            server_response_build_ms = api_bench.parse_float_header(response_headers, "x-nanovllm-response-build-ms")
            server_app_ms = api_bench.parse_float_header(response_headers, "x-nanovllm-server-app-ms")
            prompt_tokens = api_bench.parse_int_header(response_headers, "x-nanovllm-prompt-tokens")
            completion_tokens = api_bench.parse_int_header(response_headers, "x-nanovllm-completion-tokens")
            if response.status_code != 200:
                error_text = (await response.aread()).decode("utf-8", errors="ignore")
                latency_s = time.perf_counter() - started_at
                return GSM8KRequestMetrics(
                    request_index=request_index,
                    worker_id=worker_id,
                    ok=False,
                    status_code=response.status_code,
                    error=error_text[:500],
                    latency_s=latency_s,
                    ttft_s=None,
                    response_bytes=len(error_text.encode("utf-8")),
                    response_chars=len(error_text),
                    request_id=request_id,
                    finish_reason=None,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=None,
                    server_processing_ms=server_processing_ms,
                    server_queue_wait_ms=server_queue_wait_ms,
                    server_ttft_ms=server_ttft_ms,
                    server_generation_ms=server_generation_ms,
                    server_total_ms=server_total_ms,
                    server_output_tps=server_output_tps,
                    server_decode_tps=server_decode_tps,
                    server_request_parse_ms=server_request_parse_ms,
                    server_request_setup_ms=server_request_setup_ms,
                    server_response_build_ms=server_response_build_ms,
                    server_app_ms=server_app_ms,
                    sample_index=sample.sample_index,
                    gold_answer=sample.gold_answer,
                    extracted_answer=None,
                    is_correct=None,
                    prompt_preview=sample.problem[:120],
                )
            async for line in response.aiter_lines():
                if not line:
                    continue
                response_bytes += len(line.encode("utf-8")) + 1
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                usage = chunk.get("usage") or {}
                if usage:
                    prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
                    completion_tokens = usage.get("completion_tokens", completion_tokens)
                    total_tokens = usage.get("total_tokens", total_tokens)
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                choice = choices[0]
                finish_reason = choice.get("finish_reason") or finish_reason
                text_delta = choice.get("text") or ""
                if text_delta and ttft_s is None:
                    ttft_s = time.perf_counter() - started_at
                chunks.append(text_delta)
            latency_s = time.perf_counter() - started_at
            if total_tokens is None and prompt_tokens is not None and completion_tokens is not None:
                total_tokens = prompt_tokens + completion_tokens
            text = "".join(chunks)
            extracted_answer, is_correct = score_response_text(text, sample.gold_answer)
            return GSM8KRequestMetrics(
                request_index=request_index,
                worker_id=worker_id,
                ok=True,
                status_code=response.status_code,
                error=None,
                latency_s=latency_s,
                ttft_s=ttft_s,
                response_bytes=response_bytes,
                response_chars=len(text),
                request_id=request_id,
                finish_reason=finish_reason,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                server_processing_ms=server_processing_ms,
                server_queue_wait_ms=server_queue_wait_ms,
                server_ttft_ms=server_ttft_ms,
                server_generation_ms=server_generation_ms,
                server_total_ms=server_total_ms,
                server_output_tps=server_output_tps,
                server_decode_tps=server_decode_tps,
                server_request_parse_ms=server_request_parse_ms,
                server_request_setup_ms=server_request_setup_ms,
                server_response_build_ms=server_response_build_ms,
                server_app_ms=server_app_ms,
                sample_index=sample.sample_index,
                gold_answer=sample.gold_answer,
                extracted_answer=extracted_answer,
                is_correct=is_correct,
                prompt_preview=sample.problem[:120],
            )
    except Exception as exc:
        latency_s = time.perf_counter() - started_at
        return GSM8KRequestMetrics(
            request_index=request_index,
            worker_id=worker_id,
            ok=False,
            status_code=None,
            error=f"{type(exc).__name__}: {exc}",
            latency_s=latency_s,
            ttft_s=ttft_s,
            response_bytes=response_bytes,
            response_chars=sum(len(chunk) for chunk in chunks),
            request_id=request_id,
            finish_reason=finish_reason,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            server_processing_ms=server_processing_ms,
            server_queue_wait_ms=server_queue_wait_ms,
            server_ttft_ms=server_ttft_ms,
            server_generation_ms=server_generation_ms,
            server_total_ms=server_total_ms,
            server_output_tps=server_output_tps,
            server_decode_tps=server_decode_tps,
            server_request_parse_ms=server_request_parse_ms,
            server_request_setup_ms=server_request_setup_ms,
            server_response_build_ms=server_response_build_ms,
            server_app_ms=server_app_ms,
            sample_index=sample.sample_index,
            gold_answer=sample.gold_answer,
            extracted_answer=None,
            is_correct=None,
            prompt_preview=sample.problem[:120],
        )


async def worker_loop(
    *,
    worker_id: int,
    users: int,
    scheduler: api_bench.RequestScheduler,
    stats: api_bench.StatsCollector,
    sample_source: SampleSource,
    client: httpx.AsyncClient,
    url: str,
    headers: dict[str, str],
    model: str,
    max_tokens: int,
    temperature: float,
    stream: bool,
    ramp_seconds: float,
) -> None:
    if ramp_seconds > 0:
        await asyncio.sleep(ramp_seconds * worker_id / max(1, users))
    while True:
        request_index = await scheduler.reserve()
        if request_index is None:
            return
        sample = sample_source.sample(request_index)
        payload = build_payload(
            model=model,
            prompt=sample.prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=stream,
        )
        stats.mark_started()
        if stream:
            metric = await run_stream_request(
                client=client,
                url=url,
                headers=headers,
                payload=payload,
                request_index=request_index,
                worker_id=worker_id,
                sample=sample,
            )
        else:
            metric = await run_sync_request(
                client=client,
                url=url,
                headers=headers,
                payload=payload,
                request_index=request_index,
                worker_id=worker_id,
                sample=sample,
            )
        stats.add(metric)


def summarize_gsm8k_metrics(metrics: list[GSM8KRequestMetrics]) -> dict[str, Any]:
    successes = [metric for metric in metrics if metric.ok]
    extracted = [metric for metric in successes if metric.extracted_answer is not None]
    correct = [metric for metric in successes if metric.is_correct]
    unique_samples = len({metric.sample_index for metric in metrics})
    return {
        "success_requests": len(successes),
        "unique_samples_seen": unique_samples,
        "extract_count": len(extracted),
        "correct_count": len(correct),
        "extract_rate": (len(extracted) / len(successes) * 100.0) if successes else 0.0,
        "accuracy": (len(correct) / len(successes) * 100.0) if successes else 0.0,
    }


def print_gsm8k_summary(gsm8k_summary: dict[str, Any]) -> None:
    print(
        "gsm8k: "
        f"success={gsm8k_summary['success_requests']} "
        f"unique_samples={gsm8k_summary['unique_samples_seen']} "
        f"extract_rate={gsm8k_summary['extract_rate']:.2f} "
        f"acc={gsm8k_summary['accuracy']:.2f}"
    )


def print_sweep_table(rows: list[tuple[api_bench.BenchmarkSummary, dict[str, Any]]]) -> None:
    if len(rows) <= 1:
        return
    print("sweep:")
    print("  users  ok/total  rps   p50_ms  p95_ms  out_tps  extract  acc")
    for summary, gsm8k_summary in rows:
        users = summary.config["users"]
        latency_p50 = summary.latency_ms["p50"] if summary.latency_ms else 0.0
        latency_p95 = summary.latency_ms["p95"] if summary.latency_ms else 0.0
        out_tps = summary.output_tokens_per_second if summary.output_tokens_per_second is not None else 0.0
        print(
            f"  {users:5d}  {summary.success_requests:3d}/{summary.completed_requests:<5d} "
            f"{summary.requests_per_second:5.2f} {latency_p50:7.1f} {latency_p95:7.1f} "
            f"{out_tps:8.2f} {gsm8k_summary['extract_rate']:7.2f} {gsm8k_summary['accuracy']:5.2f}"
        )


def write_summary_output(
    path: str,
    rows: list[tuple[api_bench.BenchmarkSummary, dict[str, Any]]],
) -> None:
    payload_runs = [
        {
            "summary": asdict(summary),
            "gsm8k": gsm8k_summary,
        }
        for summary, gsm8k_summary in rows
    ]
    payload: Any = payload_runs[0] if len(payload_runs) == 1 else {"runs": payload_runs}
    Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


async def run_single_benchmark(args) -> tuple[api_bench.BenchmarkSummary, dict[str, Any], list[GSM8KRequestMetrics]]:
    if args.hf_gsm8k:
        samples = load_gsm8k_hf_samples(args.hf_split, args.limit)
    else:
        samples = load_gsm8k_samples(args.gsm8k_path, args.limit)
    if not samples:
        raise SystemExit("No GSM8K prompts loaded.")

    sample_source = SampleSource(samples, args.seed, args.random_sample)
    scheduler = api_bench.RequestScheduler(args.total_requests, args.duration)
    stats = api_bench.StatsCollector()
    stats.max_active_requests = args.users
    timeout = httpx.Timeout(args.timeout, connect=args.connect_timeout)
    limits = httpx.Limits(max_connections=args.users, max_keepalive_connections=args.users)
    base_url = args.base_url.rstrip("/")
    headers = api_bench.request_headers(args.api_key)
    started_at = time.perf_counter()

    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        progress_task = None
        if args.progress_interval > 0:
            progress_task = asyncio.create_task(
                api_bench.progress_reporter(stats, started_at, args.progress_interval)
            )
        try:
            tasks = [
                asyncio.create_task(
                    worker_loop(
                        worker_id=worker_id,
                        users=args.users,
                        scheduler=scheduler,
                        stats=stats,
                        sample_source=sample_source,
                        client=client,
                        url=base_url + "/v1/completions",
                        headers=headers,
                        model=args.model,
                        max_tokens=args.max_tokens,
                        temperature=args.temperature,
                        stream=args.stream,
                        ramp_seconds=args.ramp_seconds,
                    )
                )
                for worker_id in range(args.users)
            ]
            await asyncio.gather(*tasks)
        finally:
            if progress_task is not None:
                progress_task.cancel()
                await progress_task

    wall_time_s = time.perf_counter() - started_at
    config = {
        "run_label": getattr(args, "run_label", None),
        "base_url": args.base_url,
        "endpoint": "completions",
        "model": args.model,
        "users": args.users,
        "total_requests": args.total_requests,
        "duration": args.duration,
        "stream": args.stream,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "timeout": args.timeout,
        "connect_timeout": args.connect_timeout,
        "ramp_seconds": args.ramp_seconds,
        "dataset_size": len(samples),
        "hf_gsm8k": bool(args.hf_gsm8k),
        "random_sample": bool(args.random_sample),
    }
    metrics: list[GSM8KRequestMetrics] = list(stats.metrics)  # type: ignore[assignment]
    summary = api_bench.summarize_metrics(stats=stats, wall_time_s=wall_time_s, config=config)
    gsm8k_summary = summarize_gsm8k_metrics(metrics)
    return summary, gsm8k_summary, metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Load-test the OpenAI completions API with GSM8K prompts.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--users", "--concurrency", dest="users", type=int, default=32)
    parser.add_argument("--users-sweep", type=int, nargs="+", default=None)
    parser.add_argument("--total-requests", type=int, default=None)
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=768)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--gsm8k-path", default=DEFAULT_GSM8K)
    parser.add_argument("--hf-gsm8k", action="store_true")
    parser.add_argument("--hf-split", default="test")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--random-sample", action="store_true")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--connect-timeout", type=float, default=10.0)
    parser.add_argument("--ramp-seconds", type=float, default=0.0)
    parser.add_argument("--progress-interval", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--details-jsonl", default=None)
    parser.add_argument("--details-csv", default=None)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    user_values = args.users_sweep if args.users_sweep else [args.users]
    run_rows: list[tuple[api_bench.BenchmarkSummary, dict[str, Any]]] = []
    detail_rows: list[dict[str, Any]] = []
    for users in user_values:
        run_args = argparse.Namespace(**vars(args))
        run_args.users = users
        run_args.run_label = f"users={users}" if len(user_values) > 1 else "single-run"
        summary, gsm8k_summary, metrics = asyncio.run(run_single_benchmark(run_args))
        if len(user_values) > 1:
            print(f"\n== {run_args.run_label} ==")
        api_bench.print_summary(summary)
        print_gsm8k_summary(gsm8k_summary)
        run_rows.append((summary, gsm8k_summary))
        if args.details_jsonl or args.details_csv:
            detail_rows.extend(api_bench.build_metric_records(summary, metrics))
    print_sweep_table(run_rows)
    if args.output_json:
        write_summary_output(args.output_json, run_rows)
    if args.details_jsonl:
        api_bench.write_request_jsonl(args.details_jsonl, detail_rows)
    if args.details_csv:
        api_bench.write_request_csv(args.details_csv, detail_rows)


if __name__ == "__main__":
    main()
