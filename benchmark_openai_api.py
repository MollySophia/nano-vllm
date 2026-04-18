#!/usr/bin/env python3
import argparse
import asyncio
import csv
import json
import random
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import httpx


DEFAULT_PROMPTS = [
    "Explain linear attention in one short paragraph.",
    "Write a two-sentence summary of the Python GIL.",
    "Give three practical tips for debugging CUDA memory issues.",
    "用中文解释什么是状态缓存推理。",
    "Write a tiny haiku about servers under heavy load.",
    "Summarize why batching can improve LLM throughput.",
    "给出一个简短的 API 压测方案。",
    "Explain the tradeoff between latency and throughput in inference serving.",
]


@dataclass
class RequestMetrics:
    request_index: int
    worker_id: int
    ok: bool
    status_code: int | None
    error: str | None
    latency_s: float
    ttft_s: float | None
    response_bytes: int
    response_chars: int
    request_id: str | None
    finish_reason: str | None
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    server_processing_ms: float | None
    server_queue_wait_ms: float | None
    server_ttft_ms: float | None
    server_generation_ms: float | None
    server_total_ms: float | None
    server_output_tps: float | None
    server_decode_tps: float | None
    server_request_parse_ms: float | None = None
    server_request_setup_ms: float | None = None
    server_response_build_ms: float | None = None
    server_app_ms: float | None = None


@dataclass
class BenchmarkSummary:
    config: dict[str, Any]
    wall_time_s: float
    started_requests: int
    completed_requests: int
    success_requests: int
    error_requests: int
    requests_per_second: float
    success_requests_per_second: float
    input_tokens_total: int | None
    output_tokens_total: int | None
    input_tokens_per_second: float | None
    output_tokens_per_second: float | None
    output_chars_total: int
    output_chars_per_second: float
    status_codes: dict[str, int]
    latency_ms: dict[str, float] | None
    client_ttft_ms: dict[str, float] | None
    server_processing_ms: dict[str, float] | None
    server_queue_wait_ms: dict[str, float] | None
    server_ttft_ms: dict[str, float] | None
    server_generation_ms: dict[str, float] | None
    server_total_ms: dict[str, float] | None
    server_output_tps: dict[str, float] | None
    server_decode_tps: dict[str, float] | None
    sample_errors: list[str] = field(default_factory=list)
    server_request_parse_ms: dict[str, float] | None = None
    server_request_setup_ms: dict[str, float] | None = None
    server_response_build_ms: dict[str, float] | None = None
    server_app_ms: dict[str, float] | None = None


class PromptSource:
    def __init__(self, prompts: list[str], seed: int):
        if not prompts:
            raise ValueError("No prompts available for the benchmark.")
        self.prompts = prompts
        self.rng = random.Random(seed)

    def sample(self, request_index: int) -> str:
        if len(self.prompts) == 1:
            return self.prompts[0]
        return self.prompts[self.rng.randrange(len(self.prompts))]


class RequestScheduler:
    def __init__(self, total_requests: int | None, duration_s: float | None):
        if total_requests is None and duration_s is None:
            total_requests = 100
        self.total_requests = total_requests
        self.duration_s = duration_s
        self._next_request_index = 0
        self._lock = asyncio.Lock()
        self.started_at = time.perf_counter()

    async def reserve(self) -> int | None:
        async with self._lock:
            now = time.perf_counter()
            if self.duration_s is not None and now - self.started_at >= self.duration_s:
                return None
            if self.total_requests is not None and self._next_request_index >= self.total_requests:
                return None
            request_index = self._next_request_index
            self._next_request_index += 1
            return request_index


async def issue_request(
    *,
    request_index: int,
    worker_id: int,
    stats: "StatsCollector",
    prompt_source: PromptSource,
    client: httpx.AsyncClient,
    url: str,
    headers: dict[str, str],
    endpoint: str,
    model: str,
    system_prompt: str | None,
    max_tokens: int,
    temperature: float,
    stream: bool,
) -> None:
    prompt = prompt_source.sample(request_index)
    payload = build_payload(
        endpoint=endpoint,
        model=model,
        prompt=prompt,
        system_prompt=system_prompt,
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
            endpoint=endpoint,
            payload=payload,
            request_index=request_index,
            worker_id=worker_id,
        )
    else:
        metric = await run_sync_request(
            client=client,
            url=url,
            headers=headers,
            endpoint=endpoint,
            payload=payload,
            request_index=request_index,
            worker_id=worker_id,
        )
    stats.add(metric)


class StatsCollector:
    def __init__(self):
        self.metrics: list[RequestMetrics] = []
        self.status_codes: Counter[str] = Counter()
        self.sample_errors: list[str] = []
        self.started_requests = 0
        self.completed_requests = 0
        self.active_requests = 0
        self.max_active_requests = 0

    def mark_started(self):
        self.started_requests += 1
        self.active_requests += 1
        self.max_active_requests = max(self.max_active_requests, self.active_requests)

    def add(self, metric: RequestMetrics):
        self.metrics.append(metric)
        self.completed_requests += 1
        self.active_requests -= 1
        key = "none" if metric.status_code is None else str(metric.status_code)
        self.status_codes[key] += 1
        if metric.error and len(self.sample_errors) < 5:
            self.sample_errors.append(metric.error)


def percentile(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("percentile() requires a non-empty list")
    if len(values) == 1:
        return values[0]
    xs = sorted(values)
    position = (len(xs) - 1) * q
    lo = int(position)
    hi = min(lo + 1, len(xs) - 1)
    weight = position - lo
    return xs[lo] * (1.0 - weight) + xs[hi] * weight


def summarize_distribution(values: list[float], scale: float = 1.0) -> dict[str, float] | None:
    if not values:
        return None
    scaled = [value * scale for value in values]
    return {
        "count": float(len(scaled)),
        "mean": sum(scaled) / len(scaled),
        "min": min(scaled),
        "p50": percentile(scaled, 0.50),
        "p90": percentile(scaled, 0.90),
        "p95": percentile(scaled, 0.95),
        "p99": percentile(scaled, 0.99),
        "max": max(scaled),
    }


def parse_float_header(headers: httpx.Headers, name: str) -> float | None:
    value = headers.get(name)
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def parse_int_header(headers: httpx.Headers, name: str) -> int | None:
    value = headers.get(name)
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def request_headers(api_key: str | None) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def build_payload(
    *,
    endpoint: str,
    model: str,
    prompt: str,
    system_prompt: str | None,
    max_tokens: int,
    temperature: float,
    stream: bool,
) -> dict[str, Any]:
    if endpoint == "chat":
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": stream,
        }
    return {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": stream,
    }


def parse_sync_body(endpoint: str, body: dict[str, Any]) -> tuple[int, str, str | None]:
    choices = body.get("choices") or []
    if not choices:
        return 0, "", None
    choice = choices[0]
    finish_reason = choice.get("finish_reason")
    if endpoint == "chat":
        text = (choice.get("message") or {}).get("content") or ""
    else:
        text = choice.get("text") or ""
    return len(text.encode("utf-8")), text, finish_reason


async def run_sync_request(
    *,
    client: httpx.AsyncClient,
    url: str,
    headers: dict[str, str],
    endpoint: str,
    payload: dict[str, Any],
    request_index: int,
    worker_id: int,
) -> RequestMetrics:
    started_at = time.perf_counter()
    try:
        response = await client.post(url, headers=headers, json=payload)
        latency_s = time.perf_counter() - started_at
        body_text = response.text
        response_bytes = len(body_text.encode("utf-8"))
        response_chars = len(body_text)
        response_headers = response.headers
        request_id = response_headers.get("x-request-id")
        if response.status_code != 200:
            error = body_text[:500]
            return RequestMetrics(
                request_index=request_index,
                worker_id=worker_id,
                ok=False,
                status_code=response.status_code,
                error=error,
                latency_s=latency_s,
                ttft_s=None,
                response_bytes=response_bytes,
                response_chars=response_chars,
                request_id=request_id,
                finish_reason=None,
                prompt_tokens=parse_int_header(response_headers, "x-nanovllm-prompt-tokens"),
                completion_tokens=parse_int_header(response_headers, "x-nanovllm-completion-tokens"),
                total_tokens=None,
                server_processing_ms=parse_float_header(response_headers, "openai-processing-ms"),
                server_queue_wait_ms=parse_float_header(response_headers, "x-nanovllm-queue-wait-ms"),
                server_ttft_ms=parse_float_header(response_headers, "x-nanovllm-ttft-ms"),
                server_generation_ms=parse_float_header(response_headers, "x-nanovllm-generation-ms"),
                server_total_ms=parse_float_header(response_headers, "x-nanovllm-total-ms"),
                server_output_tps=parse_float_header(response_headers, "x-nanovllm-output-tokens-per-second"),
                server_decode_tps=parse_float_header(response_headers, "x-nanovllm-decode-tokens-per-second"),
                server_request_parse_ms=parse_float_header(response_headers, "x-nanovllm-request-parse-ms"),
                server_request_setup_ms=parse_float_header(response_headers, "x-nanovllm-request-setup-ms"),
                server_response_build_ms=parse_float_header(response_headers, "x-nanovllm-response-build-ms"),
                server_app_ms=parse_float_header(response_headers, "x-nanovllm-server-app-ms"),
            )
        body = response.json()
        _, text, finish_reason = parse_sync_body(endpoint, body)
        usage = body.get("usage") or {}
        prompt_tokens = usage.get("prompt_tokens") or parse_int_header(response_headers, "x-nanovllm-prompt-tokens")
        completion_tokens = usage.get("completion_tokens") or parse_int_header(response_headers, "x-nanovllm-completion-tokens")
        total_tokens = usage.get("total_tokens")
        if total_tokens is None and prompt_tokens is not None and completion_tokens is not None:
            total_tokens = prompt_tokens + completion_tokens
        return RequestMetrics(
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
            server_processing_ms=parse_float_header(response_headers, "openai-processing-ms"),
            server_queue_wait_ms=parse_float_header(response_headers, "x-nanovllm-queue-wait-ms"),
            server_ttft_ms=parse_float_header(response_headers, "x-nanovllm-ttft-ms"),
            server_generation_ms=parse_float_header(response_headers, "x-nanovllm-generation-ms"),
            server_total_ms=parse_float_header(response_headers, "x-nanovllm-total-ms"),
            server_output_tps=parse_float_header(response_headers, "x-nanovllm-output-tokens-per-second"),
            server_decode_tps=parse_float_header(response_headers, "x-nanovllm-decode-tokens-per-second"),
            server_request_parse_ms=parse_float_header(response_headers, "x-nanovllm-request-parse-ms"),
            server_request_setup_ms=parse_float_header(response_headers, "x-nanovllm-request-setup-ms"),
            server_response_build_ms=parse_float_header(response_headers, "x-nanovllm-response-build-ms"),
            server_app_ms=parse_float_header(response_headers, "x-nanovllm-server-app-ms"),
        )
    except Exception as exc:
        latency_s = time.perf_counter() - started_at
        return RequestMetrics(
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
        )


async def run_stream_request(
    *,
    client: httpx.AsyncClient,
    url: str,
    headers: dict[str, str],
    endpoint: str,
    payload: dict[str, Any],
    request_index: int,
    worker_id: int,
) -> RequestMetrics:
    started_at = time.perf_counter()
    ttft_s: float | None = None
    response_bytes = 0
    response_chars = 0
    finish_reason: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    request_id: str | None = None
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
    try:
        async with client.stream("POST", url, headers=headers, json=payload) as response:
            request_id = response.headers.get("x-request-id")
            server_processing_ms = parse_float_header(response.headers, "openai-processing-ms")
            server_queue_wait_ms = parse_float_header(response.headers, "x-nanovllm-queue-wait-ms")
            server_ttft_ms = parse_float_header(response.headers, "x-nanovllm-ttft-ms")
            server_generation_ms = parse_float_header(response.headers, "x-nanovllm-generation-ms")
            server_total_ms = parse_float_header(response.headers, "x-nanovllm-total-ms")
            server_output_tps = parse_float_header(response.headers, "x-nanovllm-output-tokens-per-second")
            server_decode_tps = parse_float_header(response.headers, "x-nanovllm-decode-tokens-per-second")
            server_request_parse_ms = parse_float_header(response.headers, "x-nanovllm-request-parse-ms")
            server_request_setup_ms = parse_float_header(response.headers, "x-nanovllm-request-setup-ms")
            server_response_build_ms = parse_float_header(response.headers, "x-nanovllm-response-build-ms")
            server_app_ms = parse_float_header(response.headers, "x-nanovllm-server-app-ms")
            prompt_tokens = parse_int_header(response.headers, "x-nanovllm-prompt-tokens")
            completion_tokens = parse_int_header(response.headers, "x-nanovllm-completion-tokens")
            if response.status_code != 200:
                error_text = (await response.aread()).decode("utf-8", errors="ignore")
                latency_s = time.perf_counter() - started_at
                return RequestMetrics(
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
                if endpoint == "chat":
                    text_delta = ((choice.get("delta") or {}).get("content")) or ""
                else:
                    text_delta = choice.get("text") or ""
                if text_delta and ttft_s is None:
                    ttft_s = time.perf_counter() - started_at
                response_chars += len(text_delta)
            latency_s = time.perf_counter() - started_at
            if total_tokens is None and prompt_tokens is not None and completion_tokens is not None:
                total_tokens = prompt_tokens + completion_tokens
            return RequestMetrics(
                request_index=request_index,
                worker_id=worker_id,
                ok=True,
                status_code=response.status_code,
                error=None,
                latency_s=latency_s,
                ttft_s=ttft_s,
                response_bytes=response_bytes,
                response_chars=response_chars,
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
            )
    except Exception as exc:
        latency_s = time.perf_counter() - started_at
        return RequestMetrics(
            request_index=request_index,
            worker_id=worker_id,
            ok=False,
            status_code=None,
            error=f"{type(exc).__name__}: {exc}",
            latency_s=latency_s,
            ttft_s=ttft_s,
            response_bytes=response_bytes,
            response_chars=response_chars,
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
        )


async def worker_loop(
    *,
    worker_id: int,
    users: int,
    scheduler: RequestScheduler,
    stats: StatsCollector,
    prompt_source: PromptSource,
    client: httpx.AsyncClient,
    url: str,
    headers: dict[str, str],
    endpoint: str,
    model: str,
    system_prompt: str | None,
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
        await issue_request(
            request_index=request_index,
            worker_id=worker_id,
            stats=stats,
            prompt_source=prompt_source,
            client=client,
            url=url,
            headers=headers,
            endpoint=endpoint,
            model=model,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=stream,
        )


async def open_loop_driver(
    *,
    arrival_rate: float,
    total_requests: int | None,
    duration_s: float | None,
    max_in_flight: int,
    stats: StatsCollector,
    prompt_source: PromptSource,
    client: httpx.AsyncClient,
    url: str,
    headers: dict[str, str],
    endpoint: str,
    model: str,
    system_prompt: str | None,
    max_tokens: int,
    temperature: float,
    stream: bool,
    seed: int,
) -> None:
    if arrival_rate <= 0:
        raise ValueError("arrival_rate must be > 0 for open-loop mode.")
    started_at = time.perf_counter()
    deadline = None if duration_s is None else started_at + duration_s
    rng = random.Random(seed)
    request_index = 0
    next_arrival_at = started_at
    in_flight_sem = asyncio.Semaphore(max_in_flight) if max_in_flight > 0 else None
    in_flight_tasks: set[asyncio.Task[None]] = set()

    async def run_open_request(index: int) -> None:
        try:
            await issue_request(
                request_index=index,
                worker_id=-1,
                stats=stats,
                prompt_source=prompt_source,
                client=client,
                url=url,
                headers=headers,
                endpoint=endpoint,
                model=model,
                system_prompt=system_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=stream,
            )
        finally:
            if in_flight_sem is not None:
                in_flight_sem.release()

    while True:
        if total_requests is not None and request_index >= total_requests:
            break
        if deadline is not None and time.perf_counter() >= deadline:
            break
        now = time.perf_counter()
        sleep_s = next_arrival_at - now
        if sleep_s > 0:
            await asyncio.sleep(sleep_s)
        if deadline is not None and time.perf_counter() >= deadline:
            break
        if in_flight_sem is not None:
            await in_flight_sem.acquire()
        task = asyncio.create_task(run_open_request(request_index))
        in_flight_tasks.add(task)
        task.add_done_callback(in_flight_tasks.discard)
        request_index += 1
        next_arrival_at += rng.expovariate(arrival_rate)

    if in_flight_tasks:
        await asyncio.gather(*in_flight_tasks)


async def progress_reporter(stats: StatsCollector, started_at: float, interval_s: float) -> None:
    try:
        while True:
            await asyncio.sleep(interval_s)
            elapsed = time.perf_counter() - started_at
            print(
                f"[progress] elapsed={elapsed:.1f}s started={stats.started_requests} "
                f"completed={stats.completed_requests} active={stats.active_requests} "
                f"ok={sum(1 for m in stats.metrics if m.ok)} err={sum(1 for m in stats.metrics if not m.ok)}"
            )
    except asyncio.CancelledError:
        return


def summarize_metrics(
    *,
    stats: StatsCollector,
    wall_time_s: float,
    config: dict[str, Any],
) -> BenchmarkSummary:
    metrics = stats.metrics
    successes = [metric for metric in metrics if metric.ok]
    failures = [metric for metric in metrics if not metric.ok]

    prompt_token_values = [metric.prompt_tokens for metric in successes if metric.prompt_tokens is not None]
    completion_token_values = [metric.completion_tokens for metric in successes if metric.completion_tokens is not None]
    input_tokens_total = sum(prompt_token_values) if prompt_token_values else None
    output_tokens_total = sum(completion_token_values) if completion_token_values else None

    return BenchmarkSummary(
        config=config,
        wall_time_s=wall_time_s,
        started_requests=stats.started_requests,
        completed_requests=stats.completed_requests,
        success_requests=len(successes),
        error_requests=len(failures),
        requests_per_second=(stats.completed_requests / wall_time_s) if wall_time_s > 0 else 0.0,
        success_requests_per_second=(len(successes) / wall_time_s) if wall_time_s > 0 else 0.0,
        input_tokens_total=input_tokens_total,
        output_tokens_total=output_tokens_total,
        input_tokens_per_second=(input_tokens_total / wall_time_s) if wall_time_s > 0 and input_tokens_total is not None else None,
        output_tokens_per_second=(output_tokens_total / wall_time_s) if wall_time_s > 0 and output_tokens_total is not None else None,
        output_chars_total=sum(metric.response_chars for metric in successes),
        output_chars_per_second=(sum(metric.response_chars for metric in successes) / wall_time_s) if wall_time_s > 0 else 0.0,
        status_codes=dict(stats.status_codes),
        latency_ms=summarize_distribution([metric.latency_s for metric in metrics], scale=1000.0),
        client_ttft_ms=summarize_distribution([metric.ttft_s for metric in metrics if metric.ttft_s is not None], scale=1000.0),
        server_processing_ms=summarize_distribution(
            [metric.server_processing_ms for metric in metrics if metric.server_processing_ms is not None]
        ),
        server_queue_wait_ms=summarize_distribution(
            [metric.server_queue_wait_ms for metric in metrics if metric.server_queue_wait_ms is not None]
        ),
        server_ttft_ms=summarize_distribution(
            [metric.server_ttft_ms for metric in metrics if metric.server_ttft_ms is not None]
        ),
        server_generation_ms=summarize_distribution(
            [metric.server_generation_ms for metric in metrics if metric.server_generation_ms is not None]
        ),
        server_total_ms=summarize_distribution(
            [metric.server_total_ms for metric in metrics if metric.server_total_ms is not None]
        ),
        server_output_tps=summarize_distribution(
            [metric.server_output_tps for metric in metrics if metric.server_output_tps is not None]
        ),
        server_decode_tps=summarize_distribution(
            [metric.server_decode_tps for metric in metrics if metric.server_decode_tps is not None]
        ),
        sample_errors=stats.sample_errors,
        server_request_parse_ms=summarize_distribution(
            [metric.server_request_parse_ms for metric in metrics if metric.server_request_parse_ms is not None]
        ),
        server_request_setup_ms=summarize_distribution(
            [metric.server_request_setup_ms for metric in metrics if metric.server_request_setup_ms is not None]
        ),
        server_response_build_ms=summarize_distribution(
            [metric.server_response_build_ms for metric in metrics if metric.server_response_build_ms is not None]
        ),
        server_app_ms=summarize_distribution(
            [metric.server_app_ms for metric in metrics if metric.server_app_ms is not None]
        ),
    )


def print_distribution(label: str, values: dict[str, float] | None, unit: str) -> None:
    if not values:
        return
    print(
        f"{label}: mean={values['mean']:.2f}{unit} p50={values['p50']:.2f}{unit} "
        f"p90={values['p90']:.2f}{unit} p95={values['p95']:.2f}{unit} "
        f"p99={values['p99']:.2f}{unit} max={values['max']:.2f}{unit}"
    )


def print_summary(summary: BenchmarkSummary) -> None:
    print(
        f"run: wall_time_s={summary.wall_time_s:.2f} started={summary.started_requests} "
        f"completed={summary.completed_requests} ok={summary.success_requests} err={summary.error_requests} "
        f"rps={summary.requests_per_second:.2f} success_rps={summary.success_requests_per_second:.2f}"
    )
    if summary.input_tokens_total is not None or summary.output_tokens_total is not None:
        parts = []
        if summary.input_tokens_total is not None:
            parts.append(
                f"input_tokens={summary.input_tokens_total} input_tps={summary.input_tokens_per_second:.2f}"
            )
        if summary.output_tokens_total is not None:
            parts.append(
                f"output_tokens={summary.output_tokens_total} output_tps={summary.output_tokens_per_second:.2f}"
            )
        print("tokens: " + " ".join(parts))
    print(
        f"output: chars_total={summary.output_chars_total} "
        f"chars_per_second={summary.output_chars_per_second:.2f}"
    )
    print("status_codes: " + ", ".join(f"{code}={count}" for code, count in sorted(summary.status_codes.items())))
    print_distribution("latency_ms", summary.latency_ms, "ms")
    print_distribution("client_ttft_ms", summary.client_ttft_ms, "ms")
    print_distribution("server_processing_ms", summary.server_processing_ms, "ms")
    print_distribution("server_queue_wait_ms", summary.server_queue_wait_ms, "ms")
    print_distribution("server_ttft_ms", summary.server_ttft_ms, "ms")
    print_distribution("server_generation_ms", summary.server_generation_ms, "ms")
    print_distribution("server_total_ms", summary.server_total_ms, "ms")
    print_distribution("server_request_parse_ms", summary.server_request_parse_ms, "ms")
    print_distribution("server_request_setup_ms", summary.server_request_setup_ms, "ms")
    print_distribution("server_response_build_ms", summary.server_response_build_ms, "ms")
    print_distribution("server_app_ms", summary.server_app_ms, "ms")
    print_distribution("server_output_tps", summary.server_output_tps, "")
    print_distribution("server_decode_tps", summary.server_decode_tps, "")
    if summary.sample_errors:
        print("sample_errors:")
        for error in summary.sample_errors:
            print(f"  - {error}")


def print_sweep_table(summaries: list[BenchmarkSummary]) -> None:
    if len(summaries) <= 1:
        return
    print("sweep:")
    print("  target      ok/total  rps   p50_ms  p95_ms  ttft_p50_ms  out_tps")
    for summary in summaries:
        if summary.config.get("load_mode") == "open-loop":
            target = f"{summary.config['arrival_rate']:.1f} rps"
        else:
            target = f"{summary.config['users']} users"
        latency_p50 = summary.latency_ms["p50"] if summary.latency_ms else 0.0
        latency_p95 = summary.latency_ms["p95"] if summary.latency_ms else 0.0
        ttft_p50 = summary.client_ttft_ms["p50"] if summary.client_ttft_ms else 0.0
        out_tps = summary.output_tokens_per_second if summary.output_tokens_per_second is not None else 0.0
        print(
            f"  {target:10s}  {summary.success_requests:3d}/{summary.completed_requests:<5d} "
            f"{summary.requests_per_second:5.2f} {latency_p50:7.1f} {latency_p95:7.1f} "
            f"{ttft_p50:11.1f} {out_tps:8.2f}"
        )


def build_metric_records(summary: BenchmarkSummary, metrics: list[RequestMetrics]) -> list[dict[str, Any]]:
    base = {
        "run_label": summary.config.get("run_label"),
        "base_url": summary.config.get("base_url"),
        "load_mode": summary.config.get("load_mode"),
        "model": summary.config.get("model"),
        "endpoint": summary.config.get("endpoint"),
        "users": summary.config.get("users"),
        "arrival_rate": summary.config.get("arrival_rate"),
        "max_in_flight": summary.config.get("max_in_flight"),
        "stream": summary.config.get("stream"),
        "max_tokens": summary.config.get("max_tokens"),
        "temperature": summary.config.get("temperature"),
    }
    return [{**base, **asdict(metric)} for metric in metrics]


def write_summary_output(path: str, summaries: list[BenchmarkSummary]) -> None:
    payload: Any
    if len(summaries) == 1:
        payload = asdict(summaries[0])
    else:
        payload = {"runs": [asdict(summary) for summary in summaries]}
    Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def write_request_jsonl(path: str, rows: list[dict[str, Any]]) -> None:
    with Path(path).open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_request_csv(path: str, rows: list[dict[str, Any]]) -> None:
    if not rows:
        Path(path).write_text("")
        return
    fieldnames = list(rows[0].keys())
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_prompts(prompt_file: str | None, inline_prompts: list[str], prompt_repeat: int) -> list[str]:
    prompts = list(inline_prompts)
    if prompt_file:
        path = Path(prompt_file)
        if path.suffix == ".jsonl":
            for line in path.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if "prompt" in record:
                    prompts.append(str(record["prompt"]))
                elif "text" in record:
                    prompts.append(str(record["text"]))
        else:
            for line in path.read_text().splitlines():
                line = line.strip()
                if line:
                    prompts.append(line)
    if not prompts:
        prompts = list(DEFAULT_PROMPTS)
    prompts *= max(1, prompt_repeat)
    return prompts


async def run_single_benchmark(args) -> tuple[BenchmarkSummary, list[RequestMetrics]]:
    total_requests = args.total_requests
    if total_requests is None and args.duration is None:
        total_requests = 100
    prompts = load_prompts(args.prompt_file, args.prompt, args.prompt_repeat)
    prompt_source = PromptSource(prompts, args.seed)
    stats = StatsCollector()
    timeout = httpx.Timeout(args.timeout, connect=args.connect_timeout)
    max_connections = args.max_connections
    if max_connections is None:
        if args.load_mode == "closed-loop":
            max_connections = args.users
        else:
            max_connections = max(256, args.users)
            if args.max_in_flight > 0:
                max_connections = max(max_connections, args.max_in_flight)
    max_keepalive_connections = (
        max_connections if args.max_keepalive_connections is None else args.max_keepalive_connections
    )
    limits = httpx.Limits(
        max_connections=max_connections,
        max_keepalive_connections=max_keepalive_connections,
    )
    base_url = args.base_url.rstrip("/")
    path = "/v1/chat/completions" if args.endpoint == "chat" else "/v1/completions"
    headers = request_headers(args.api_key)
    started_at = time.perf_counter()

    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        progress_task = None
        if args.progress_interval > 0:
            progress_task = asyncio.create_task(progress_reporter(stats, started_at, args.progress_interval))
        try:
            if args.load_mode == "closed-loop":
                scheduler = RequestScheduler(total_requests, args.duration)
                tasks = [
                    asyncio.create_task(
                        worker_loop(
                            worker_id=worker_id,
                            users=args.users,
                            scheduler=scheduler,
                            stats=stats,
                            prompt_source=prompt_source,
                            client=client,
                            url=base_url + path,
                            headers=headers,
                            endpoint=args.endpoint,
                            model=args.model,
                            system_prompt=args.system_prompt,
                            max_tokens=args.max_tokens,
                            temperature=args.temperature,
                            stream=args.stream,
                            ramp_seconds=args.ramp_seconds,
                        )
                    )
                    for worker_id in range(args.users)
                ]
                await asyncio.gather(*tasks)
            else:
                await open_loop_driver(
                    arrival_rate=args.arrival_rate,
                    total_requests=total_requests,
                    duration_s=args.duration,
                    max_in_flight=args.max_in_flight,
                    stats=stats,
                    prompt_source=prompt_source,
                    client=client,
                    url=base_url + path,
                    headers=headers,
                    endpoint=args.endpoint,
                    model=args.model,
                    system_prompt=args.system_prompt,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    stream=args.stream,
                    seed=args.seed,
                )
        finally:
            if progress_task is not None:
                progress_task.cancel()
                await progress_task

    wall_time_s = time.perf_counter() - started_at
    config = {
        "run_label": getattr(args, "run_label", None),
        "base_url": args.base_url,
        "load_mode": args.load_mode,
        "endpoint": args.endpoint,
        "model": args.model,
        "users": args.users if args.load_mode == "closed-loop" else None,
        "arrival_rate": args.arrival_rate if args.load_mode == "open-loop" else None,
        "max_in_flight": args.max_in_flight if args.load_mode == "open-loop" else None,
        "max_connections": max_connections,
        "total_requests": total_requests,
        "duration": args.duration,
        "stream": args.stream,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "prompt_count": len(prompts),
        "timeout": args.timeout,
        "connect_timeout": args.connect_timeout,
        "ramp_seconds": args.ramp_seconds,
    }
    summary = summarize_metrics(stats=stats, wall_time_s=wall_time_s, config=config)
    return summary, stats.metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Load-test an OpenAI-compatible API endpoint.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--endpoint", choices=["chat", "completions"], default="chat")
    parser.add_argument("--load-mode", choices=["closed-loop", "open-loop"], default="closed-loop")
    parser.add_argument(
        "--users",
        "--concurrency",
        dest="users",
        type=int,
        default=32,
        help="Closed-loop user/task count. In open-loop mode this is only used as a default connection cap.",
    )
    parser.add_argument("--users-sweep", type=int, nargs="+", default=None)
    parser.add_argument(
        "--arrival-rate",
        type=float,
        default=32.0,
        help="Target request arrival rate in requests/sec for open-loop mode.",
    )
    parser.add_argument("--arrival-rate-sweep", type=float, nargs="+", default=None)
    parser.add_argument(
        "--max-in-flight",
        type=int,
        default=0,
        help="Optional open-loop client-side in-flight cap; 0 means unlimited.",
    )
    parser.add_argument("--max-connections", type=int, default=None)
    parser.add_argument("--max-keepalive-connections", type=int, default=None)
    parser.add_argument("--total-requests", type=int, default=None)
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--system-prompt", default=None)
    parser.add_argument("--prompt", action="append", default=[])
    parser.add_argument("--prompt-file", default=None)
    parser.add_argument("--prompt-repeat", type=int, default=1)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--timeout", type=float, default=120.0)
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
    if args.load_mode == "open-loop":
        sweep_values = args.arrival_rate_sweep if args.arrival_rate_sweep else [args.arrival_rate]
    else:
        sweep_values = args.users_sweep if args.users_sweep else [args.users]
    summaries: list[BenchmarkSummary] = []
    detail_rows: list[dict[str, Any]] = []
    for run_index, sweep_value in enumerate(sweep_values):
        run_args = argparse.Namespace(**vars(args))
        if args.load_mode == "open-loop":
            run_args.arrival_rate = float(sweep_value)
            run_args.run_label = (
                f"arrival_rate={run_args.arrival_rate:g}"
                if len(sweep_values) > 1
                else "single-run"
            )
        else:
            run_args.users = int(sweep_value)
            run_args.run_label = f"users={run_args.users}" if len(sweep_values) > 1 else "single-run"
        summary, metrics = asyncio.run(run_single_benchmark(run_args))
        if len(sweep_values) > 1:
            print(f"\n== {run_args.run_label} ==")
        print_summary(summary)
        summaries.append(summary)
        if args.details_jsonl or args.details_csv:
            detail_rows.extend(build_metric_records(summary, metrics))
    print_sweep_table(summaries)
    if args.output_json:
        write_summary_output(args.output_json, summaries)
    if args.details_jsonl:
        write_request_jsonl(args.details_jsonl, detail_rows)
    if args.details_csv:
        write_request_csv(args.details_csv, detail_rows)


if __name__ == "__main__":
    main()
