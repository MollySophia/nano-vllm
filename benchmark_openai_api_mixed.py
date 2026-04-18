#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import random
import textwrap
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import httpx

from benchmark_openai_api import (
    RequestMetrics,
    RequestScheduler,
    StatsCollector,
    build_payload,
    print_distribution,
    print_summary,
    request_headers,
    run_stream_request,
    run_sync_request,
    summarize_metrics,
    write_request_csv,
    write_request_jsonl,
)


SHORT_PROMPTS = [
    "Explain linear attention in one short paragraph.",
    "Give three practical tips for debugging CUDA memory issues.",
    "用中文简短解释什么是状态缓存推理。",
    "Summarize why batching improves throughput.",
    "Write a tiny haiku about servers under heavy load.",
    "Give a compact API validation checklist.",
]

MEDIUM_PROMPTS = [
    "You are reviewing an inference service. List five likely latency bottlenecks, then give a short plan to isolate them without changing model weights.",
    "Explain the tradeoff between latency and throughput in LLM serving, and include the role of batching, queueing, and prompt length.",
    "用中文说明 prefix cache、state cache、prefill 和 decode 的关系，并补充两个常见误区。",
    "Write a concise note for an engineer comparing single-process serving, reverse-proxy frontends, and queue-based ingress for an OpenAI-compatible API.",
]

LONG_PROMPTS = [
    textwrap.dedent(
        """
        You are debugging a production inference stack that serves an OpenAI-compatible API.
        The symptoms are healthy GPU utilization, rising queue wait under bursty traffic, and mixed prompt lengths.
        Explain a practical debugging plan that separates transport overhead, request validation, prompt rendering,
        batching efficiency, and model-side decode throughput.
        """
    ).strip(),
    textwrap.dedent(
        """
        用中文写一段面向工程师的说明，主题是高并发推理服务中的混合负载。
        覆盖不同 prompt 长度对 prefill 的影响、不同 max_tokens 对 decode 的影响，
        以及为什么单看短 prompt benchmark 往往会误判真实性能。
        """
    ).strip(),
]

LONG_COMPLETION_PROMPTS = [
    textwrap.dedent(
        """
        Write a compact engineering memo about serving optimizations for a text generation API.
        Cover request admission, batch formation, prompt length variance, decode saturation,
        and API compatibility tradeoffs.
        """
    ).strip(),
    textwrap.dedent(
        """
        给出一份面向服务端工程师的说明，主题是 OpenAI-compatible 接口下的混合流量压测。
        需要提到：短 prompt 与长 prompt 的区别；同步请求和流式请求对延迟统计的影响；
        以及如何设计更接近真实用户行为的 workload。
        """
    ).strip(),
]


@dataclass(frozen=True)
class WorkloadProfile:
    name: str
    weight: float
    endpoint: str
    stream: bool
    max_tokens: int
    temperature: float
    system_prompt: str | None
    prompts: tuple[str, ...]


@dataclass
class MixedMetricRecord:
    profile_name: str
    endpoint: str
    stream: bool
    max_tokens: int
    prompt_chars: int
    prompt_preview: str
    metric: RequestMetrics


class WeightedProfileSource:
    def __init__(self, profiles: list[WorkloadProfile], seed: int):
        if not profiles:
            raise ValueError("At least one workload profile is required.")
        weights = [profile.weight for profile in profiles]
        if any(weight <= 0 for weight in weights):
            raise ValueError("All workload profile weights must be positive.")
        self._profiles = profiles
        self._cum_weights: list[float] = []
        total = 0.0
        for weight in weights:
            total += weight
            self._cum_weights.append(total)
        self._total_weight = total
        self._rng = random.Random(seed)

    def sample(self) -> WorkloadProfile:
        ticket = self._rng.random() * self._total_weight
        for profile, cum_weight in zip(self._profiles, self._cum_weights, strict=True):
            if ticket < cum_weight:
                return profile
        return self._profiles[-1]

    def sample_prompt(self, profile: WorkloadProfile) -> str:
        prompts = profile.prompts
        if len(prompts) == 1:
            return prompts[0]
        return prompts[self._rng.randrange(len(prompts))]


def default_profiles() -> list[WorkloadProfile]:
    return [
        WorkloadProfile(
            name="chat_short_sync",
            weight=0.30,
            endpoint="chat",
            stream=False,
            max_tokens=48,
            temperature=0.0,
            system_prompt="You are a concise assistant.",
            prompts=tuple(SHORT_PROMPTS),
        ),
        WorkloadProfile(
            name="chat_medium_sync",
            weight=0.24,
            endpoint="chat",
            stream=False,
            max_tokens=96,
            temperature=0.0,
            system_prompt="You are a concise assistant.",
            prompts=tuple(MEDIUM_PROMPTS),
        ),
        WorkloadProfile(
            name="chat_long_sync",
            weight=0.14,
            endpoint="chat",
            stream=False,
            max_tokens=96,
            temperature=0.0,
            system_prompt="You are a concise assistant.",
            prompts=tuple(LONG_PROMPTS),
        ),
        WorkloadProfile(
            name="chat_short_stream",
            weight=0.16,
            endpoint="chat",
            stream=True,
            max_tokens=48,
            temperature=0.0,
            system_prompt="You are a concise assistant.",
            prompts=tuple(SHORT_PROMPTS + MEDIUM_PROMPTS),
        ),
        WorkloadProfile(
            name="completion_medium_sync",
            weight=0.10,
            endpoint="completions",
            stream=False,
            max_tokens=96,
            temperature=0.0,
            system_prompt=None,
            prompts=tuple(MEDIUM_PROMPTS + LONG_COMPLETION_PROMPTS),
        ),
        WorkloadProfile(
            name="completion_short_stream",
            weight=0.06,
            endpoint="completions",
            stream=True,
            max_tokens=48,
            temperature=0.0,
            system_prompt=None,
            prompts=tuple(SHORT_PROMPTS),
        ),
    ]


def load_profiles(path: str | None) -> list[WorkloadProfile]:
    if path is None:
        return default_profiles()
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Workload JSON must be a list of profile objects.")
    profiles: list[WorkloadProfile] = []
    for item in payload:
        if not isinstance(item, dict):
            raise ValueError("Each workload profile must be an object.")
        prompts = item.get("prompts")
        if not isinstance(prompts, list) or not prompts or not all(
            isinstance(prompt, str) for prompt in prompts
        ):
            raise ValueError("Each workload profile must define a non-empty prompts string list.")
        profiles.append(
            WorkloadProfile(
                name=str(item["name"]),
                weight=float(item["weight"]),
                endpoint=str(item["endpoint"]),
                stream=bool(item.get("stream", False)),
                max_tokens=int(item["max_tokens"]),
                temperature=float(item.get("temperature", 0.0)),
                system_prompt=item.get("system_prompt"),
                prompts=tuple(prompts),
            )
        )
    return profiles


async def mixed_worker_loop(
    *,
    worker_id: int,
    users: int,
    scheduler: RequestScheduler,
    global_stats: StatsCollector,
    stats_by_profile: dict[str, StatsCollector],
    mixed_records: list[MixedMetricRecord],
    profile_source: WeightedProfileSource,
    client: httpx.AsyncClient,
    headers: dict[str, str],
    model: str,
    ramp_seconds: float,
) -> None:
    if ramp_seconds > 0:
        await asyncio.sleep(ramp_seconds * worker_id / max(1, users))
    while True:
        request_index = await scheduler.reserve()
        if request_index is None:
            return
        profile = profile_source.sample()
        prompt = profile_source.sample_prompt(profile)
        path = "/v1/chat/completions" if profile.endpoint == "chat" else "/v1/completions"
        payload = build_payload(
            endpoint=profile.endpoint,
            model=model,
            prompt=prompt,
            system_prompt=profile.system_prompt,
            max_tokens=profile.max_tokens,
            temperature=profile.temperature,
            stream=profile.stream,
        )
        global_stats.mark_started()
        stats_by_profile[profile.name].mark_started()
        if profile.stream:
            metric = await run_stream_request(
                client=client,
                url=path,
                headers=headers,
                endpoint=profile.endpoint,
                payload=payload,
                request_index=request_index,
                worker_id=worker_id,
            )
        else:
            metric = await run_sync_request(
                client=client,
                url=path,
                headers=headers,
                endpoint=profile.endpoint,
                payload=payload,
                request_index=request_index,
                worker_id=worker_id,
            )
        global_stats.add(metric)
        stats_by_profile[profile.name].add(metric)
        mixed_records.append(
            MixedMetricRecord(
                profile_name=profile.name,
                endpoint=profile.endpoint,
                stream=profile.stream,
                max_tokens=profile.max_tokens,
                prompt_chars=len(prompt),
                prompt_preview=prompt[:120],
                metric=metric,
            )
        )


async def progress_reporter(stats: StatsCollector, started_at: float, interval_s: float) -> None:
    try:
        while True:
            await asyncio.sleep(interval_s)
            elapsed = time.perf_counter() - started_at
            ok = sum(1 for metric in stats.metrics if metric.ok)
            err = sum(1 for metric in stats.metrics if not metric.ok)
            print(
                f"[progress] elapsed={elapsed:.1f}s started={stats.started_requests} "
                f"completed={stats.completed_requests} active={stats.active_requests} "
                f"ok={ok} err={err}"
            )
    except asyncio.CancelledError:
        return


def print_profile_table(summaries: dict[str, Any]) -> None:
    print("profiles:")
    print("  profile                 ok/total   rps   lat_p50  lat_p95  queue_p50  out_tok_s")
    for name, summary in summaries.items():
        latency_p50 = summary.latency_ms["p50"] if summary.latency_ms else 0.0
        latency_p95 = summary.latency_ms["p95"] if summary.latency_ms else 0.0
        queue_p50 = summary.server_queue_wait_ms["p50"] if summary.server_queue_wait_ms else 0.0
        out_tps = summary.output_tokens_per_second or 0.0
        print(
            f"  {name:20s} {summary.success_requests:3d}/{summary.completed_requests:<5d} "
            f"{summary.requests_per_second:5.2f} {latency_p50:8.1f} {latency_p95:8.1f} "
            f"{queue_p50:10.1f} {out_tps:10.2f}"
        )


def print_mix_config(profiles: list[WorkloadProfile]) -> None:
    print("mix_config:")
    for profile in profiles:
        print(
            f"  - {profile.name}: weight={profile.weight:.2f} "
            f"endpoint={profile.endpoint} stream={int(profile.stream)} "
            f"max_tokens={profile.max_tokens} prompt_pool={len(profile.prompts)}"
        )


def build_detail_rows(records: list[MixedMetricRecord]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        rows.append(
            {
                "profile_name": record.profile_name,
                "endpoint": record.endpoint,
                "stream": record.stream,
                "max_tokens": record.max_tokens,
                "prompt_chars": record.prompt_chars,
                "prompt_preview": record.prompt_preview,
                **asdict(record.metric),
            }
        )
    return rows


async def run_mixed_benchmark(args) -> tuple[Any, dict[str, Any], list[MixedMetricRecord], list[WorkloadProfile]]:
    profiles = load_profiles(args.workload_json)
    profile_source = WeightedProfileSource(profiles, args.seed)
    scheduler = RequestScheduler(args.total_requests, args.duration)
    global_stats = StatsCollector()
    global_stats.max_active_requests = args.users
    stats_by_profile = {profile.name: StatsCollector() for profile in profiles}
    mixed_records: list[MixedMetricRecord] = []

    timeout = httpx.Timeout(args.timeout, connect=args.connect_timeout)
    limits = httpx.Limits(
        max_connections=args.users,
        max_keepalive_connections=args.users,
    )
    started_at = time.perf_counter()
    async with httpx.AsyncClient(
        base_url=args.base_url.rstrip("/"),
        timeout=timeout,
        limits=limits,
    ) as client:
        progress_task = None
        if args.progress_interval > 0:
            progress_task = asyncio.create_task(
                progress_reporter(global_stats, started_at, args.progress_interval)
            )
        try:
            tasks = [
                asyncio.create_task(
                    mixed_worker_loop(
                        worker_id=worker_id,
                        users=args.users,
                        scheduler=scheduler,
                        global_stats=global_stats,
                        stats_by_profile=stats_by_profile,
                        mixed_records=mixed_records,
                        profile_source=profile_source,
                        client=client,
                        headers=request_headers(args.api_key),
                        model=args.model,
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
    overall_summary = summarize_metrics(
        stats=global_stats,
        wall_time_s=wall_time_s,
        config={
            "run_label": getattr(args, "run_label", "mixed"),
            "base_url": args.base_url,
            "model": args.model,
            "users": args.users,
            "total_requests": args.total_requests,
            "duration": args.duration,
            "timeout": args.timeout,
            "connect_timeout": args.connect_timeout,
            "ramp_seconds": args.ramp_seconds,
            "profile_count": len(profiles),
            "mode": "mixed",
        },
    )
    profile_summaries: dict[str, Any] = {}
    for profile in profiles:
        stats = stats_by_profile[profile.name]
        profile_summaries[profile.name] = summarize_metrics(
            stats=stats,
            wall_time_s=wall_time_s,
            config={
                "profile_name": profile.name,
                "endpoint": profile.endpoint,
                "stream": profile.stream,
                "max_tokens": profile.max_tokens,
                "weight": profile.weight,
            },
        )
    return overall_summary, profile_summaries, mixed_records, profiles


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a mixed-workload benchmark against an OpenAI-compatible API.",
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--users", "--concurrency", dest="users", type=int, default=128)
    parser.add_argument("--total-requests", type=int, default=1024)
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--connect-timeout", type=float, default=10.0)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--ramp-seconds", type=float, default=0.0)
    parser.add_argument("--progress-interval", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workload-json", default=None)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--details-jsonl", default=None)
    parser.add_argument("--details-csv", default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    overall_summary, profile_summaries, mixed_records, profiles = asyncio.run(
        run_mixed_benchmark(args)
    )
    print_mix_config(profiles)
    print()
    print_summary(overall_summary)
    print()
    print_profile_table(profile_summaries)
    print()
    print_distribution("overall_server_request_parse_ms", overall_summary.server_request_parse_ms, "ms")
    print_distribution("overall_server_queue_wait_ms", overall_summary.server_queue_wait_ms, "ms")
    if args.output_json:
        payload = {
            "overall": asdict(overall_summary),
            "profiles": {name: asdict(summary) for name, summary in profile_summaries.items()},
            "mix_config": [asdict(profile) for profile in profiles],
        }
        Path(args.output_json).write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    rows = build_detail_rows(mixed_records)
    if args.details_jsonl:
        write_request_jsonl(args.details_jsonl, rows)
    if args.details_csv:
        write_request_csv(args.details_csv, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
