#!/usr/bin/env python3
import argparse
import asyncio

from benchmark_openai_api import (
    build_metric_records,
    print_summary,
    run_single_benchmark,
    write_request_csv,
    write_request_jsonl,
    write_summary_output,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a large-request OpenAI API stress test and optionally fail on health thresholds.",
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--endpoint", choices=["chat", "completions"], default="completions")
    parser.add_argument("--users", "--concurrency", dest="users", type=int, default=128)
    parser.add_argument("--total-requests", type=int, default=50000)
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
    parser.add_argument("--progress-interval", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--details-jsonl", default=None)
    parser.add_argument("--details-csv", default=None)
    parser.add_argument("--min-success-rate", type=float, default=1.0)
    parser.add_argument("--max-error-requests", type=int, default=0)
    parser.add_argument("--max-p95-latency-ms", type=float, default=None)
    parser.add_argument("--max-p95-queue-wait-ms", type=float, default=None)
    return parser


def _dist_value(dist: dict[str, float] | None, key: str) -> float | None:
    if not dist:
        return None
    return dist.get(key)


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    args.run_label = f"stress-users={args.users}"
    summary, metrics = asyncio.run(run_single_benchmark(args))
    print_summary(summary)
    if args.output_json:
        write_summary_output(args.output_json, [summary])
    if args.details_jsonl:
        write_request_jsonl(args.details_jsonl, build_metric_records(summary, metrics))
    if args.details_csv:
        write_request_csv(args.details_csv, build_metric_records(summary, metrics))

    success_rate = (summary.success_requests / summary.completed_requests) if summary.completed_requests else 0.0
    failures = []
    if success_rate < args.min_success_rate:
        failures.append(
            f"success_rate={success_rate:.6f} below min_success_rate={args.min_success_rate:.6f}"
        )
    if summary.error_requests > args.max_error_requests:
        failures.append(
            f"error_requests={summary.error_requests} above max_error_requests={args.max_error_requests}"
        )
    latency_p95 = _dist_value(summary.latency_ms, "p95")
    if args.max_p95_latency_ms is not None and latency_p95 is not None and latency_p95 > args.max_p95_latency_ms:
        failures.append(
            f"latency_p95_ms={latency_p95:.2f} above max_p95_latency_ms={args.max_p95_latency_ms:.2f}"
        )
    queue_p95 = _dist_value(summary.server_queue_wait_ms, "p95")
    if (
        args.max_p95_queue_wait_ms is not None
        and queue_p95 is not None
        and queue_p95 > args.max_p95_queue_wait_ms
    ):
        failures.append(
            f"queue_wait_p95_ms={queue_p95:.2f} above max_p95_queue_wait_ms={args.max_p95_queue_wait_ms:.2f}"
        )

    if failures:
        print("stress_verdict: FAIL")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("stress_verdict: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
