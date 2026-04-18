#!/usr/bin/env python3
import argparse
import asyncio
import sys

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
        description="Run a concurrency sweep against the OpenAI-compatible API and print a compact performance table.",
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--endpoint", choices=["chat", "completions"], default="completions")
    parser.add_argument("--users-sweep", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64, 128])
    parser.add_argument("--total-requests", type=int, default=256)
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=128)
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
    parser.add_argument("--fail-on-errors", action="store_true")
    return parser


def _fmt(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def _dist_value(dist: dict[str, float] | None, key: str) -> float | None:
    if not dist:
        return None
    return dist.get(key)


def print_perf_table(summaries) -> None:
    if not summaries:
        return
    print("perf_sweep:")
    print("  users  ok/total  rps   lat_p50  lat_p95  queue_p50  ttft_p50  srv_dec_p50  out_tok_s")
    for summary in summaries:
        print(
            "  "
            f"{summary.config['users']:5d}  "
            f"{summary.success_requests:3d}/{summary.completed_requests:<5d} "
            f"{summary.requests_per_second:5.2f}  "
            f"{_fmt(_dist_value(summary.latency_ms, 'p50')):>7}  "
            f"{_fmt(_dist_value(summary.latency_ms, 'p95')):>7}  "
            f"{_fmt(_dist_value(summary.server_queue_wait_ms, 'p50')):>9}  "
            f"{_fmt(_dist_value(summary.client_ttft_ms, 'p50')):>8}  "
            f"{_fmt(_dist_value(summary.server_decode_tps, 'p50')):>11}  "
            f"{_fmt(summary.output_tokens_per_second):>9}"
        )


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    summaries = []
    detail_rows = []
    for users in args.users_sweep:
        run_args = argparse.Namespace(**vars(args))
        run_args.users = users
        run_args.run_label = f"users={users}"
        summary, metrics = asyncio.run(run_single_benchmark(run_args))
        print(f"\n== {run_args.run_label} ==")
        print_summary(summary)
        summaries.append(summary)
        if args.details_jsonl or args.details_csv:
            detail_rows.extend(build_metric_records(summary, metrics))
    print()
    print_perf_table(summaries)
    if args.output_json:
        write_summary_output(args.output_json, summaries)
    if args.details_jsonl:
        write_request_jsonl(args.details_jsonl, detail_rows)
    if args.details_csv:
        write_request_csv(args.details_csv, detail_rows)
    if args.fail_on_errors and any(summary.error_requests > 0 for summary in summaries):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
