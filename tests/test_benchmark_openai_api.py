import argparse
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import httpx

import benchmark_openai_api as bench


try:
    from tests.openai_api_test_utils import patched_app
except ModuleNotFoundError as exc:  # pragma: no cover - depends on optional API deps.
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


class BenchmarkOpenAIAPIHelpersTest(unittest.TestCase):
    def test_build_payload_for_chat_and_completions(self):
        chat_payload = bench.build_payload(
            endpoint="chat",
            model="rwkv-test",
            prompt="hello",
            system_prompt="be terse",
            max_tokens=12,
            temperature=0.2,
            stream=True,
        )
        completion_payload = bench.build_payload(
            endpoint="completions",
            model="rwkv-test",
            prompt="hello",
            system_prompt=None,
            max_tokens=8,
            temperature=0.0,
            stream=False,
        )

        self.assertEqual(
            chat_payload,
            {
                "model": "rwkv-test",
                "messages": [
                    {"role": "system", "content": "be terse"},
                    {"role": "user", "content": "hello"},
                ],
                "max_tokens": 12,
                "temperature": 0.2,
                "stream": True,
            },
        )
        self.assertEqual(
            completion_payload,
            {
                "model": "rwkv-test",
                "prompt": "hello",
                "max_tokens": 8,
                "temperature": 0.0,
                "stream": False,
            },
        )

    def test_parse_sync_body_for_chat_and_completions(self):
        completion = {
            "choices": [{"text": "OK", "finish_reason": "stop"}],
        }
        chat = {
            "choices": [{"message": {"content": "CHAT"}, "finish_reason": "length"}],
        }

        self.assertEqual(bench.parse_sync_body("completions", completion), (2, "OK", "stop"))
        self.assertEqual(bench.parse_sync_body("chat", chat), (4, "CHAT", "length"))

    def test_load_prompts_supports_plaintext_jsonl_and_defaults(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            txt_path = Path(tmpdir) / "prompts.txt"
            txt_path.write_text("a\n\nb\n", encoding="utf-8")
            jsonl_path = Path(tmpdir) / "prompts.jsonl"
            jsonl_path.write_text(
                json.dumps({"prompt": "c"}, ensure_ascii=False) + "\n"
                + json.dumps({"text": "d"}, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

            txt_prompts = bench.load_prompts(str(txt_path), ["inline"], 2)
            jsonl_prompts = bench.load_prompts(str(jsonl_path), [], 1)
            default_prompts = bench.load_prompts(None, [], 1)

        self.assertEqual(txt_prompts, ["inline", "a", "b", "inline", "a", "b"])
        self.assertEqual(jsonl_prompts, ["c", "d"])
        self.assertEqual(default_prompts, bench.DEFAULT_PROMPTS)

    def test_summarize_metrics_and_build_metric_records(self):
        stats = bench.StatsCollector()
        stats.mark_started()
        stats.add(
            bench.RequestMetrics(
                request_index=0,
                worker_id=0,
                ok=True,
                status_code=200,
                error=None,
                latency_s=0.1,
                ttft_s=0.04,
                response_bytes=30,
                response_chars=5,
                request_id="req_1",
                finish_reason="stop",
                prompt_tokens=3,
                completion_tokens=2,
                total_tokens=5,
                server_processing_ms=10.0,
                server_queue_wait_ms=1.0,
                server_ttft_ms=4.0,
                server_generation_ms=8.0,
                server_total_ms=10.0,
                server_output_tps=250.0,
                server_decode_tps=300.0,
            )
        )
        stats.mark_started()
        stats.add(
            bench.RequestMetrics(
                request_index=1,
                worker_id=1,
                ok=False,
                status_code=400,
                error="bad request",
                latency_s=0.2,
                ttft_s=None,
                response_bytes=10,
                response_chars=0,
                request_id="req_2",
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
            )
        )

        summary = bench.summarize_metrics(
            stats=stats,
            wall_time_s=2.0,
            config={"run_label": "users=2", "users": 2, "endpoint": "chat"},
        )
        rows = bench.build_metric_records(summary, stats.metrics)

        self.assertEqual(summary.success_requests, 1)
        self.assertEqual(summary.error_requests, 1)
        self.assertEqual(summary.status_codes, {"200": 1, "400": 1})
        self.assertEqual(summary.input_tokens_total, 3)
        self.assertEqual(summary.output_tokens_total, 2)
        self.assertEqual(summary.sample_errors, ["bad request"])
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["run_label"], "users=2")
        self.assertEqual(rows[0]["endpoint"], "chat")
        self.assertEqual(rows[0]["request_id"], "req_1")

    def test_write_summary_and_request_outputs(self):
        summary = bench.BenchmarkSummary(
            config={"users": 1},
            wall_time_s=1.0,
            started_requests=1,
            completed_requests=1,
            success_requests=1,
            error_requests=0,
            requests_per_second=1.0,
            success_requests_per_second=1.0,
            input_tokens_total=3,
            output_tokens_total=2,
            input_tokens_per_second=3.0,
            output_tokens_per_second=2.0,
            output_chars_total=2,
            output_chars_per_second=2.0,
            status_codes={"200": 1},
            latency_ms={"mean": 10.0, "min": 10.0, "p50": 10.0, "p90": 10.0, "p95": 10.0, "p99": 10.0, "max": 10.0, "count": 1.0},
            client_ttft_ms=None,
            server_processing_ms=None,
            server_queue_wait_ms=None,
            server_ttft_ms=None,
            server_generation_ms=None,
            server_total_ms=None,
            server_output_tps=None,
            server_decode_tps=None,
            sample_errors=[],
        )
        rows = [{"request_index": 0, "text": "ok"}]
        with tempfile.TemporaryDirectory() as tmpdir:
            summary_path = Path(tmpdir) / "summary.json"
            jsonl_path = Path(tmpdir) / "details.jsonl"
            csv_path = Path(tmpdir) / "details.csv"
            bench.write_summary_output(str(summary_path), [summary])
            bench.write_request_jsonl(str(jsonl_path), rows)
            bench.write_request_csv(str(csv_path), rows)

            summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
            jsonl_lines = jsonl_path.read_text(encoding="utf-8").splitlines()
            csv_text = csv_path.read_text(encoding="utf-8")

        self.assertEqual(summary_payload["success_requests"], 1)
        self.assertEqual(json.loads(jsonl_lines[0]), rows[0])
        self.assertIn("request_index,text", csv_text)
        self.assertIn("0,ok", csv_text)


class BenchmarkOpenAIAPIIntegrationTest(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        if IMPORT_ERROR is not None:
            raise unittest.SkipTest(f"OpenAI API test dependencies unavailable: {IMPORT_ERROR}")

    async def test_run_sync_request_parses_usage_and_headers_from_api(self):
        with patched_app(completion_text="OK") as (app, _llm, _factory):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
                metric = await bench.run_sync_request(
                    client=client,
                    url="http://testserver/v1/completions",
                    headers={"Content-Type": "application/json"},
                    endpoint="completions",
                    payload={
                        "model": "rwkv-test",
                        "prompt": "abc",
                        "max_tokens": 4,
                        "temperature": 0.0,
                        "stream": False,
                    },
                    request_index=0,
                    worker_id=0,
                )

        self.assertTrue(metric.ok)
        self.assertEqual(metric.status_code, 200)
        self.assertEqual(metric.response_chars, 2)
        self.assertEqual(metric.finish_reason, "stop")
        self.assertEqual(metric.prompt_tokens, 3)
        self.assertEqual(metric.completion_tokens, 2)
        self.assertEqual(metric.total_tokens, 5)
        self.assertIsNotNone(metric.server_processing_ms)
        self.assertIsNotNone(metric.server_output_tps)

    async def test_run_stream_request_parses_chat_stream_deltas(self):
        with patched_app(chat_text="CHAT", per_token_delay_s=0.01) as (app, _llm, _factory):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
                metric = await bench.run_stream_request(
                    client=client,
                    url="http://testserver/v1/chat/completions",
                    headers={"Content-Type": "application/json"},
                    endpoint="chat",
                    payload={
                        "model": "rwkv-test",
                        "messages": [{"role": "user", "content": "hello"}],
                        "max_tokens": 6,
                        "temperature": 0.0,
                        "stream": True,
                    },
                    request_index=0,
                    worker_id=0,
                )

        self.assertTrue(metric.ok)
        self.assertEqual(metric.status_code, 200)
        self.assertEqual(metric.response_chars, 4)
        self.assertEqual(metric.finish_reason, "stop")
        self.assertEqual(metric.prompt_tokens, len("User: hello\n\nAssistant:"))
        self.assertIsNotNone(metric.ttft_s)
        self.assertGreater(metric.ttft_s, 0.0)

    async def test_run_single_benchmark_executes_multi_user_loop_against_local_app(self):
        with patched_app(completion_text="OK") as (app, _llm, _factory):
            real_async_client = httpx.AsyncClient

            def make_client(*args, **kwargs):
                kwargs["transport"] = httpx.ASGITransport(app=app)
                return real_async_client(*args, **kwargs)

            args = argparse.Namespace(
                base_url="http://testserver",
                model="rwkv-test",
                endpoint="completions",
                users=2,
                total_requests=4,
                duration=None,
                stream=False,
                max_tokens=4,
                temperature=0.0,
                system_prompt=None,
                prompt=["alpha", "beta"],
                prompt_file=None,
                prompt_repeat=1,
                api_key=None,
                timeout=5.0,
                connect_timeout=1.0,
                ramp_seconds=0.0,
                progress_interval=0.0,
                seed=0,
                run_label="users=2",
            )
            with mock.patch.object(bench.httpx, "AsyncClient", side_effect=make_client):
                summary, metrics = await bench.run_single_benchmark(args)

        self.assertEqual(summary.config["users"], 2)
        self.assertEqual(summary.started_requests, 4)
        self.assertEqual(summary.completed_requests, 4)
        self.assertEqual(summary.success_requests, 4)
        self.assertEqual(summary.error_requests, 0)
        self.assertEqual(len(metrics), 4)
        self.assertGreater(summary.requests_per_second, 0.0)
        self.assertEqual(summary.output_tokens_total, 8)

    async def test_run_single_benchmark_streaming_collects_ttft_and_partial_metrics(self):
        with patched_app(chat_text="CHAT", per_token_delay_s=0.01) as (app, _llm, _factory):
            real_async_client = httpx.AsyncClient

            def make_client(*args, **kwargs):
                kwargs["transport"] = httpx.ASGITransport(app=app)
                return real_async_client(*args, **kwargs)

            args = argparse.Namespace(
                base_url="http://testserver",
                model="rwkv-test",
                endpoint="chat",
                users=3,
                total_requests=6,
                duration=None,
                stream=True,
                max_tokens=8,
                temperature=0.0,
                system_prompt=None,
                prompt=["alpha", "beta"],
                prompt_file=None,
                prompt_repeat=1,
                api_key=None,
                timeout=5.0,
                connect_timeout=1.0,
                ramp_seconds=0.0,
                progress_interval=0.0,
                seed=0,
                run_label="users=3",
            )
            with mock.patch.object(bench.httpx, "AsyncClient", side_effect=make_client):
                summary, metrics = await bench.run_single_benchmark(args)

        self.assertEqual(summary.success_requests, 6)
        self.assertEqual(summary.error_requests, 0)
        self.assertEqual(len(metrics), 6)
        self.assertIsNotNone(summary.client_ttft_ms)
        self.assertIsNotNone(summary.server_processing_ms)
        self.assertIsNotNone(summary.server_queue_wait_ms)
        self.assertIsNotNone(summary.server_ttft_ms)
        self.assertIsNone(summary.server_total_ms)
        self.assertIsNone(summary.server_output_tps)
        self.assertIsNone(summary.output_tokens_total)
        self.assertEqual(summary.output_chars_total, 24)


class BenchmarkOpenAIAPICLITest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if IMPORT_ERROR is not None:
            raise unittest.SkipTest(f"OpenAI API test dependencies unavailable: {IMPORT_ERROR}")

    def test_main_users_sweep_writes_aggregated_outputs(self):
        with patched_app(completion_text="OK") as (app, _llm, _factory):
            real_async_client = httpx.AsyncClient

            def make_client(*args, **kwargs):
                kwargs["transport"] = httpx.ASGITransport(app=app)
                return real_async_client(*args, **kwargs)

            with tempfile.TemporaryDirectory() as tmpdir:
                summary_path = Path(tmpdir) / "summary.json"
                jsonl_path = Path(tmpdir) / "details.jsonl"
                csv_path = Path(tmpdir) / "details.csv"
                argv = [
                    "benchmark_openai_api.py",
                    "--base-url",
                    "http://testserver",
                    "--model",
                    "rwkv-test",
                    "--endpoint",
                    "completions",
                    "--users-sweep",
                    "1",
                    "2",
                    "--total-requests",
                    "4",
                    "--max-tokens",
                    "4",
                    "--prompt",
                    "alpha",
                    "--progress-interval",
                    "0",
                    "--output-json",
                    str(summary_path),
                    "--details-jsonl",
                    str(jsonl_path),
                    "--details-csv",
                    str(csv_path),
                ]
                with (
                    mock.patch.object(bench.httpx, "AsyncClient", side_effect=make_client),
                    mock.patch.object(sys, "argv", argv),
                    mock.patch("sys.stdout", new_callable=io.StringIO) as stdout,
                ):
                    bench.main()

                summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
                jsonl_rows = [
                    json.loads(line)
                    for line in jsonl_path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
                csv_lines = csv_path.read_text(encoding="utf-8").splitlines()

        self.assertIn("runs", summary_payload)
        self.assertEqual([run["config"]["users"] for run in summary_payload["runs"]], [1, 2])
        self.assertEqual([run["success_requests"] for run in summary_payload["runs"]], [4, 4])
        self.assertEqual(len(jsonl_rows), 8)
        self.assertEqual({row["run_label"] for row in jsonl_rows}, {"users=1", "users=2"})
        self.assertEqual(len(csv_lines), 9)
        self.assertIn("sweep:", stdout.getvalue())
        self.assertIn("== users=1 ==", stdout.getvalue())
        self.assertIn("== users=2 ==", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
