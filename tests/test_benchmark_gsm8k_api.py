import json
import tempfile
import unittest
from pathlib import Path

import benchmark_gsm8k_api as bench


class BenchmarkGSM8KAPIHelpersTest(unittest.TestCase):
    def test_load_gsm8k_samples_builds_legacy_prompts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "gsm8k.jsonl"
            path.write_text(
                json.dumps({"problem": "1+1?", "answer": "#### 2"}, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            samples = bench.load_gsm8k_samples(str(path), limit=0)

        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0].gold_answer, "2")
        self.assertIn("Problem: 1+1?", samples[0].prompt)

    def test_sample_source_cycles_or_randomizes(self):
        samples = [
            bench.GSM8KSample(sample_index=0, problem="a", gold_answer="1", prompt="pa"),
            bench.GSM8KSample(sample_index=1, problem="b", gold_answer="2", prompt="pb"),
        ]

        cycle = bench.SampleSource(samples, seed=0, random_sample=False)
        random_source = bench.SampleSource(samples, seed=0, random_sample=True)

        self.assertEqual(cycle.sample(0).sample_index, 0)
        self.assertEqual(cycle.sample(1).sample_index, 1)
        self.assertEqual(cycle.sample(2).sample_index, 0)
        self.assertIn(random_source.sample(0).sample_index, {0, 1})

    def test_score_response_text_extracts_and_compares(self):
        extracted, correct = bench.score_response_text("reasoning\n#### 42", "42")

        self.assertEqual(extracted, "42")
        self.assertTrue(correct)

    def test_summarize_gsm8k_metrics(self):
        metrics = [
            bench.GSM8KRequestMetrics(
                request_index=0,
                worker_id=0,
                ok=True,
                status_code=200,
                error=None,
                latency_s=0.1,
                ttft_s=None,
                response_bytes=10,
                response_chars=4,
                request_id="r1",
                finish_reason="stop",
                prompt_tokens=3,
                completion_tokens=2,
                total_tokens=5,
                server_processing_ms=None,
                server_queue_wait_ms=None,
                server_ttft_ms=None,
                server_generation_ms=None,
                server_total_ms=None,
                server_output_tps=None,
                server_decode_tps=None,
                sample_index=0,
                gold_answer="42",
                extracted_answer="42",
                is_correct=True,
            ),
            bench.GSM8KRequestMetrics(
                request_index=1,
                worker_id=0,
                ok=True,
                status_code=200,
                error=None,
                latency_s=0.1,
                ttft_s=None,
                response_bytes=10,
                response_chars=4,
                request_id="r2",
                finish_reason="stop",
                prompt_tokens=3,
                completion_tokens=2,
                total_tokens=5,
                server_processing_ms=None,
                server_queue_wait_ms=None,
                server_ttft_ms=None,
                server_generation_ms=None,
                server_total_ms=None,
                server_output_tps=None,
                server_decode_tps=None,
                sample_index=1,
                gold_answer="7",
                extracted_answer=None,
                is_correct=False,
            ),
        ]

        summary = bench.summarize_gsm8k_metrics(metrics)

        self.assertEqual(summary["success_requests"], 2)
        self.assertEqual(summary["unique_samples_seen"], 2)
        self.assertEqual(summary["extract_count"], 1)
        self.assertEqual(summary["correct_count"], 1)
        self.assertAlmostEqual(summary["extract_rate"], 50.0)
        self.assertAlmostEqual(summary["accuracy"], 50.0)


if __name__ == "__main__":
    unittest.main()
