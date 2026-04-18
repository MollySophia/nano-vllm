import unittest

from nanovllm.engine.model_runner import ModelRunProfiler, _build_prefill_bucket_plan


class ModelRunnerPrefillBucketingTest(unittest.TestCase):
    def test_bucket_plan_groups_by_descending_length_and_preserves_order_within_bucket(self):
        buckets, logical_tokens, flat_padded_tokens, bucketed_padded_tokens = _build_prefill_bucket_plan([64, 256, 64, 16, 256])

        self.assertEqual(buckets, [[1, 4], [0, 2], [3]])
        self.assertEqual(logical_tokens, 656)
        self.assertEqual(flat_padded_tokens, 1280)
        self.assertEqual(bucketed_padded_tokens, 656)

    def test_bucket_plan_skips_zero_length_entries(self):
        buckets, logical_tokens, flat_padded_tokens, bucketed_padded_tokens = _build_prefill_bucket_plan([0, 32, 0, 8])

        self.assertEqual(buckets, [[1], [3]])
        self.assertEqual(logical_tokens, 40)
        self.assertEqual(flat_padded_tokens, 64)
        self.assertEqual(bucketed_padded_tokens, 40)

    def test_profiler_accumulates_prefill_padding_metrics(self):
        profiler = ModelRunProfiler(label="test")

        profiler.record_step(
            kind="prefill",
            seq_count=3,
            total_s=1.0,
            prepare_s=0.2,
            forward_s=0.5,
            sample_s=0.1,
            post_s=0.1,
            prefill_exec_batches=2,
            prefill_logical_tokens=320,
            prefill_flat_padded_tokens=768,
            prefill_bucketed_padded_tokens=320,
        )
        profiler.record_step(
            kind="decode",
            seq_count=3,
            total_s=0.5,
            prepare_s=0.1,
            forward_s=0.2,
            sample_s=0.1,
            post_s=0.1,
        )

        self.assertEqual(profiler.prefill_exec_batches, 2)
        self.assertEqual(profiler.prefill_logical_tokens, 320)
        self.assertEqual(profiler.prefill_flat_padded_tokens, 768)
        self.assertEqual(profiler.prefill_bucketed_padded_tokens, 320)
        self.assertEqual(profiler.step_counts["prefill"], 1)
        self.assertEqual(profiler.step_counts["decode"], 1)


if __name__ == "__main__":
    unittest.main()
