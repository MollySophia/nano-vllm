import unittest
from types import SimpleNamespace

from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence, SequenceStatus


def _dummy_config(**overrides):
    base = dict(
        max_num_seqs=16,
        max_num_batched_tokens=1024,
        rwkv_prefill_max_batch_size=16,
        rwkv_prefill_token_budget=1024,
        rwkv_prefill_chunk_size=4,
        eos=-1,
        rwkv_state_cache_enable=False,
        num_state_blocks=64,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


class SchedulerChunkedPrefillTest(unittest.TestCase):
    def test_postprocess_skips_none_token_without_mutating_sequence(self):
        scheduler = Scheduler(_dummy_config())
        seq = Sequence(list(range(8)))
        seq.status = SequenceStatus.RUNNING
        seq.block_table = [0]
        seq.num_cached_tokens = 4
        scheduler.running.append(seq)

        scheduler.postprocess([seq], [None])

        self.assertEqual(seq.num_tokens, 8)
        self.assertEqual(seq.num_completion_tokens, 0)
        self.assertEqual(seq.status, SequenceStatus.RUNNING)
        self.assertIn(seq, scheduler.running)

    def test_schedule_prefers_running_partial_prefill_before_decode(self):
        scheduler = Scheduler(_dummy_config())
        partial = Sequence(list(range(10)))
        partial.status = SequenceStatus.RUNNING
        partial.block_table = [0]
        partial.num_cached_tokens = 4
        scheduler.running.append(partial)

        ready = Sequence(list(range(6)))
        ready.status = SequenceStatus.RUNNING
        ready.block_table = [1]
        ready.num_cached_tokens = ready.num_prompt_tokens
        scheduler.running.append(ready)

        scheduled, is_prefill = scheduler.schedule()

        self.assertTrue(is_prefill)
        self.assertEqual(scheduled, [partial])


if __name__ == "__main__":
    unittest.main()
