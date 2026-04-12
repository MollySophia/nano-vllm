import unittest
from types import SimpleNamespace

from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.sequence import SequenceStatus
from nanovllm.sampling_params import SamplingParams


def _config(num_state_blocks: int = 2):
    return SimpleNamespace(
        max_num_seqs=16,
        max_num_batched_tokens=4096,
        rwkv_prefill_max_batch_size=16,
        rwkv_prefill_token_budget=4096,
        eos=0,
        rwkv_state_cache_enable=False,
        num_state_blocks=num_state_blocks,
    )


class LegacySchedulerTest(unittest.TestCase):
    def test_prefill_decode_and_finish_release_block(self):
        scheduler = Scheduler(_config(num_state_blocks=1))
        request = Sequence(
            [1, 2, 3],
            SamplingParams(
                temperature=0.0,
                ignore_eos=False,
                max_tokens=2,
            ),
        )
        scheduler.add(request)

        seqs, is_prefill = scheduler.schedule()
        self.assertTrue(is_prefill)
        self.assertEqual(seqs, [request])
        self.assertEqual(request.block_table, [0])
        self.assertEqual(request.status, SequenceStatus.RUNNING)
        self.assertEqual(list(scheduler.block_manager.free_block_ids), [])

        scheduler.postprocess(seqs, [5])
        self.assertEqual(request.num_completion_tokens, 1)
        self.assertEqual(request.status, SequenceStatus.RUNNING)
        self.assertEqual(request.block_table, [0])

        seqs, is_prefill = scheduler.schedule()
        self.assertFalse(is_prefill)
        self.assertEqual(seqs, [request])

        scheduler.postprocess(seqs, [0])
        self.assertTrue(request.is_finished)
        self.assertEqual(request.status, SequenceStatus.FINISHED)
        self.assertEqual(request.block_table, [])
        self.assertEqual(list(scheduler.block_manager.free_block_ids), [0])
        self.assertFalse(scheduler.running)


if __name__ == "__main__":
    unittest.main()
