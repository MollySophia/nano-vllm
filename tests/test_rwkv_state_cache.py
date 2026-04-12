import unittest
from types import SimpleNamespace

from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.state_cache import StatePrefixIndex, StateSlotManager
from nanovllm.sampling_params import SamplingParams


def _config(num_state_blocks: int = 8):
    return SimpleNamespace(
        max_num_seqs=16,
        max_num_batched_tokens=4096,
        rwkv_prefill_max_batch_size=16,
        rwkv_prefill_token_budget=4096,
        eos=0,
        rwkv_state_cache_enable=True,
        num_state_blocks=num_state_blocks,
    )


def _seq(token_ids: list[int], max_tokens: int = 4) -> Sequence:
    return Sequence(
        token_ids,
        SamplingParams(
            temperature=0.0,
            ignore_eos=True,
            max_tokens=max_tokens,
        ),
    )


class RWKVStateCacheTest(unittest.TestCase):
    def test_slot_manager_lru_skips_pinned_slot(self):
        slots = StateSlotManager(2)
        index = StatePrefixIndex()

        a = slots.allocate_writable_slot(requires_zero_init=True)
        key_a = index.insert([1, 2], 2, a.slot_id)
        slots.mark_cached(a.slot_id, key_a, 2)

        b = slots.allocate_writable_slot(requires_zero_init=True)
        key_b = index.insert([3, 4], 2, b.slot_id)
        slots.mark_cached(b.slot_id, key_b, 2)

        slots.pin_cached(a.slot_id)
        reused = slots.allocate_writable_slot(requires_zero_init=False)

        self.assertIsNotNone(reused)
        self.assertEqual(reused.slot_id, b.slot_id)
        self.assertTrue(reused.requires_zero_init is False)
        self.assertEqual(slots.slot_meta[a.slot_id].state.name, "CACHED_PINNED")
        self.assertEqual(slots.slot_meta[b.slot_id].state.name, "LIVE")

    def test_scheduler_exact_hit_reuses_cached_prompt_slot_and_allocates_new_live_slot(self):
        scheduler = Scheduler(_config())
        source = scheduler.slot_manager.allocate_writable_slot(requires_zero_init=True)
        cache_key = scheduler.prefix_index.insert([1, 2, 3, 4], 4, source.slot_id)
        scheduler.slot_manager.mark_cached(source.slot_id, cache_key, 4)

        seq = _seq([1, 2, 3, 4])
        scheduler.add(seq)
        seqs, is_prefill = scheduler.schedule()

        self.assertTrue(is_prefill)
        self.assertEqual(len(seqs), 1)
        scheduled = seqs[0]
        self.assertTrue(scheduled.exact_cache_hit)
        self.assertEqual(scheduled.cached_prefix_len, 4)
        self.assertEqual(scheduled.prompt_cache_slot, source.slot_id)
        self.assertEqual(scheduled.cache_hit_slot, source.slot_id)
        self.assertNotEqual(scheduled.state_slot, source.slot_id)

    def test_scheduler_partial_hit_allocates_new_prompt_cache_and_live_slots(self):
        scheduler = Scheduler(_config())
        source = scheduler.slot_manager.allocate_writable_slot(requires_zero_init=True)
        cache_key = scheduler.prefix_index.insert([1, 2, 3], 3, source.slot_id)
        scheduler.slot_manager.mark_cached(source.slot_id, cache_key, 3)

        seq = _seq([1, 2, 3, 4, 5])
        scheduler.add(seq)
        seqs, is_prefill = scheduler.schedule()

        self.assertTrue(is_prefill)
        self.assertEqual(len(seqs), 1)
        scheduled = seqs[0]
        self.assertFalse(scheduled.exact_cache_hit)
        self.assertEqual(scheduled.cached_prefix_len, 3)
        self.assertEqual(scheduled.cache_hit_slot, source.slot_id)
        self.assertNotEqual(scheduled.prompt_cache_slot, source.slot_id)
        self.assertNotEqual(scheduled.state_slot, source.slot_id)
        self.assertNotEqual(scheduled.state_slot, scheduled.prompt_cache_slot)


if __name__ == "__main__":
    unittest.main()
