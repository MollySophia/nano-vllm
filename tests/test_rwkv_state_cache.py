import unittest
from types import SimpleNamespace

from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.state_cache import StatePrefixIndex, StateSlotManager
from nanovllm.sampling_params import SamplingParams


def _config(num_state_blocks: int = 8, stop_token_seqs=()):
    return SimpleNamespace(
        max_num_seqs=16,
        max_num_batched_tokens=4096,
        rwkv_prefill_max_batch_size=16,
        rwkv_prefill_token_budget=4096,
        rwkv_prefill_chunk_size=-1,
        eos=0,
        stop_token_seqs=stop_token_seqs,
        rwkv_state_cache_enable=True,
        num_state_blocks=num_state_blocks,
    )


def _seq(token_ids: list[int], max_tokens: int = 4, ignore_eos: bool = True) -> Sequence:
    return Sequence(
        token_ids,
        SamplingParams(
            temperature=0.0,
            ignore_eos=ignore_eos,
            max_tokens=max_tokens,
        ),
    )


class RWKVStateCacheTest(unittest.TestCase):
    def test_prefix_index_reinsert_same_slot_replaces_old_key(self):
        index = StatePrefixIndex()

        old_key = index.insert([1, 2], 2, 0)
        self.assertEqual(old_key, (1, 2))
        old_hit = index.lookup([1, 2, 9])
        self.assertIsNotNone(old_hit)
        self.assertEqual(old_hit.slot_id, 0)
        self.assertEqual(old_hit.prefix_len, 2)

        new_key = index.insert([3, 4], 2, 0)
        self.assertEqual(new_key, (3, 4))
        self.assertIsNone(index.lookup([1, 2, 9]))

        new_hit = index.lookup([3, 4, 5])
        self.assertIsNotNone(new_hit)
        self.assertEqual(new_hit.slot_id, 0)
        self.assertEqual(new_hit.prefix_len, 2)

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

    def test_prefix_index_insert_canonicalizes_rwkv_mobile_cache_key_suffixes(self):
        from nanovllm.tokenizers import RWKVTokenizer

        index = StatePrefixIndex(cache_key_token_rewriter=RWKVTokenizer.canonicalize_state_cache_token_ids)

        cache_key = index.insert([7, 10080, 261, 8], 4, 0)

        self.assertEqual(cache_key, (7, 28329, 11, 8))

        canonical_hit = index.lookup([7, 28329, 11, 8, 9])
        self.assertIsNotNone(canonical_hit)
        assert canonical_hit is not None
        self.assertEqual(canonical_hit.slot_id, 0)
        self.assertEqual(canonical_hit.prefix_len, 4)

        raw_hit = index.lookup([7, 10080, 261, 8, 9])
        self.assertIsNotNone(raw_hit)
        assert raw_hit is not None
        self.assertEqual(raw_hit.slot_id, 0)
        self.assertEqual(raw_hit.prefix_len, 4)

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

    def test_hidden_eos_requires_finalize_decode_before_finish(self):
        scheduler = Scheduler(_config())
        seq = _seq([1, 2, 3], max_tokens=4, ignore_eos=False)
        scheduler.add(seq)

        seqs, is_prefill = scheduler.schedule()
        self.assertTrue(is_prefill)

        scheduler.postprocess(seqs, [0])
        self.assertFalse(seq.is_finished)
        self.assertTrue(seq.pending_hidden_finalize)
        self.assertEqual(seq.raw_completion_token_ids, [0])
        self.assertEqual(seq.completion_token_ids, [])

        seq.num_cached_tokens = seq.num_prompt_tokens
        decode_seqs, is_prefill = scheduler.schedule()
        self.assertFalse(is_prefill)
        self.assertEqual(decode_seqs, [seq])

        scheduler.postprocess(decode_seqs, [None])
        self.assertTrue(seq.is_finished)
        self.assertFalse(seq.pending_hidden_finalize)

    def test_decode_schedules_hidden_finalize_before_regular_decode(self):
        scheduler = Scheduler(_config())
        finalize_seq = _seq([1, 2, 3], max_tokens=4, ignore_eos=False)
        regular_seq = _seq([4, 5, 6], max_tokens=4, ignore_eos=False)
        for slot_id, seq in enumerate((finalize_seq, regular_seq)):
            seq.num_cached_tokens = seq.num_prompt_tokens
            seq.state_slot = slot_id
        finalize_seq.status = regular_seq.status = SequenceStatus.RUNNING
        finalize_seq.pending_hidden_finalize = True
        scheduler.running.append(finalize_seq)
        scheduler.running.append(regular_seq)

        scheduled = scheduler.schedule_decode_only()

        self.assertEqual(scheduled, [finalize_seq])
        self.assertEqual(list(scheduler.running), [finalize_seq, regular_seq])

    def test_scheduler_preempt_releases_live_slots_and_unpins_cache_hit(self):
        scheduler = Scheduler(_config())
        source = scheduler.slot_manager.allocate_writable_slot(requires_zero_init=True)
        cache_key = scheduler.prefix_index.insert([1, 2, 3], 3, source.slot_id)
        scheduler.slot_manager.mark_cached(source.slot_id, cache_key, 3)

        seq = _seq([1, 2, 3, 4])
        scheduler.add(seq)
        seqs, is_prefill = scheduler.schedule()

        self.assertTrue(is_prefill)
        scheduled = seqs[0]
        prompt_slot = scheduled.prompt_cache_slot
        state_slot = scheduled.state_slot
        cache_hit_slot = scheduled.cache_hit_slot
        self.assertEqual(scheduler.slot_manager.slot_meta[cache_hit_slot].state.name, "CACHED_PINNED")

        scheduler.preempt(scheduled)

        self.assertEqual(scheduler.slot_manager.slot_meta[cache_hit_slot].state.name, "CACHED_EVICTABLE")
        self.assertEqual(scheduler.slot_manager.slot_meta[prompt_slot].state.name, "FREE")
        self.assertEqual(scheduler.slot_manager.slot_meta[state_slot].state.name, "FREE")
        self.assertIsNone(seq.state_slot)
        self.assertIsNone(seq.prompt_cache_slot)
        self.assertIsNone(seq.cache_hit_slot)
        self.assertEqual(seq.cached_prefix_len, 0)
        self.assertEqual(seq.num_cached_tokens, 0)
        self.assertFalse(seq.exact_cache_hit)
        self.assertFalse(seq.state_slot_materialized)


if __name__ == "__main__":
    unittest.main()
