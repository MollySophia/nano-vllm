import unittest

from nanovllm.engine.model_runner import _bs1_requires_sequence_sampler, _resolve_state_slot_layout


class ModelRunnerSlotLayoutTest(unittest.TestCase):
    def test_non_cache_mode_caps_slots_to_requested_max_num_seqs_plus_graph_slot(self):
        total_slots, num_state_blocks, bs1_graph_slot, effective_max_num_seqs = _resolve_state_slot_layout(
            total_slots_capacity=100,
            requested_max_num_seqs=8,
            rwkv_state_cache_enable=False,
            world_size=1,
            enforce_eager=False,
        )
        self.assertEqual(total_slots, 9)
        self.assertEqual(num_state_blocks, 8)
        self.assertEqual(bs1_graph_slot, 8)
        self.assertEqual(effective_max_num_seqs, 8)

    def test_non_cache_mode_eager_uses_exact_requested_slot_count(self):
        total_slots, num_state_blocks, bs1_graph_slot, effective_max_num_seqs = _resolve_state_slot_layout(
            total_slots_capacity=100,
            requested_max_num_seqs=8,
            rwkv_state_cache_enable=False,
            world_size=1,
            enforce_eager=True,
        )
        self.assertEqual(total_slots, 8)
        self.assertEqual(num_state_blocks, 8)
        self.assertEqual(bs1_graph_slot, -1)
        self.assertEqual(effective_max_num_seqs, 8)

    def test_cache_mode_keeps_extra_slots_beyond_active_seq_limit(self):
        total_slots, num_state_blocks, bs1_graph_slot, effective_max_num_seqs = _resolve_state_slot_layout(
            total_slots_capacity=100,
            requested_max_num_seqs=8,
            rwkv_state_cache_enable=True,
            world_size=1,
            enforce_eager=False,
        )
        self.assertEqual(total_slots, 100)
        self.assertEqual(num_state_blocks, 99)
        self.assertEqual(bs1_graph_slot, 99)
        self.assertEqual(effective_max_num_seqs, 8)

    def test_auto_max_num_seqs_stays_as_large_as_available(self):
        total_slots, num_state_blocks, bs1_graph_slot, effective_max_num_seqs = _resolve_state_slot_layout(
            total_slots_capacity=100,
            requested_max_num_seqs=-1,
            rwkv_state_cache_enable=False,
            world_size=1,
            enforce_eager=False,
        )
        self.assertEqual(total_slots, 100)
        self.assertEqual(num_state_blocks, 99)
        self.assertEqual(bs1_graph_slot, 99)
        self.assertEqual(effective_max_num_seqs, 99)

    def test_bs1_penalties_force_sequence_sampler_even_at_zero_temperature(self):
        self.assertFalse(
            _bs1_requires_sequence_sampler(
                temperature=0.0,
                presence_penalty=0.0,
                repetition_penalty=0.0,
            )
        )
        self.assertTrue(
            _bs1_requires_sequence_sampler(
                temperature=0.0,
                presence_penalty=0.5,
                repetition_penalty=0.0,
            )
        )
        self.assertTrue(
            _bs1_requires_sequence_sampler(
                temperature=0.0,
                presence_penalty=0.0,
                repetition_penalty=0.25,
            )
        )


if __name__ == "__main__":
    unittest.main()
