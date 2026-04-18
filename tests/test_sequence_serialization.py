import pickle
import unittest

from nanovllm.engine.sequence import Sequence
from nanovllm.sampling_params import SamplingParams


class SequenceSerializationTest(unittest.TestCase):
    def test_prompt_only_roundtrip_preserves_token_ids_and_sampling_fields(self):
        seq = Sequence(
            [11, 12, 13],
            SamplingParams(
                temperature=0.7,
                top_k=40,
                top_p=0.9,
                presence_penalty=0.1,
                repetition_penalty=0.2,
                penalty_decay=0.95,
                max_tokens=16,
                ignore_eos=True,
            ),
        )
        seq.state_slot = 3
        seq.prompt_cache_slot = 4
        seq.cache_hit_slot = 5
        seq.cached_prefix_len = 2
        seq.exact_cache_hit = True
        seq.final_cache_published = True
        seq.state_slot_materialized = True
        seq.active_state_slot = 6
        seq.pending_hidden_finalize = True
        seq.penalty_state = {42: 1.25}
        seq.allow_sparse_penalty_state = True

        clone = pickle.loads(pickle.dumps(seq))

        self.assertEqual(clone.token_ids, [11, 12, 13])
        self.assertEqual(clone.last_token, 13)
        self.assertEqual(clone.num_tokens, 3)
        self.assertEqual(clone.num_prompt_tokens, 3)
        self.assertEqual(clone.temperature, 0.7)
        self.assertEqual(clone.top_k, 40)
        self.assertEqual(clone.top_p, 0.9)
        self.assertEqual(clone.presence_penalty, 0.1)
        self.assertEqual(clone.repetition_penalty, 0.2)
        self.assertEqual(clone.penalty_decay, 0.95)
        self.assertEqual(clone.max_tokens, 16)
        self.assertTrue(clone.ignore_eos)
        self.assertEqual(clone.state_slot, 3)
        self.assertEqual(clone.prompt_cache_slot, 4)
        self.assertEqual(clone.cache_hit_slot, 5)
        self.assertEqual(clone.cached_prefix_len, 2)
        self.assertTrue(clone.exact_cache_hit)
        self.assertTrue(clone.final_cache_published)
        self.assertTrue(clone.state_slot_materialized)
        self.assertEqual(clone.active_state_slot, 6)
        self.assertTrue(clone.pending_hidden_finalize)
        self.assertEqual(clone.penalty_state, {42: 1.25})
        self.assertTrue(clone.allow_sparse_penalty_state)

    def test_completion_roundtrip_preserves_last_token_and_runtime_state(self):
        seq = Sequence(
            [21, 22],
            SamplingParams(
                temperature=1.0,
                top_k=-1,
                top_p=0.8,
                presence_penalty=0.0,
                repetition_penalty=0.0,
                penalty_decay=1.0,
                max_tokens=8,
                ignore_eos=False,
            ),
        )
        seq.append_token(23)
        seq.state_slot = 7
        seq.cached_prefix_len = 1
        seq.active_state_slot = 9
        seq.penalty_state = {23: 0.5}

        clone = pickle.loads(pickle.dumps(seq))

        self.assertEqual(clone.num_tokens, 3)
        self.assertEqual(clone.num_prompt_tokens, 2)
        self.assertEqual(clone.num_completion_tokens, 1)
        self.assertEqual(clone.last_token, 23)
        self.assertEqual(clone.temperature, 1.0)
        self.assertEqual(clone.top_p, 0.8)
        self.assertEqual(clone.max_tokens, 8)
        self.assertFalse(clone.ignore_eos)
        self.assertEqual(clone.state_slot, 7)
        self.assertEqual(clone.cached_prefix_len, 1)
        self.assertEqual(clone.active_state_slot, 9)
        self.assertEqual(clone.penalty_state, {23: 0.5})

    def test_hidden_stop_token_roundtrip_preserves_visible_completion_view(self):
        seq = Sequence(
            [31, 32],
            SamplingParams(
                temperature=0.0,
                max_tokens=8,
            ),
        )
        seq.append_token(261)
        seq.hidden_completion_token_count = 1
        seq.last_token_hidden_from_output = True

        clone = pickle.loads(pickle.dumps(seq))

        self.assertEqual(clone.token_ids, [31, 32, 261])
        self.assertEqual(clone.num_raw_completion_tokens, 1)
        self.assertEqual(clone.num_completion_tokens, 0)
        self.assertEqual(clone.raw_completion_token_ids, [261])
        self.assertEqual(clone.completion_token_ids, [])
        self.assertEqual(clone.hidden_completion_token_count, 1)
        self.assertTrue(clone.last_token_hidden_from_output)


if __name__ == "__main__":
    unittest.main()
