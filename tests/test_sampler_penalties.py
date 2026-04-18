import unittest

import torch

from nanovllm.engine.sequence import Sequence
from nanovllm.layers.sampler import Sampler
from nanovllm.sampling_params import SamplingParams


class SamplerPenaltyTests(unittest.TestCase):

    @staticmethod
    def _make_seq() -> Sequence:
        seq = Sequence(
            [0],
            SamplingParams(
                temperature=0.0,
                top_k=1,
                top_p=0.0,
                presence_penalty=10.0,
                repetition_penalty=4.0,
                penalty_decay=0.5,
                max_tokens=2,
            ),
        )
        seq.state_slot = 0
        return seq

    def test_cpu_fallback_updates_slot_penalties_and_affects_next_step(self):
        sampler = Sampler()
        seq = self._make_seq()
        logits = torch.tensor([[5.0, 1.0, 0.0, -1.0]], dtype=torch.float32)
        slot_penalties = torch.zeros(1, 4, dtype=torch.float32)

        first = sampler(logits, [seq], slot_penalties=slot_penalties, slot_ids=[0])
        self.assertEqual(first.tolist(), [0])
        self.assertTrue(torch.equal(slot_penalties[0], torch.tensor([1.0, 0.0, 0.0, 0.0])))

        second = sampler(logits, [seq], slot_penalties=slot_penalties, slot_ids=[0])
        self.assertEqual(second.tolist(), [1])
        self.assertTrue(torch.equal(slot_penalties[0], torch.tensor([0.5, 1.0, 0.0, 0.0])))

    def test_sparse_occurrence_state_matches_reference_formula(self):
        sampler = Sampler()
        seq = self._make_seq()
        seq.allow_sparse_penalty_state = True
        logits = torch.tensor([[5.0, 1.0, 0.0, -1.0]], dtype=torch.float32)

        first = sampler(logits, [seq])
        self.assertEqual(first.tolist(), [0])
        self.assertEqual(seq.penalty_state, {0: 1.0})

        second = sampler(logits, [seq])
        self.assertEqual(second.tolist(), [1])
        self.assertEqual(seq.penalty_state, {0: 0.5, 1: 1.0})

    def test_penalties_require_slot_state(self):
        sampler = Sampler()
        seq = self._make_seq()
        logits = torch.tensor([[1.0, 0.0, -1.0, -2.0]], dtype=torch.float32)
        with self.assertRaises(ValueError):
            sampler(logits, [seq])


if __name__ == "__main__":
    unittest.main()
