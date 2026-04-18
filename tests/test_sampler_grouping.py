import unittest
from unittest import mock

import torch

from nanovllm.engine.sequence import Sequence
from nanovllm.layers import sampler as sampler_module
from nanovllm.layers.sampler import Sampler
from nanovllm.sampling_params import SamplingParams


def _seq(*, temperature: float, top_k: int = -1, top_p: float = 1.0) -> Sequence:
    return Sequence(
        [1],
        SamplingParams(
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            max_tokens=1,
        ),
    )


class SamplerGroupingTest(unittest.TestCase):
    def test_group_indices_by_sampling_exact(self):
        sampler = Sampler()
        groups = sampler._group_indices_by_sampling(
            [
                _seq(temperature=0.71, top_p=0.91),
                _seq(temperature=0.71, top_p=0.91),
                _seq(temperature=0.76, top_p=0.91),
                _seq(temperature=1e-4, top_p=0.95),
            ]
        )

        self.assertEqual(
            groups,
            [
                (("sample", 0.71, -1, 0.91, 0.0, 0.0, 1.0), [0, 1]),
                (("sample", 0.76, -1, 0.91, 0.0, 0.0, 1.0), [2]),
                (("greedy", 0.0, 1, 0.0, 0.0, 0.0, 1.0), [3]),
            ],
        )

    def test_group_indices_by_sampling_bucketed(self):
        sampler = Sampler(
            temperature_bucket_resolution=0.05,
            top_p_bucket_resolution=0.05,
        )
        groups = sampler._group_indices_by_sampling(
            [
                _seq(temperature=0.71, top_p=0.92),
                _seq(temperature=0.72, top_p=0.89),
                _seq(temperature=0.79, top_p=0.96),
                _seq(temperature=1e-4, top_p=0.95),
            ]
        )

        self.assertEqual(
            groups,
            [
                (("sample", 0.7, -1, 0.9, 0.0, 0.0, 1.0), [0, 1]),
                (("sample", 0.8, -1, 0.95, 0.0, 0.0, 1.0), [2]),
                (("greedy", 0.0, 1, 0.0, 0.0, 0.0, 1.0), [3]),
            ],
        )

    def test_forward_groups_by_bucket_before_sampling(self):
        sampler = Sampler(
            temperature_bucket_resolution=0.05,
            top_p_bucket_resolution=0.05,
        )
        logits = torch.tensor(
            [
                [0.1, 0.9, 0.2, 0.3],
                [0.4, 0.2, 0.1, 0.8],
                [0.7, 0.1, 0.2, 0.3],
                [0.2, 0.1, 1.4, 0.3],
            ],
            dtype=torch.float32,
        )
        seqs = [
            _seq(temperature=0.71, top_p=0.92, top_k=40),
            _seq(temperature=0.72, top_p=0.89, top_k=40),
            _seq(temperature=0.79, top_p=0.96, top_k=20),
            _seq(temperature=1e-4, top_p=0.95),
        ]

        calls = []

        def fake_rapid(logits, states, temperature, top_k, top_p):
            calls.append((tuple(logits.shape), temperature, top_k, top_p))
            value = 3 if logits.shape[0] == 2 else 1
            return torch.full((logits.shape[0],), value, dtype=torch.int32, device=logits.device)

        with (
            mock.patch.object(sampler, "_can_use_rapid_sampling", side_effect=lambda group_logits, config: True),
            mock.patch.object(
                sampler,
                "_ensure_rand_states",
                side_effect=lambda batch_size, device: torch.zeros(batch_size, dtype=torch.uint8, device=device),
            ),
            mock.patch.object(
                sampler_module.rapid_sampling,
                "batch_sampling_temperature_topk_topp",
                side_effect=fake_rapid,
            ),
        ):
            out = sampler(logits, seqs)

        self.assertEqual(
            calls,
            [
                ((2, 4), 0.7, 40, 0.9),
                ((1, 4), 0.8, 20, 0.95),
            ],
        )
        self.assertEqual(out.tolist(), [3, 3, 1, 2])

    def test_forward_uses_whole_batch_fast_path_for_uniform_greedy(self):
        sampler = Sampler()
        logits = torch.tensor(
            [
                [0.1, 0.9, 0.2, 0.3],
                [0.4, 0.2, 0.1, 0.8],
                [0.2, 1.1, 0.4, 0.3],
            ],
            dtype=torch.float32,
        )
        seqs = [
            _seq(temperature=0.0),
            _seq(temperature=0.0),
            _seq(temperature=0.0),
        ]

        with mock.patch.object(sampler, "_group_indices_by_sampling", side_effect=AssertionError("grouping should be skipped")):
            out = sampler(logits, seqs)

        self.assertEqual(out.tolist(), [1, 3, 1])

    def test_forward_uses_whole_batch_fast_path_for_uniform_sampling(self):
        sampler = Sampler()
        logits = torch.tensor(
            [
                [0.1, 0.9, 0.2, 0.3],
                [0.4, 0.2, 0.1, 0.8],
            ],
            dtype=torch.float32,
        )
        seqs = [
            _seq(temperature=0.7, top_k=40, top_p=0.9),
            _seq(temperature=0.7, top_k=40, top_p=0.9),
        ]

        with (
            mock.patch.object(sampler, "_group_indices_by_sampling", side_effect=AssertionError("grouping should be skipped")),
            mock.patch.object(
                sampler,
                "_sample_without_penalties",
                return_value=torch.tensor([3, 1], dtype=torch.int64),
            ) as sample_mock,
        ):
            out = sampler(logits, seqs)

        sample_mock.assert_called_once()
        self.assertEqual(out.tolist(), [3, 1])


if __name__ == "__main__":
    unittest.main()
