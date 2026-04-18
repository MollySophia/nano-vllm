import unittest
from unittest import mock

import torch

from nanovllm.models import rwkv7_ops


class RWKV7OpsTest(unittest.TestCase):
    def test_one_batch_out_by_slot_runs_uses_state_slices_for_contiguous_input_runs(self):
        state_cache = torch.arange(8 * 2 * 2, dtype=torch.float32).view(8, 2, 2)
        slot_mapping_in = torch.tensor([5, 6, 1, 2], dtype=torch.int64)
        slot_mapping_out = torch.tensor([1, 2, 5, 6], dtype=torch.int64)
        payload = torch.zeros(4, 2, 2, dtype=torch.float32)
        seen = []

        def fake_kernel(state_in, state_out, r, w, k, v, kk, kka, positions):
            seen.append((state_in, state_out, r.shape[0]))
            return torch.zeros_like(r)

        with mock.patch.object(rwkv7_ops, "wkv7_one_batch_cuda", side_effect=fake_kernel):
            out = rwkv7_ops._wkv7_one_batch_out_by_slot_runs(
                state_cache,
                slot_mapping_in,
                slot_mapping_out,
                payload,
                payload,
                payload,
                payload,
                payload,
                payload,
                torch.zeros(4, dtype=torch.int64),
            )

        self.assertEqual(out.shape, payload.shape)
        self.assertEqual(len(seen), 2)
        self.assertEqual([batch for _, _, batch in seen], [2, 2])
        cache_ptr = state_cache.untyped_storage().data_ptr()
        self.assertTrue(all(state_in.untyped_storage().data_ptr() == cache_ptr for state_in, _, _ in seen))
        self.assertTrue(all(state_out.untyped_storage().data_ptr() == cache_ptr for _, state_out, _ in seen))

    def test_one_batch_out_by_slot_runs_chunks_noncontiguous_input_runs(self):
        state_cache = torch.arange(16 * 2 * 2, dtype=torch.float32).view(16, 2, 2)
        slot_mapping_in = torch.tensor([10, 12, 14, 3, 5], dtype=torch.int64)
        slot_mapping_out = torch.tensor([1, 2, 3, 4, 5], dtype=torch.int64)
        payload = torch.zeros(5, 2, 2, dtype=torch.float32)
        seen_batches = []

        def fake_kernel(state_in, state_out, r, w, k, v, kk, kka, positions):
            seen_batches.append((state_in.shape[0], state_in.untyped_storage().data_ptr()))
            return torch.zeros_like(r)

        with (
            mock.patch.object(rwkv7_ops, "_MAX_NONCONTIGUOUS_STATE_GATHER_ROWS", 2),
            mock.patch.object(rwkv7_ops, "wkv7_one_batch_cuda", side_effect=fake_kernel),
        ):
            out = rwkv7_ops._wkv7_one_batch_out_by_slot_runs(
                state_cache,
                slot_mapping_in,
                slot_mapping_out,
                payload,
                payload,
                payload,
                payload,
                payload,
                payload,
                torch.zeros(5, dtype=torch.int64),
            )

        self.assertEqual(out.shape, payload.shape)
        self.assertEqual([batch for batch, _ in seen_batches], [2, 2, 1])
        cache_ptr = state_cache.untyped_storage().data_ptr()
        self.assertNotEqual(seen_batches[0][1], cache_ptr)
        self.assertNotEqual(seen_batches[1][1], cache_ptr)


if __name__ == "__main__":
    unittest.main()
