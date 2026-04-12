import unittest

from nanovllm.utils.rwkv_int8 import (
    describe_rwkv_int8_mode,
    normalize_rwkv_int8_lm_head_flags,
    resolve_rwkv_int8_lm_head_flags,
)


class RWKVInt8UtilsTest(unittest.TestCase):
    def test_resolve_defaults_int8_to_marlin_lm_head(self):
        self.assertEqual(
            resolve_rwkv_int8_lm_head_flags(rwkv_quant_int8=True),
            (True, True),
        )

    def test_resolve_fp16_lm_head_override_disables_int8_lm_head(self):
        self.assertEqual(
            resolve_rwkv_int8_lm_head_flags(
                rwkv_quant_int8=True,
                rwkv_int8_fp16_lm_head=True,
            ),
            (False, False),
        )

    def test_resolve_rejects_fp16_lm_head_without_int8(self):
        with self.assertRaises(ValueError):
            resolve_rwkv_int8_lm_head_flags(
                rwkv_quant_int8=False,
                rwkv_int8_fp16_lm_head=True,
            )

    def test_normalize_matches_resolve_semantics(self):
        self.assertEqual(
            normalize_rwkv_int8_lm_head_flags(rwkv_quant_int8=True),
            (True, True),
        )
        self.assertEqual(
            normalize_rwkv_int8_lm_head_flags(
                rwkv_quant_int8=True,
                rwkv_int8_fp16_lm_head=True,
            ),
            (False, False),
        )

    def test_describe_mode_labels(self):
        self.assertEqual(
            describe_rwkv_int8_mode(
                rwkv_quant_int8=False,
                rwkv_quant_int8_lm_head=False,
                rwkv_quant_int8_lm_head_marlin=False,
            ),
            "fp16",
        )
        self.assertEqual(
            describe_rwkv_int8_mode(
                rwkv_quant_int8=True,
                rwkv_quant_int8_lm_head=False,
                rwkv_quant_int8_lm_head_marlin=False,
            ),
            "int8_fp16_lm_head",
        )
        self.assertEqual(
            describe_rwkv_int8_mode(
                rwkv_quant_int8=True,
                rwkv_quant_int8_lm_head=True,
                rwkv_quant_int8_lm_head_marlin=True,
            ),
            "int8_marlin_lm_head",
        )


if __name__ == "__main__":
    unittest.main()
