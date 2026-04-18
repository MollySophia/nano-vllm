import unittest

from nanovllm.tokenizers import RWKVTokenizer


class RWKVTokenizerTest(unittest.TestCase):
    def test_encode_supports_rwkv_end_of_text_alias(self):
        tokenizer = RWKVTokenizer()

        self.assertEqual(tokenizer.encode("<|rwkv_end_of_text|>"), [0])

    def test_encode_keeps_legacy_endoftext_alias(self):
        tokenizer = RWKVTokenizer()

        self.assertEqual(tokenizer.encode("<|endoftext|>"), [0])

    def test_decode_uses_rwkv_end_of_text_text(self):
        tokenizer = RWKVTokenizer()

        self.assertEqual(tokenizer.decode([0]), "<|rwkv_end_of_text|>")

    def test_special_token_round_trip_stays_on_token_zero(self):
        tokenizer = RWKVTokenizer()

        self.assertEqual(
            tokenizer.encode(tokenizer.decode([0])),
            [0],
        )

    def test_apply_chat_template_matches_rwkv_mobile_defaults(self):
        tokenizer = RWKVTokenizer()

        rendered = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": "Be terse."},
                {"role": "user", "content": "Hello"},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )

        self.assertEqual(rendered, "System: Be terse.\n\nUser: Hello\n\nAssistant:")

    def test_apply_chat_template_collapses_user_and_system_blank_lines(self):
        tokenizer = RWKVTokenizer()

        rendered = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": "line1\r\n\r\nline2"},
                {"role": "user", "content": "hello\n\n\nworld"},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )

        self.assertEqual(rendered, "System: line1\nline2\n\nUser: hello\nworld\n\nAssistant:")

    def test_apply_chat_template_after_assistant_turn_switches_back_to_user(self):
        tokenizer = RWKVTokenizer()

        rendered = tokenizer.apply_chat_template(
            [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi"},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )

        self.assertEqual(rendered, "User: Hello\n\nAssistant: Hi\n\nUser:")

    def test_apply_chat_template_can_return_token_ids(self):
        tokenizer = RWKVTokenizer()

        token_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": "Hello"}],
            tokenize=True,
            add_generation_prompt=True,
        )

        self.assertEqual(token_ids, tokenizer.encode("User: Hello\n\nAssistant:"))

    def test_state_cache_canonicalization_matches_rwkv_mobile_behavior(self):
        canonical_ids = RWKVTokenizer.canonicalize_state_cache_token_ids(
            [7, 10080, 261, 8, 9830, 261, 9, 19137, 261]
        )

        self.assertEqual(canonical_ids, [7, 28329, 11, 8, 28324, 11, 9, 28331, 11])

    def test_default_stop_token_sequences_match_rwkv_mobile_defaults(self):
        tokenizer = RWKVTokenizer()

        self.assertEqual(
            tokenizer.get_default_stop_token_seqs(),
            (
                (261,),
                (28329, 11),
                (28324, 11),
                (28331, 11),
                (5585,),
            ),
        )


if __name__ == "__main__":
    unittest.main()
