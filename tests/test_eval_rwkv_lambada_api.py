import unittest

from eval_rwkv_lambada_api import (
    RWKV_EOD_TEXT,
    RawSample,
    TokenizedSample,
    load_lambada_texts,
    score_sample_from_completion,
)


class EvalRWKVLambadaAPITest(unittest.TestCase):
    def test_load_lambada_texts_prepends_eod_text(self):
        samples = load_lambada_texts(
            "nanovllm/eval_data/lambada_test.jsonl",
            limit=1,
            pad_eod=True,
        )

        self.assertEqual(len(samples), 1)
        self.assertTrue(samples[0].prefix_text.startswith(RWKV_EOD_TEXT))
        self.assertTrue(samples[0].target_text.startswith(" "))

    def test_score_sample_from_completion_uses_target_suffix(self):
        sample = TokenizedSample(
            prefix_text="prefix",
            target_text=" target",
            prefix_token_ids=[1, 2],
            target_token_ids=[3, 4],
            target_tokens=[" ta", "rget"],
        )
        response_body = {
            "choices": [
                {
                    "logprobs": {
                        "token_logprobs": [None, -0.1, -0.2, -0.3],
                        "top_logprobs": [
                            None,
                            {"x": -0.1},
                            {" ta": -0.2},
                            {"rget": -0.3},
                        ],
                    }
                }
            ]
        }

        logprob_sum, correct = score_sample_from_completion(sample, response_body)

        self.assertAlmostEqual(logprob_sum, -0.5)
        self.assertTrue(correct)

    def test_score_sample_from_completion_marks_top1_mismatch_incorrect(self):
        sample = TokenizedSample(
            prefix_text="prefix",
            target_text=" target",
            prefix_token_ids=[1],
            target_token_ids=[2],
            target_tokens=[" target"],
        )
        response_body = {
            "choices": [
                {
                    "logprobs": {
                        "token_logprobs": [None, -1.2],
                        "top_logprobs": [
                            None,
                            {" other": -0.1},
                        ],
                    }
                }
            ]
        }

        logprob_sum, correct = score_sample_from_completion(sample, response_body)

        self.assertAlmostEqual(logprob_sum, -1.2)
        self.assertFalse(correct)


if __name__ == "__main__":
    unittest.main()
