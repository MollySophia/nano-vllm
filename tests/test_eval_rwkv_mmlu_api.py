import unittest

from eval_rwkv_mmlu_api import (
    CHOICES,
    Sample,
    apply_choice_scores,
    build_scoring_prompt,
    extract_choice_logprob_and_prompt_tokens,
)


class EvalRWKVMMLUAPITest(unittest.TestCase):
    def test_build_scoring_prompt_prepends_eod_text(self):
        prompt = build_scoring_prompt("Question?", " A")

        self.assertTrue(prompt.startswith("<|rwkv_end_of_text|>"))
        self.assertTrue(prompt.endswith(" A"))

    def test_extract_choice_logprob_and_prompt_tokens_reads_last_token(self):
        logprob, prompt_tokens = extract_choice_logprob_and_prompt_tokens(
            {
                "choices": [
                    {
                        "logprobs": {
                            "token_logprobs": [None, -0.25],
                        }
                    }
                ],
                "usage": {"prompt_tokens": 123},
            }
        )

        self.assertAlmostEqual(logprob, -0.25)
        self.assertEqual(prompt_tokens, 123)

    def test_apply_choice_scores_picks_highest_logprob(self):
        sample = Sample(
            question="q",
            choices=["a", "b", "c", "d"],
            subject="math",
            answer=2,
            prompt="prompt",
        )

        apply_choice_scores(
            sample,
            choice_scores=[-3.0, -1.0, -0.5, -2.0],
            logical_prompt_tokens=77,
        )

        self.assertEqual(len(CHOICES), 4)
        self.assertEqual(sample.prompt_tokens, 77)
        self.assertEqual(sample.predicted, 2)
        self.assertTrue(sample.is_correct)


if __name__ == "__main__":
    unittest.main()
