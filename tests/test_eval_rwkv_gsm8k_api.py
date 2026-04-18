import unittest

from eval_rwkv_gsm8k_api import (
    Sample,
    apply_single_stage_response,
    apply_two_stage_responses,
    completion_text_and_usage,
)


class EvalRWKVGSM8KAPITest(unittest.TestCase):
    def test_completion_text_and_usage_reads_openai_shape(self):
        text, prompt_tokens, completion_tokens = completion_text_and_usage(
            {
                "choices": [{"text": "#### 42"}],
                "usage": {"prompt_tokens": 11, "completion_tokens": 3},
            }
        )

        self.assertEqual(text, "#### 42")
        self.assertEqual(prompt_tokens, 11)
        self.assertEqual(completion_tokens, 3)

    def test_apply_single_stage_response_extracts_answer_and_usage(self):
        sample = Sample(
            problem="2+2",
            gold_answer="4",
            prompt="prompt",
        )

        prompt_tokens, completion_tokens = apply_single_stage_response(
            sample,
            {
                "choices": [{"text": "Reasoning\n#### 4"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 4},
            },
        )

        self.assertEqual(prompt_tokens, 10)
        self.assertEqual(completion_tokens, 4)
        self.assertEqual(sample.extracted_answer, "4")
        self.assertTrue(sample.is_correct)
        self.assertEqual(sample.prompt_tokens, 10)
        self.assertEqual(sample.output_tokens, 4)

    def test_apply_two_stage_responses_renders_full_prediction(self):
        sample = Sample(
            problem="2+2",
            gold_answer="4",
            prompt="prompt",
            expected_context=(
                "User: q\n\nAssistant: <think><|completions_of_cot|></think>\n"
                "Therefore, the answer is \\(\\boxed{<|final_answer|>}\\)."
            ),
        )

        prompt_tokens, completion_tokens = apply_two_stage_responses(
            sample,
            {
                "choices": [{"text": "Need one step</think> ignored"}],
                "usage": {"prompt_tokens": 8, "completion_tokens": 5},
            },
            {
                "choices": [{"text": "4}\\\\ trailing"}],
                "usage": {"prompt_tokens": 12, "completion_tokens": 2},
            },
        )

        self.assertEqual(prompt_tokens, 20)
        self.assertEqual(completion_tokens, 7)
        self.assertIn("Need one step", sample.prediction_text)
        self.assertEqual(sample.extracted_answer, "4")
        self.assertTrue(sample.is_correct)


if __name__ == "__main__":
    unittest.main()
