import unittest


try:
    from tests.openai_api_test_utils import patched_test_client
except ModuleNotFoundError as exc:  # pragma: no cover - depends on optional API deps.
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


class OpenAIAPIErrorsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if IMPORT_ERROR is not None:
            raise unittest.SkipTest(f"OpenAI API test dependencies unavailable: {IMPORT_ERROR}")

    def test_completion_rejects_unsupported_controls(self):
        cases = [
            ({"n": 2}, "n"),
            ({"stop": "\n"}, "stop"),
            ({"seed": 0}, "seed"),
            ({"top_p": 1.5}, "top_p"),
            ({"presence_penalty": 3.0}, "presence_penalty"),
            ({"frequency_penalty": 1.5}, "frequency_penalty"),
            ({"penalty_decay": 1.1}, "penalty_decay"),
            ({"prompt": ["a", "b"]}, "prompt"),
            ({"stream": True, "logprobs": 1}, "logprobs"),
            ({"stream": True, "echo": True}, "echo"),
            ({"max_tokens": 0}, "max_tokens"),
        ]
        with patched_test_client() as (client, _llm, _factory):
            for overrides, param in cases:
                with self.subTest(overrides=overrides):
                    response = client.post(
                        "/v1/completions",
                        json={
                            "model": "rwkv-test",
                            "prompt": "abc",
                            "max_tokens": 4,
                            **overrides,
                        },
                    )
                    self.assertEqual(response.status_code, 400)
                    error = response.json()["error"]
                    self.assertEqual(error["type"], "invalid_request_error")
                    self.assertEqual(error["param"], param)

    def test_completion_rejects_unknown_extra_field(self):
        with patched_test_client() as (client, _llm, _factory):
            response = client.post(
                "/v1/completions",
                json={
                    "model": "rwkv-test",
                    "prompt": "abc",
                    "max_tokens": 4,
                    "service_tier": "default",
                },
            )

        self.assertEqual(response.status_code, 400)
        self.assertIn("Unsupported request field(s): service_tier", response.json()["error"]["message"])

    def test_completion_requires_exactly_one_prompt_input(self):
        with patched_test_client() as (client, _llm, _factory):
            both = client.post(
                "/v1/completions",
                json={
                    "model": "rwkv-test",
                    "prompt": "abc",
                    "prompt_token_ids": [1, 2, 3],
                    "max_tokens": 4,
                },
            )
            missing = client.post(
                "/v1/completions",
                json={
                    "model": "rwkv-test",
                    "max_tokens": 4,
                },
            )

        self.assertEqual(both.status_code, 400)
        self.assertIn("prompt", both.json()["error"]["message"])
        self.assertEqual(missing.status_code, 400)
        self.assertIn("prompt", missing.json()["error"]["message"])

    def test_chat_rejects_unsupported_fields_and_invalid_messages(self):
        cases = [
            ({"messages": []}, "messages"),
            (
                {"messages": [{"role": "user", "content": [{"type": "image", "text": None}]}]},
                "messages",
            ),
            ({"tools": [{"type": "function"}]}, "tools"),
            ({"tool_choice": "auto"}, "tool_choice"),
            ({"parallel_tool_calls": True}, "parallel_tool_calls"),
            ({"response_format": {"type": "json_object"}}, "response_format"),
            ({"top_logprobs": 1}, "top_logprobs"),
            ({"max_tokens": 2, "max_completion_tokens": 3}, "max_completion_tokens"),
        ]
        with patched_test_client() as (client, _llm, _factory):
            for overrides, param in cases:
                with self.subTest(overrides=overrides):
                    payload = {
                        "model": "rwkv-test",
                        "messages": [{"role": "user", "content": "Hello"}],
                        "max_tokens": 4,
                    }
                    payload.update(overrides)
                    response = client.post("/v1/chat/completions", json=payload)
                    self.assertEqual(response.status_code, 400)
                    error = response.json()["error"]
                    self.assertEqual(error["type"], "invalid_request_error")
                    self.assertEqual(error["param"], param)

    def test_chat_rejects_unknown_extra_field(self):
        with patched_test_client() as (client, _llm, _factory):
            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "rwkv-test",
                    "messages": [{"role": "user", "content": "Hello"}],
                    "service_tier": "default",
                },
            )

        self.assertEqual(response.status_code, 400)
        self.assertIn("Unsupported request field(s): service_tier", response.json()["error"]["message"])

    def test_chat_rejects_invalid_stream_options(self):
        with patched_test_client() as (client, _llm, _factory):
            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "rwkv-test",
                    "messages": [{"role": "user", "content": "Hello"}],
                    "stream": True,
                    "stream_options": {"include_usage": "yes"},
                },
            )

        self.assertEqual(response.status_code, 400)
        error = response.json()["error"]
        self.assertEqual(error["type"], "invalid_request_error")
        self.assertEqual(error["param"], "stream_options")
        self.assertIn("stream_options.include_usage", error["message"])

    def test_tokenize_and_detokenize_validate_payloads(self):
        with patched_test_client() as (client, _llm, _factory):
            tokenize_response = client.post(
                "/v1/tokenize",
                json={"model": "rwkv-test", "token_ids": [1, 2]},
            )
            detokenize_response = client.post(
                "/v1/detokenize",
                json={"model": "rwkv-test", "token_ids": [1, "x"]},
            )

        self.assertEqual(tokenize_response.status_code, 400)
        self.assertIn("token_ids", tokenize_response.json()["error"]["message"])
        self.assertEqual(detokenize_response.status_code, 400)
        self.assertIn("token_ids", detokenize_response.json()["error"]["message"])

    def test_model_mismatch_and_authentication_errors_use_openai_shape(self):
        with patched_test_client(api_key="secret") as (client, _llm, _factory):
            missing_auth = client.post(
                "/v1/completions",
                json={"model": "rwkv-test", "prompt": "abc", "max_tokens": 4},
            )
            wrong_model = client.post(
                "/v1/completions",
                headers={"Authorization": "Bearer secret"},
                json={"model": "other-model", "prompt": "abc", "max_tokens": 4},
            )

        self.assertEqual(missing_auth.status_code, 401)
        self.assertEqual(missing_auth.json()["error"]["code"], "invalid_api_key")
        self.assertEqual(wrong_model.status_code, 404)
        self.assertEqual(wrong_model.json()["error"]["param"], "model")
        self.assertEqual(wrong_model.json()["error"]["code"], "model_not_found")

    def test_validation_errors_are_translated_to_openai_error_shape(self):
        with patched_test_client() as (client, _llm, _factory):
            response = client.post("/v1/completions", json={"model": "rwkv-test"})

        self.assertEqual(response.status_code, 400)
        error = response.json()["error"]
        self.assertEqual(error["type"], "invalid_request_error")
        self.assertIsNone(error["param"])
        self.assertIn("Invalid request body", error["message"])


if __name__ == "__main__":
    unittest.main()
