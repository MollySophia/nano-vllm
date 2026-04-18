import unittest


try:
    from tests.openai_api_test_utils import (
        FakeTemplateTokenizer,
        patched_test_client,
    )
except ModuleNotFoundError as exc:  # pragma: no cover - depends on optional API deps.
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


class OpenAIAPIEndpointsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if IMPORT_ERROR is not None:
            raise unittest.SkipTest(f"OpenAI API test dependencies unavailable: {IMPORT_ERROR}")

    def test_health_endpoint_is_public(self):
        with patched_test_client(api_key="secret") as (client, llm, _factory):
            response = client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok", "model": "rwkv-test"})
        self.assertTrue(llm.exit_called)

    def test_models_endpoints_require_auth_and_return_metadata(self):
        with patched_test_client(api_key="secret") as (client, _llm, _factory):
            unauthorized = client.get("/v1/models")
            listed = client.get("/v1/models", headers={"Authorization": "Bearer secret"})
            retrieved = client.get("/v1/models/rwkv-test", headers={"Authorization": "Bearer secret"})

        self.assertEqual(unauthorized.status_code, 401)
        self.assertEqual(unauthorized.json()["error"]["code"], "invalid_api_key")
        self.assertEqual(listed.status_code, 200)
        self.assertEqual(listed.json()["object"], "list")
        self.assertEqual(listed.json()["data"][0]["id"], "rwkv-test")
        self.assertEqual(retrieved.status_code, 200)
        self.assertEqual(retrieved.json()["id"], "rwkv-test")
        self.assertEqual(retrieved.json()["owned_by"], "nano-vllm")

    def test_sync_completion_returns_expected_shape_usage_and_headers(self):
        with patched_test_client(completion_text="OK") as (client, llm, factory):
            response = client.post(
                "/v1/completions",
                json={
                    "model": "rwkv-test",
                    "prompt": "abc",
                    "max_tokens": 5,
                    "temperature": 0.25,
                    "top_p": 0.9,
                },
            )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["object"], "text_completion")
        self.assertEqual(body["model"], "rwkv-test")
        self.assertEqual(body["choices"][0]["text"], "OK")
        self.assertEqual(body["choices"][0]["finish_reason"], "stop")
        self.assertEqual(body["usage"], {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5})
        self.assertTrue(body["id"].startswith("cmpl-"))
        self.assertEqual(response.headers["x-nanovllm-streaming"], "false")
        self.assertEqual(response.headers["x-nanovllm-metrics-scope"], "final")
        self.assertEqual(response.headers["x-nanovllm-prompt-tokens"], "3")
        self.assertEqual(response.headers["x-nanovllm-completion-tokens"], "2")
        self.assertIn("x-nanovllm-output-tokens-per-second", response.headers)
        self.assertEqual(len(llm.received_requests), 1)
        request = llm.received_requests[0]
        self.assertEqual(request["prompt_text"], "abc")
        self.assertEqual(request["sampling_params"].temperature, 0.25)
        self.assertEqual(request["sampling_params"].top_p, 0.9)
        self.assertEqual(request["sampling_params"].max_tokens, 5)
        self.assertEqual(factory.calls[0][0], "/models/rwkv-test.pth")

    def test_sync_completion_maps_openai_penalties_to_internal_sampling_params(self):
        with patched_test_client(completion_text="OK") as (client, llm, _factory):
            response = client.post(
                "/v1/completions",
                json={
                    "model": "rwkv-test",
                    "prompt": "abc",
                    "max_tokens": 5,
                    "presence_penalty": 0.6,
                    "frequency_penalty": 0.2,
                },
            )

        self.assertEqual(response.status_code, 200)
        request = llm.received_requests[0]
        self.assertAlmostEqual(request["sampling_params"].presence_penalty, 0.6)
        self.assertAlmostEqual(request["sampling_params"].repetition_penalty, 0.2)
        self.assertAlmostEqual(request["sampling_params"].penalty_decay, 0.996)

    def test_sync_completion_accepts_penalty_decay_extension_field(self):
        with patched_test_client(completion_text="OK") as (client, llm, _factory):
            response = client.post(
                "/v1/completions",
                json={
                    "model": "rwkv-test",
                    "prompt": "abc",
                    "max_tokens": 5,
                    "penalty_decay": 0.9,
                },
            )

        self.assertEqual(response.status_code, 200)
        request = llm.received_requests[0]
        self.assertAlmostEqual(request["sampling_params"].penalty_decay, 0.9)

    def test_sync_chat_completion_uses_rwkv_chat_template_and_max_completion_tokens(self):
        with patched_test_client(chat_text="WXYZ") as (client, llm, _factory):
            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "rwkv-test",
                    "messages": [
                        {"role": "developer", "content": "Be terse."},
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "Hello"},
                                {"type": "text", "text": " world"},
                            ],
                        },
                    ],
                    "temperature": 0.1,
                    "max_completion_tokens": 2,
                },
            )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["object"], "chat.completion")
        self.assertEqual(body["choices"][0]["message"], {"role": "assistant", "content": "WX"})
        self.assertEqual(body["choices"][0]["finish_reason"], "length")
        self.assertEqual(body["usage"]["completion_tokens"], 2)
        self.assertEqual(
            llm.received_requests[0]["prompt_text"],
            "System: Be terse.\nUser: Hello world\nAssistant:",
        )
        self.assertEqual(llm.received_requests[0]["sampling_params"].max_tokens, 2)

    def test_sync_chat_completion_uses_tokenizer_chat_template_when_available(self):
        tokenizer = FakeTemplateTokenizer(template_text="<CHAT> templated prompt")
        with patched_test_client(tokenizer=tokenizer, chat_text="OK") as (client, llm, _factory):
            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "rwkv-test",
                    "messages": [{"role": "user", "content": "Hello"}],
                    "max_tokens": 4,
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["choices"][0]["message"]["content"], "OK")
        self.assertEqual(llm.received_requests[0]["prompt_text"], "<CHAT> templated prompt")
        self.assertEqual(
            tokenizer.calls,
            [([{"role": "user", "content": "Hello"}], False, True)],
        )

    def test_sync_completion_does_not_use_chat_template_when_available(self):
        tokenizer = FakeTemplateTokenizer(template_text="<CHAT> templated prompt")
        with patched_test_client(tokenizer=tokenizer, completion_text="OK") as (client, llm, _factory):
            response = client.post(
                "/v1/completions",
                json={
                    "model": "rwkv-test",
                    "prompt": "abc",
                    "max_tokens": 2,
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["choices"][0]["text"], "OK")
        self.assertEqual(llm.received_requests[0]["prompt_text"], "abc")
        self.assertEqual(tokenizer.calls, [])

    def test_sync_completion_returns_logprobs_for_generated_tokens(self):
        with patched_test_client(completion_text="OK") as (client, _llm, _factory):
            response = client.post(
                "/v1/completions",
                json={
                    "model": "rwkv-test",
                    "prompt": "abc",
                    "max_tokens": 2,
                    "logprobs": 2,
                },
            )

        self.assertEqual(response.status_code, 200)
        logprobs = response.json()["choices"][0]["logprobs"]
        self.assertEqual(logprobs["tokens"], ["O", "K"])
        self.assertEqual(len(logprobs["token_logprobs"]), 2)
        self.assertEqual(len(logprobs["top_logprobs"]), 2)
        self.assertIn("O", logprobs["top_logprobs"][0])
        self.assertEqual(logprobs["text_offset"], [0, 1])

    def test_sync_completion_accepts_prompt_token_ids(self):
        with patched_test_client(completion_text="OK") as (client, llm, _factory):
            response = client.post(
                "/v1/completions",
                json={
                    "model": "rwkv-test",
                    "prompt_token_ids": [97, 98, 99],
                    "max_tokens": 2,
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["choices"][0]["text"], "OK")
        self.assertEqual(llm.received_requests[0]["prompt_text"], "abc")

    def test_sync_completion_echo_with_logprobs_scores_prompt_tokens(self):
        with patched_test_client(completion_text="Z") as (client, _llm, _factory):
            response = client.post(
                "/v1/completions",
                json={
                    "model": "rwkv-test",
                    "prompt": "abc",
                    "max_tokens": 0,
                    "echo": True,
                    "logprobs": 2,
                },
            )

        self.assertEqual(response.status_code, 200)
        choice = response.json()["choices"][0]
        self.assertEqual(choice["text"], "abc")
        logprobs = choice["logprobs"]
        self.assertEqual(logprobs["tokens"], ["a", "b", "c"])
        self.assertEqual(logprobs["token_logprobs"][0], None)
        self.assertIsNotNone(logprobs["token_logprobs"][1])
        self.assertEqual(logprobs["text_offset"], [0, 1, 2])

    def test_tokenize_and_detokenize_endpoints_round_trip(self):
        with patched_test_client() as (client, _llm, _factory):
            tokenize_response = client.post(
                "/v1/tokenize",
                json={"model": "rwkv-test", "text": "abc"},
            )
            detokenize_response = client.post(
                "/v1/detokenize",
                json={"model": "rwkv-test", "token_ids": [97, 98, 99]},
            )

        self.assertEqual(tokenize_response.status_code, 200)
        self.assertEqual(
            tokenize_response.json(),
            {
                "token_ids": [97, 98, 99],
                "tokens": ["a", "b", "c"],
                "text_offset": [0, 1, 2],
                "count": 3,
            },
        )
        self.assertEqual(detokenize_response.status_code, 200)
        self.assertEqual(detokenize_response.json(), {"text": "abc", "count": 3})

    def test_create_app_passes_runtime_kwargs_to_llm(self):
        with patched_test_client(
            llm_kwargs={
                "tensor_parallel_size": 2,
                "max_num_seqs": 128,
                "rwkv_state_cache_enable": True,
            }
        ) as (_client, _llm, factory):
            pass

        self.assertEqual(
            factory.calls[0][1],
            {
                "tensor_parallel_size": 2,
                "max_num_seqs": 128,
                "rwkv_state_cache_enable": True,
            },
        )


if __name__ == "__main__":
    unittest.main()
