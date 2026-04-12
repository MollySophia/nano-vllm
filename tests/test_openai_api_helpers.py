import threading
import unittest


try:
    from nanovllm.entrypoints.openai.api_server import (
        ChatMessage,
        ChatCompletionRequest,
        CompletionRequest,
        OpenAIAPIError,
        ServerState,
        TextPart,
        _chat_stream_finish_reason,
        _coerce_text_content,
        _default_model_name,
        _normalize_prompt,
        _render_chat_prompt,
        _require_api_key,
        _response_headers,
        _sampling_params_from_chat,
        _sampling_params_from_completion,
        _validated_sampling_params,
    )
except ModuleNotFoundError as exc:  # pragma: no cover - depends on optional API deps.
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


class _FallbackTokenizer:
    pass


class _TemplateTokenizer:
    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        self.calls.append((messages, tokenize, add_generation_prompt))
        return "templated"


class OpenAIAPIHelpersTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if IMPORT_ERROR is not None:
            raise unittest.SkipTest(f"OpenAI API test dependencies unavailable: {IMPORT_ERROR}")

    def test_validated_sampling_params_preserves_top_p_and_defaults(self):
        sampling = _validated_sampling_params(
            temperature=None,
            top_p=0.75,
            max_tokens=None,
        )
        self.assertEqual(sampling.temperature, 1.0)
        self.assertEqual(sampling.top_p, 0.75)
        self.assertEqual(sampling.max_tokens, 16)

    def test_sampling_params_from_chat_prefers_max_completion_tokens(self):
        req = ChatCompletionRequest(
            model="rwkv-test",
            messages=[{"role": "user", "content": "hi"}],
            temperature=0.6,
            top_p=0.8,
            max_completion_tokens=32,
        )
        sampling = _sampling_params_from_chat(req)
        self.assertEqual(sampling.temperature, 0.6)
        self.assertEqual(sampling.top_p, 0.8)
        self.assertEqual(sampling.max_tokens, 32)

    def test_sampling_params_from_chat_rejects_mismatched_token_limits(self):
        req = ChatCompletionRequest(
            model="rwkv-test",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=16,
            max_completion_tokens=32,
        )
        with self.assertRaises(OpenAIAPIError) as ctx:
            _sampling_params_from_chat(req)
        self.assertEqual(ctx.exception.param, "max_completion_tokens")

    def test_sampling_params_from_completion_rejects_unsupported_logprobs(self):
        req = CompletionRequest(
            model="rwkv-test",
            prompt="hi",
            logprobs=1,
        )
        with self.assertRaises(OpenAIAPIError) as ctx:
            _sampling_params_from_completion(req)
        self.assertEqual(ctx.exception.param, "logprobs")

    def test_render_chat_prompt_fallback_maps_developer_to_system(self):
        prompt = _render_chat_prompt(
            _FallbackTokenizer(),
            [
                ChatMessage(role="developer", content="Follow the rules."),
                ChatMessage(role="user", content=[TextPart(type="text", text="Hello")]),
            ],
        )
        self.assertEqual(prompt, "System: Follow the rules.\nUser: Hello\nAssistant:")

    def test_render_chat_prompt_prefers_template_when_available(self):
        tokenizer = _TemplateTokenizer()
        prompt = _render_chat_prompt(
            tokenizer,
            [ChatMessage(role="user", content="Hello")],
        )
        self.assertEqual(prompt, "templated")
        self.assertEqual(len(tokenizer.calls), 1)
        messages, tokenize, add_generation_prompt = tokenizer.calls[0]
        self.assertEqual(messages, [{"role": "user", "content": "Hello"}])
        self.assertFalse(tokenize)
        self.assertTrue(add_generation_prompt)

    def test_coerce_text_content_rejects_non_text_parts(self):
        with self.assertRaises(OpenAIAPIError) as ctx:
            _coerce_text_content([TextPart(type="image", text=None)])
        self.assertEqual(ctx.exception.param, "messages")

    def test_default_model_name_and_normalize_prompt(self):
        self.assertEqual(_default_model_name("/models/foo/model.pth"), "model")
        self.assertEqual(_normalize_prompt("hello"), "hello")
        self.assertEqual(_normalize_prompt(["hello"]), "hello")
        with self.assertRaises(OpenAIAPIError) as ctx:
            _normalize_prompt(["a", "b"])
        self.assertEqual(ctx.exception.param, "prompt")

    def test_require_api_key_and_headers(self):
        state = ServerState(
            llm=None,
            model_id="rwkv-test",
            created=0,
            api_key="secret",
            lock=threading.Lock(),
        )
        _require_api_key(state, "Bearer secret")
        with self.assertRaises(OpenAIAPIError) as ctx:
            _require_api_key(state, "Bearer wrong")
        self.assertEqual(ctx.exception.status_code, 401)

        headers = _response_headers(
            request_id="req_123",
            prompt_token_count=10,
            completion_token_count=4,
            queue_wait_s=0.010,
            processing_s=0.020,
            ttft_s=0.030,
            generation_s=0.080,
            total_s=0.090,
            streaming=False,
        )
        self.assertEqual(headers["x-request-id"], "req_123")
        self.assertEqual(headers["x-nanovllm-streaming"], "false")
        self.assertEqual(headers["x-nanovllm-completion-tokens"], "4")
        self.assertEqual(headers["x-nanovllm-output-tokens-per-second"], "50.000")
        self.assertEqual(headers["x-nanovllm-decode-tokens-per-second"], "60.000")
        self.assertEqual(_chat_stream_finish_reason("length"), None)
        self.assertEqual(_chat_stream_finish_reason("stop"), "stop")


if __name__ == "__main__":
    unittest.main()
