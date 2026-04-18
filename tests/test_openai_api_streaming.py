import json
import time
import unittest


try:
    from tests.openai_api_test_utils import patched_test_client
except ModuleNotFoundError as exc:  # pragma: no cover - depends on optional API deps.
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


def _read_sse_payloads(response) -> list[str]:
    payloads: list[str] = []
    for line in response.iter_lines():
        if not line or not line.startswith("data: "):
            continue
        payloads.append(line[6:])
    return payloads


class OpenAIAPIStreamingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if IMPORT_ERROR is not None:
            raise unittest.SkipTest(f"OpenAI API test dependencies unavailable: {IMPORT_ERROR}")

    def test_completion_stream_emits_text_deltas_finish_chunk_and_done(self):
        with patched_test_client(completion_text="HEY") as (client, _llm, _factory):
            with client.stream(
                "POST",
                "/v1/completions",
                json={
                    "model": "rwkv-test",
                    "prompt": "abc",
                    "max_tokens": 5,
                    "stream": True,
                },
            ) as response:
                payloads = _read_sse_payloads(response)
                headers = dict(response.headers)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(headers["x-nanovllm-streaming"], "true")
        self.assertEqual(headers["x-nanovllm-metrics-scope"], "partial")
        self.assertNotIn("x-nanovllm-completion-tokens", headers)
        self.assertEqual(payloads[-1], "[DONE]")
        events = [json.loads(payload) for payload in payloads[:-1]]
        self.assertEqual([event["choices"][0]["text"] for event in events[:-1]], ["H", "E", "Y"])
        self.assertEqual(events[-1]["choices"][0]["text"], "")
        self.assertEqual(events[-1]["choices"][0]["finish_reason"], "stop")
        self.assertEqual(events[0]["object"], "text_completion")

    def test_chat_stream_emits_assistant_role_then_content_and_suppresses_length_finish_reason(self):
        with patched_test_client(chat_text="OK") as (client, _llm, _factory):
            with client.stream(
                "POST",
                "/v1/chat/completions",
                json={
                    "model": "rwkv-test",
                    "messages": [{"role": "user", "content": "Hello"}],
                    "max_tokens": 2,
                    "stream": True,
                },
            ) as response:
                payloads = _read_sse_payloads(response)
                headers = dict(response.headers)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(headers["x-nanovllm-streaming"], "true")
        self.assertEqual(headers["x-nanovllm-metrics-scope"], "partial")
        self.assertEqual(payloads[-1], "[DONE]")
        events = [json.loads(payload) for payload in payloads[:-1]]
        self.assertEqual(
            events[0]["choices"][0]["delta"],
            {"role": "assistant", "content": ""},
        )
        self.assertEqual(
            [event["choices"][0]["delta"] for event in events[1:-1]],
            [{"content": "O"}, {"content": "K"}],
        )
        self.assertEqual(events[-1]["choices"][0]["delta"], {})
        self.assertIsNone(events[-1]["choices"][0]["finish_reason"])
        self.assertEqual(events[0]["object"], "chat.completion.chunk")

    def test_chat_stream_can_emit_usage_chunk_when_requested(self):
        with patched_test_client(chat_text="OK") as (client, _llm, _factory):
            with client.stream(
                "POST",
                "/v1/chat/completions",
                json={
                    "model": "rwkv-test",
                    "messages": [{"role": "user", "content": "Hello"}],
                    "max_tokens": 2,
                    "stream": True,
                    "stream_options": {"include_usage": True},
                },
            ) as response:
                payloads = _read_sse_payloads(response)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(payloads[-1], "[DONE]")
        events = [json.loads(payload) for payload in payloads[:-1]]
        usage_event = events[-1]
        self.assertEqual(usage_event["object"], "chat.completion.chunk")
        self.assertEqual(usage_event["choices"], [])
        self.assertEqual(usage_event["usage"]["completion_tokens"], 2)
        self.assertGreater(usage_event["usage"]["prompt_tokens"], 0)
        self.assertEqual(
            usage_event["usage"]["total_tokens"],
            usage_event["usage"]["prompt_tokens"] + usage_event["usage"]["completion_tokens"],
        )
        self.assertEqual(events[-2]["choices"][0]["delta"], {})

    def test_completion_stream_can_emit_usage_chunk_when_requested(self):
        with patched_test_client(completion_text="HEY") as (client, _llm, _factory):
            with client.stream(
                "POST",
                "/v1/completions",
                json={
                    "model": "rwkv-test",
                    "prompt": "abc",
                    "max_tokens": 5,
                    "stream": True,
                    "stream_options": {"include_usage": True},
                },
            ) as response:
                payloads = _read_sse_payloads(response)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(payloads[-1], "[DONE]")
        events = [json.loads(payload) for payload in payloads[:-1]]
        usage_event = events[-1]
        self.assertEqual(usage_event["object"], "text_completion")
        self.assertEqual(usage_event["choices"], [])
        self.assertEqual(usage_event["usage"]["completion_tokens"], 3)
        self.assertGreater(usage_event["usage"]["prompt_tokens"], 0)
        self.assertEqual(events[-2]["choices"][0]["text"], "")

    def test_completion_stream_disconnect_cancels_backend_request(self):
        with patched_test_client(
            completion_text="ABCDEFGHIJKLMNOPQRSTUVWXYZ" * 10,
            per_token_delay_s=0.01,
        ) as (client, llm, _factory):
            with client.stream(
                "POST",
                "/v1/completions",
                json={
                    "model": "rwkv-test",
                    "prompt": "abc",
                    "max_tokens": 512,
                    "stream": True,
                },
            ) as response:
                iterator = response.iter_lines()
                first_payload = next(line for line in iterator if line and line.startswith("data: "))
                self.assertTrue(first_payload.startswith("data: "))

            deadline = time.time() + 1.0
            while time.time() < deadline and not llm.is_finished():
                time.sleep(0.01)

        self.assertTrue(llm.is_finished())


if __name__ == "__main__":
    unittest.main()
