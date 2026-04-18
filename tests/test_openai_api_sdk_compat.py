import asyncio
import unittest

import httpx


try:
    from fastapi.testclient import TestClient
    from openai import AsyncOpenAI, AuthenticationError, NotFoundError, OpenAI
    from tests.openai_api_test_utils import patched_app
except ModuleNotFoundError as exc:  # pragma: no cover - depends on optional API deps.
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


class OpenAIAPISDKCompatTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if IMPORT_ERROR is not None:
            raise unittest.SkipTest(f"OpenAI API test dependencies unavailable: {IMPORT_ERROR}")

    def test_openai_sdk_sync_completion_create_returns_parsed_object(self):
        with patched_app(completion_text="OK") as (app, _llm, _factory):
            with TestClient(app) as http_client:
                sdk = OpenAI(
                    api_key="local-test-key",
                    base_url="http://testserver/v1",
                    http_client=http_client,
                )
                try:
                    completion = sdk.completions.create(
                        model="rwkv-test",
                        prompt="abc",
                        max_tokens=4,
                        temperature=0.0,
                    )
                finally:
                    sdk.close()

        self.assertEqual(completion.model, "rwkv-test")
        self.assertEqual(completion.choices[0].text, "OK")
        self.assertEqual(completion.choices[0].finish_reason, "stop")
        self.assertEqual(completion.usage.prompt_tokens, 3)
        self.assertEqual(completion.usage.completion_tokens, 2)

    def test_openai_sdk_sync_completion_accepts_penalty_decay_via_extra_body(self):
        with patched_app(completion_text="OK") as (app, llm, _factory):
            with TestClient(app) as http_client:
                sdk = OpenAI(
                    api_key="local-test-key",
                    base_url="http://testserver/v1",
                    http_client=http_client,
                )
                try:
                    completion = sdk.completions.create(
                        model="rwkv-test",
                        prompt="abc",
                        max_tokens=4,
                        temperature=0.0,
                        extra_body={"penalty_decay": 0.9},
                    )
                finally:
                    sdk.close()

        self.assertEqual(completion.choices[0].text, "OK")
        self.assertAlmostEqual(llm.received_requests[0]["sampling_params"].penalty_decay, 0.9)

    def test_openai_sdk_chat_stream_get_final_completion_works_for_length_stop(self):
        with patched_app(chat_text="CHAT") as (app, _llm, _factory):
            with TestClient(app) as http_client:
                sdk = OpenAI(
                    api_key="local-test-key",
                    base_url="http://testserver/v1",
                    http_client=http_client,
                )
                content_deltas: list[str] = []
                try:
                    with sdk.chat.completions.stream(
                        model="rwkv-test",
                        messages=[{"role": "user", "content": "hello"}],
                        max_tokens=2,
                    ) as stream:
                        for event in stream:
                            if event.type == "content.delta":
                                content_deltas.append(event.delta)
                        final = stream.get_final_completion()
                finally:
                    sdk.close()

        self.assertEqual([delta for delta in content_deltas if delta], ["C", "H"])
        self.assertEqual(final.choices[0].message.content, "CH")
        self.assertEqual(final.choices[0].finish_reason, None)

    def test_openai_sdk_authentication_error_preserves_openai_fields(self):
        with patched_app(api_key="secret") as (app, _llm, _factory):
            with TestClient(app) as http_client:
                sdk = OpenAI(
                    api_key="wrong-key",
                    base_url="http://testserver/v1",
                    http_client=http_client,
                )
                try:
                    with self.assertRaises(AuthenticationError) as ctx:
                        sdk.completions.create(
                            model="rwkv-test",
                            prompt="abc",
                            max_tokens=4,
                        )
                finally:
                    sdk.close()

        self.assertEqual(ctx.exception.status_code, 401)
        self.assertEqual(ctx.exception.code, "invalid_api_key")
        self.assertEqual(ctx.exception.type, "authentication_error")
        self.assertEqual(ctx.exception.param, None)

    def test_openai_sdk_not_found_error_preserves_model_error_fields(self):
        with patched_app() as (app, _llm, _factory):
            with TestClient(app) as http_client:
                sdk = OpenAI(
                    api_key="local-test-key",
                    base_url="http://testserver/v1",
                    http_client=http_client,
                )
                try:
                    with self.assertRaises(NotFoundError) as ctx:
                        sdk.completions.create(
                            model="missing-model",
                            prompt="abc",
                            max_tokens=4,
                        )
                finally:
                    sdk.close()

        self.assertEqual(ctx.exception.status_code, 404)
        self.assertEqual(ctx.exception.code, "model_not_found")
        self.assertEqual(ctx.exception.type, "invalid_request_error")
        self.assertEqual(ctx.exception.param, "model")


class OpenAIAPIAsyncSDKCompatTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if IMPORT_ERROR is not None:
            raise unittest.SkipTest(f"OpenAI API test dependencies unavailable: {IMPORT_ERROR}")

    def test_openai_async_sdk_chat_completion_create_returns_parsed_object(self):
        async def run():
            with patched_app(chat_text="CHAT") as (app, _llm, _factory):
                http_client = httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=app),
                    base_url="http://testserver",
                )
                sdk = AsyncOpenAI(
                    api_key="local-test-key",
                    base_url="http://testserver/v1",
                    http_client=http_client,
                )
                try:
                    return await sdk.chat.completions.create(
                        model="rwkv-test",
                        messages=[{"role": "user", "content": "hello"}],
                        max_tokens=8,
                        temperature=0.0,
                    )
                finally:
                    await sdk.close()

        completion = asyncio.run(run())
        self.assertEqual(completion.model, "rwkv-test")
        self.assertEqual(completion.choices[0].message.content, "CHAT")
        self.assertEqual(completion.choices[0].finish_reason, "stop")
        self.assertEqual(completion.usage.prompt_tokens, len("User: hello\nAssistant:"))
        self.assertEqual(completion.usage.completion_tokens, 4)


if __name__ == "__main__":
    unittest.main()
