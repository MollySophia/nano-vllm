import asyncio
import unittest

import httpx


try:
    from tests.openai_api_test_utils import patched_app
except ModuleNotFoundError as exc:  # pragma: no cover - depends on optional API deps.
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


class OpenAIAPIConcurrencyTest(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        if IMPORT_ERROR is not None:
            raise unittest.SkipTest(f"OpenAI API test dependencies unavailable: {IMPORT_ERROR}")

    async def test_requests_can_batch_together_without_large_queue_wait(self):
        with patched_app(completion_text="ABCD", per_token_delay_s=0.03) as (app, llm, _factory):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
                payload = {
                    "model": "rwkv-test",
                    "prompt": "abc",
                    "max_tokens": 4,
                }
                responses = await asyncio.gather(
                    client.post("/v1/completions", json=payload),
                    client.post("/v1/completions", json=payload),
                )

        self.assertEqual([response.status_code for response in responses], [200, 200])
        waits_ms = sorted(float(response.headers["x-nanovllm-queue-wait-ms"]) for response in responses)
        self.assertLess(waits_ms[0], 50.0)
        self.assertLess(waits_ms[1], 50.0)
        self.assertEqual(
            [response.json()["choices"][0]["text"] for response in responses],
            ["ABCD", "ABCD"],
        )
        self.assertEqual(len(llm.received_requests), 2)

    async def test_decode_batch_runs_before_new_prefill_batch(self):
        with patched_app(completion_text="ABCD", per_token_delay_s=0.05) as (app, llm, _factory):
            assert app.state.server.batcher is not None
            app.state.server.batcher._cold_start_batch_wait_s = 0.0
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
                payload = {
                    "model": "rwkv-test",
                    "max_tokens": 4,
                }
                first_task = asyncio.create_task(
                    client.post("/v1/completions", json={**payload, "prompt": "first"})
                )
                started = await asyncio.to_thread(
                    llm.model_runner.wait_for_call_count,
                    1,
                    1.0,
                )
                self.assertTrue(started)
                second_task = asyncio.create_task(
                    client.post("/v1/completions", json={**payload, "prompt": "second"})
                )
                first_response, second_response = await asyncio.gather(first_task, second_task)

        self.assertEqual(first_response.status_code, 200)
        self.assertEqual(second_response.status_code, 200)
        self.assertGreaterEqual(len(llm.model_runner.calls), 3)
        first_seq_id = llm.model_runner.calls[0][2][0]
        self.assertEqual(llm.model_runner.calls[0], ("run", True, [first_seq_id]))
        self.assertEqual(llm.model_runner.calls[1], ("run", False, [first_seq_id]))
        self.assertEqual(llm.model_runner.calls[2][0], "run")
        self.assertTrue(llm.model_runner.calls[2][1])
        self.assertNotEqual(llm.model_runner.calls[2][2], [first_seq_id])

    async def test_pending_requests_wait_for_active_capacity(self):
        with patched_app(
            completion_text="ABCD",
            per_token_delay_s=0.05,
            llm_kwargs={"max_num_seqs": 1},
        ) as (app, llm, _factory):
            assert app.state.server.batcher is not None
            app.state.server.batcher._cold_start_batch_wait_s = 0.0
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
                payload = {
                    "model": "rwkv-test",
                    "max_tokens": 4,
                }
                first_task = asyncio.create_task(
                    client.post("/v1/completions", json={**payload, "prompt": "first"})
                )
                started = await asyncio.to_thread(
                    llm.model_runner.wait_for_call_count,
                    1,
                    1.0,
                )
                self.assertTrue(started)
                second_task = asyncio.create_task(
                    client.post("/v1/completions", json={**payload, "prompt": "second"})
                )
                await asyncio.sleep(0.02)
                self.assertEqual(len(llm.received_requests), 1)
                first_response, second_response = await asyncio.gather(first_task, second_task)

        self.assertEqual(first_response.status_code, 200)
        self.assertEqual(second_response.status_code, 200)
        self.assertEqual(len(llm.received_requests), 2)


if __name__ == "__main__":
    unittest.main()
