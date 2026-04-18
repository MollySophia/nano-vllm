import asyncio
import os
import threading
import unittest
from queue import SimpleQueue
from types import SimpleNamespace
from unittest import mock

from nanovllm.engine.sequence import Sequence, SequenceStatus


try:
    from nanovllm.entrypoints.openai.api_server import (
        BatchedRequest,
        ChatMessage,
        ChatCompletionRequest,
        CompletionRequest,
        OpenAIAPIError,
        FrontendResponseBridge,
        PreparedOpenAIRequest,
        PromptTokenCache,
        RequestBatcher,
        ServerState,
        TextPart,
        _chat_stream_finish_reason,
        _coerce_text_content,
        _deserialize_prepared_request,
        _default_model_name,
        _prepare_completion_request,
        _prepare_request_from_payload,
        _resolve_frontend_runtime,
        _normalize_prompt,
        _render_chat_prompt,
        _require_api_key,
        _response_headers,
        _serialize_prepared_request,
        _submit_request,
        _sampling_params_from_chat,
        _sampling_params_from_completion,
        _validated_sampling_params,
        DEFAULT_OPENAI_MAX_TOKENS,
    )
except ModuleNotFoundError as exc:  # pragma: no cover - depends on optional API deps.
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None

try:
    from tests.openai_api_test_utils import FakeLLM
except ModuleNotFoundError:  # pragma: no cover - depends on optional API deps.
    FakeLLM = None


class _FallbackTokenizer:
    pass


class _TemplateTokenizer:
    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        self.calls.append((messages, tokenize, add_generation_prompt))
        return "templated"

    def encode(self, text):
        return [ord(ch) for ch in text]


class _QueueLikeState:
    def __init__(self, tokenizer):
        self.model_id = "rwkv-test"
        self.tokenizer = tokenizer


class _CountingTokenizer:
    def __init__(self):
        self.encode_calls: list[str] = []

    def encode(self, text):
        self.encode_calls.append(text)
        return [ord(ch) for ch in text]


class _BurstScheduler:
    def __init__(self, *, running=None, waiting=None, max_num_seqs=512):
        self.running = list(running or [])
        self.waiting = list(waiting or [])
        self.max_num_seqs = max_num_seqs

    def _prefill_step_tokens(self, seq):
        return seq.prefill_step_tokens(-1)


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
        self.assertEqual(sampling.max_tokens, DEFAULT_OPENAI_MAX_TOKENS)
        self.assertAlmostEqual(sampling.penalty_decay, 0.996)

    def test_validated_sampling_params_maps_openai_penalties_to_internal_penalties(self):
        sampling = _validated_sampling_params(
            temperature=0.4,
            top_p=0.8,
            max_tokens=7,
            presence_penalty=0.6,
            frequency_penalty=0.2,
            penalty_decay=0.91,
        )
        self.assertEqual(sampling.temperature, 0.4)
        self.assertEqual(sampling.top_p, 0.8)
        self.assertEqual(sampling.max_tokens, 7)
        self.assertAlmostEqual(sampling.presence_penalty, 0.6)
        self.assertAlmostEqual(sampling.repetition_penalty, 0.2)
        self.assertAlmostEqual(sampling.penalty_decay, 0.91)

    def test_validated_sampling_params_defaults_penalty_decay_when_omitted(self):
        sampling = _validated_sampling_params(
            temperature=0.4,
            top_p=0.8,
            max_tokens=7,
            presence_penalty=0.6,
            frequency_penalty=0.2,
        )
        self.assertAlmostEqual(sampling.penalty_decay, 0.996)

    def test_prepare_completion_request_uses_default_max_tokens_when_omitted(self):
        state = _QueueLikeState(_TemplateTokenizer())
        req = CompletionRequest(model="rwkv-test", prompt="hello")

        prepared = _prepare_completion_request(state, req)

        self.assertEqual(prepared.requested_max_tokens, DEFAULT_OPENAI_MAX_TOKENS)
        self.assertEqual(prepared.sampling_params.max_tokens, DEFAULT_OPENAI_MAX_TOKENS)

    def test_sampling_params_from_chat_prefers_max_completion_tokens(self):
        req = ChatCompletionRequest(
            model="rwkv-test",
            messages=[{"role": "user", "content": "hi"}],
            temperature=0.6,
            top_p=0.8,
            presence_penalty=0.3,
            frequency_penalty=0.1,
            max_completion_tokens=32,
        )
        sampling = _sampling_params_from_chat(req)
        self.assertEqual(sampling.temperature, 0.6)
        self.assertEqual(sampling.top_p, 0.8)
        self.assertEqual(sampling.max_tokens, 32)
        self.assertAlmostEqual(sampling.presence_penalty, 0.3)
        self.assertAlmostEqual(sampling.repetition_penalty, 0.1)
        self.assertAlmostEqual(sampling.penalty_decay, 0.996)

    def test_sampling_params_from_chat_accepts_stream_options_include_usage(self):
        req = ChatCompletionRequest(
            model="rwkv-test",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=16,
            stream=True,
            stream_options={"include_usage": True},
        )
        sampling = _sampling_params_from_chat(req)
        self.assertEqual(sampling.max_tokens, 16)

    def test_sampling_params_from_chat_rejects_invalid_stream_options(self):
        req = ChatCompletionRequest(
            model="rwkv-test",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=16,
            stream=True,
            stream_options={"include_usage": "yes"},
        )
        with self.assertRaises(OpenAIAPIError) as ctx:
            _sampling_params_from_chat(req)
        self.assertEqual(ctx.exception.param, "stream_options")

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

    def test_sampling_params_from_completion_allows_logprobs_and_echo(self):
        req = CompletionRequest(
            model="rwkv-test",
            prompt="hi",
            logprobs=2,
            echo=True,
            max_tokens=0,
        )
        sampling = _sampling_params_from_completion(req)
        self.assertEqual(sampling.max_tokens, 1)

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
            prompt_token_cache=PromptTokenCache(max_entries=8),
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

    def test_frontend_response_bridge_routes_frames_by_request_id(self):
        async def run_case():
            response_queue = SimpleQueue()
            bridge = FrontendResponseBridge(response_queue)
            bridge.start()
            request_a = bridge.register("req_a")
            request_b = bridge.register("req_b")
            try:
                response_queue.put({"request_id": "req_b", "kind": "delta", "text": "B"})
                response_queue.put({"request_id": "req_a", "kind": "result", "text": "A"})
                frame_b = await asyncio.wait_for(request_b.get(), timeout=1.0)
                frame_a = await asyncio.wait_for(request_a.get(), timeout=1.0)
            finally:
                bridge.unregister("req_a")
                bridge.unregister("req_b")
                bridge.stop()
            self.assertEqual(frame_b["text"], "B")
            self.assertEqual(frame_a["text"], "A")

        asyncio.run(run_case())

    def test_prepared_request_round_trips_through_serialization(self):
        prepared = PreparedOpenAIRequest(
            prompt_text="hello",
            sampling_params=_validated_sampling_params(
                temperature=0.2,
                top_p=0.9,
                max_tokens=12,
                presence_penalty=0.5,
                frequency_penalty=0.25,
            ),
            requested_max_tokens=12,
            prompt_token_ids=[1, 2, 3],
            capture_logprobs=True,
            top_logprobs=3,
            echo=True,
        )

        payload = _serialize_prepared_request(prepared)
        restored = _deserialize_prepared_request(payload)

        self.assertEqual(restored.prompt_text, "hello")
        self.assertEqual(restored.prompt_token_ids, [1, 2, 3])
        self.assertEqual(restored.requested_max_tokens, 12)
        self.assertEqual(restored.sampling_params.temperature, 0.2)
        self.assertEqual(restored.sampling_params.top_p, 0.9)
        self.assertEqual(restored.sampling_params.max_tokens, 12)
        self.assertAlmostEqual(restored.sampling_params.presence_penalty, 0.5)
        self.assertAlmostEqual(restored.sampling_params.repetition_penalty, 0.25)
        self.assertAlmostEqual(restored.sampling_params.penalty_decay, 0.996)
        self.assertTrue(restored.capture_logprobs)
        self.assertEqual(restored.top_logprobs, 3)
        self.assertTrue(restored.echo)

    def test_prepare_request_from_payload_accepts_state_with_tokenizer_only(self):
        state = _QueueLikeState(_TemplateTokenizer())
        prepared = _prepare_request_from_payload(
            state,
            endpoint="chat",
            payload={
                "model": "rwkv-test",
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 4,
            },
        )

        self.assertEqual(prepared.prompt_text, "templated")
        self.assertEqual(prepared.prompt_token_ids, state.tokenizer.encode("templated"))
        self.assertEqual(prepared.requested_max_tokens, 4)

    def test_prepare_completion_request_reuses_prompt_token_cache(self):
        tokenizer = _CountingTokenizer()

        class _CachedState:
            model_id = "rwkv-test"
            prompt_token_cache = PromptTokenCache(max_entries=8)

            def __init__(self, tokenizer):
                self.tokenizer = tokenizer

        state = _CachedState(tokenizer)
        req = CompletionRequest(model="rwkv-test", prompt="hello", max_tokens=4)

        first = _prepare_completion_request(state, req)
        second = _prepare_completion_request(state, req)

        self.assertEqual(first.prompt_token_ids, [104, 101, 108, 108, 111])
        self.assertEqual(second.prompt_token_ids, [104, 101, 108, 108, 111])
        self.assertEqual(tokenizer.encode_calls, ["hello"])

    def test_submit_request_can_complete_via_completion_notify_without_asyncio_signals(self):
        if FakeLLM is None:
            self.skipTest("OpenAI API fake test utilities unavailable.")

        async def run_case():
            llm = FakeLLM("/models/rwkv-test.pth", completion_text="OK")
            batcher = RequestBatcher(llm)
            state = ServerState(
                llm=llm,
                model_id="rwkv-test",
                created=0,
                api_key=None,
                lock=threading.Lock(),
                prompt_token_cache=PromptTokenCache(max_entries=8),
                batcher=batcher,
            )
            completed = SimpleQueue()
            batcher.start()
            try:
                request = _submit_request(
                    state,
                    endpoint="completion",
                    prepared=PreparedOpenAIRequest(
                        prompt_text="abc",
                        sampling_params=_validated_sampling_params(
                            temperature=0.0,
                            top_p=1.0,
                            max_tokens=2,
                        ),
                        requested_max_tokens=2,
                        prompt_token_ids=[97, 98, 99],
                    ),
                    request_id="req_1",
                    created=0,
                    http_received_at=0.0,
                    handler_started_at=0.0,
                    stream=False,
                    completion_notify=completed.put,
                    frontend_id=7,
                    use_async_signals=False,
                )
                self.assertIsNone(request.ready_event)
                self.assertIsNone(request.done_event)
                completed_request = await asyncio.to_thread(completed.get)
                self.assertIs(completed_request, request)
                self.assertEqual(request.frontend_id, 7)
                self.assertEqual(request.finish_reason, "length")
                self.assertEqual(request.completion_token_ids, [79, 75])
            finally:
                batcher.stop()
                llm.exit()

        asyncio.run(run_case())

    def test_submit_request_allocates_ready_event_only_for_streaming(self):
        if FakeLLM is None:
            self.skipTest("OpenAI API fake test utilities unavailable.")

        async def run_case():
            llm = FakeLLM("/models/rwkv-test.pth", completion_text="OK")
            batcher = RequestBatcher(llm)
            state = ServerState(
                llm=llm,
                model_id="rwkv-test",
                created=0,
                api_key=None,
                lock=threading.Lock(),
                prompt_token_cache=PromptTokenCache(max_entries=8),
                batcher=batcher,
            )
            batcher.start()
            try:
                base_prepared = PreparedOpenAIRequest(
                    prompt_text="abc",
                    sampling_params=_validated_sampling_params(
                        temperature=0.0,
                        top_p=1.0,
                        max_tokens=2,
                    ),
                    requested_max_tokens=2,
                    prompt_token_ids=[97, 98, 99],
                )
                non_stream = _submit_request(
                    state,
                    endpoint="completion",
                    prepared=base_prepared,
                    request_id="req_non_stream",
                    created=0,
                    http_received_at=0.0,
                    handler_started_at=0.0,
                    stream=False,
                )
                stream = _submit_request(
                    state,
                    endpoint="completion",
                    prepared=base_prepared,
                    request_id="req_stream",
                    created=0,
                    http_received_at=0.0,
                    handler_started_at=0.0,
                    stream=True,
                )
                self.assertIsNone(non_stream.ready_event)
                self.assertIsNotNone(non_stream.done_event)
                self.assertIsNotNone(stream.ready_event)
                self.assertIsNotNone(stream.done_event)
            finally:
                batcher.stop()
                llm.exit()

        asyncio.run(run_case())

    def test_request_batcher_defers_admission_when_only_small_headroom_is_free(self):
        if FakeLLM is None:
            self.skipTest("OpenAI API fake test utilities unavailable.")

        llm = FakeLLM("/models/rwkv-test.pth", llm_kwargs={"max_num_seqs": 128})
        batcher = RequestBatcher(llm)
        batcher._prefill_admission_reserve_slots = 16
        batcher._prefill_admission_max_delay_s = 0.200
        for seq_id in range(120):
            batcher._active[seq_id] = object()
        batcher._pending.append(
            BatchedRequest(
                request_id="req_pending",
                endpoint="completion",
                prompt_text="hello",
                sampling_params=_validated_sampling_params(
                    temperature=0.0,
                    top_p=1.0,
                    max_tokens=4,
                ),
                requested_max_tokens=4,
                prompt_token_ids_input=[104, 101, 108, 108, 111],
                capture_logprobs=False,
                top_logprobs=0,
                echo=False,
                created=0,
                http_received_at=9.900,
                handler_started_at=9.900,
                http_started_at=9.900,
                stream=False,
            )
        )

        self.assertFalse(batcher._should_admit_pending_requests(10.000, admission_capacity=8))
        self.assertTrue(batcher._should_admit_pending_requests(10.120, admission_capacity=16))
        self.assertTrue(batcher._should_admit_pending_requests(10.200, admission_capacity=8))

    def test_request_batcher_admits_immediately_when_headroom_is_large(self):
        if FakeLLM is None:
            self.skipTest("OpenAI API fake test utilities unavailable.")

        llm = FakeLLM("/models/rwkv-test.pth", llm_kwargs={"max_num_seqs": 128})
        batcher = RequestBatcher(llm)
        batcher._prefill_admission_reserve_slots = 16
        batcher._prefill_admission_max_delay_s = 0.200
        for seq_id in range(80):
            batcher._active[seq_id] = object()
        batcher._pending.append(
            BatchedRequest(
                request_id="req_pending",
                endpoint="completion",
                prompt_text="hello",
                sampling_params=_validated_sampling_params(
                    temperature=0.0,
                    top_p=1.0,
                    max_tokens=4,
                ),
                requested_max_tokens=4,
                prompt_token_ids_input=[104, 101, 108, 108, 111],
                capture_logprobs=False,
                top_logprobs=0,
                echo=False,
                created=0,
                http_received_at=9.990,
                handler_started_at=9.990,
                http_started_at=9.990,
                stream=False,
            )
        )

        self.assertTrue(batcher._should_admit_pending_requests(10.000, admission_capacity=48))

    def test_request_batcher_defers_stage_admission_during_decode_burst(self):
        if FakeLLM is None:
            self.skipTest("OpenAI API fake test utilities unavailable.")

        llm = FakeLLM("/models/rwkv-test.pth", llm_kwargs={"max_num_seqs": 128})
        decode_ready = [Sequence([1, 2, 3, 4]) for _ in range(64)]
        for seq in decode_ready:
            seq.num_cached_tokens = seq.num_prompt_tokens
        llm.scheduler = _BurstScheduler(running=decode_ready, waiting=[], max_num_seqs=128)
        batcher = RequestBatcher(llm)
        batcher._decode_burst_steps = 4
        batcher._decode_burst_min_ready = 32
        batcher._prefill_stage_max_delay_s = 0.200
        batcher._decode_steps_since_prefill = 1
        batcher._active = {idx: object() for idx in range(64)}
        batcher._pending.append(
            BatchedRequest(
                request_id="req_pending",
                endpoint="completion",
                prompt_text="hello",
                sampling_params=_validated_sampling_params(
                    temperature=0.0,
                    top_p=1.0,
                    max_tokens=4,
                ),
                requested_max_tokens=4,
                prompt_token_ids_input=[104, 101, 108, 108, 111],
                capture_logprobs=False,
                top_logprobs=0,
                echo=False,
                created=0,
                http_received_at=9.900,
                handler_started_at=9.900,
                http_started_at=9.900,
                stream=False,
            )
        )

        self.assertFalse(batcher._should_admit_pending_requests(10.000, admission_capacity=32))

    def test_request_batcher_allows_stage_admission_after_decode_burst_budget_is_used(self):
        if FakeLLM is None:
            self.skipTest("OpenAI API fake test utilities unavailable.")

        llm = FakeLLM("/models/rwkv-test.pth", llm_kwargs={"max_num_seqs": 128})
        decode_ready = [Sequence([1, 2, 3, 4]) for _ in range(64)]
        for seq in decode_ready:
            seq.num_cached_tokens = seq.num_prompt_tokens
        llm.scheduler = _BurstScheduler(running=decode_ready, waiting=[], max_num_seqs=128)
        batcher = RequestBatcher(llm)
        batcher._decode_burst_steps = 4
        batcher._decode_burst_min_ready = 32
        batcher._prefill_stage_max_delay_s = 0.200
        batcher._decode_steps_since_prefill = 4
        batcher._active = {idx: object() for idx in range(64)}
        batcher._pending.append(
            BatchedRequest(
                request_id="req_pending",
                endpoint="completion",
                prompt_text="hello",
                sampling_params=_validated_sampling_params(
                    temperature=0.0,
                    top_p=1.0,
                    max_tokens=4,
                ),
                requested_max_tokens=4,
                prompt_token_ids_input=[104, 101, 108, 108, 111],
                capture_logprobs=False,
                top_logprobs=0,
                echo=False,
                created=0,
                http_received_at=9.900,
                handler_started_at=9.900,
                http_started_at=9.900,
                stream=False,
            )
        )

        self.assertTrue(batcher._should_admit_pending_requests(10.000, admission_capacity=32))

    def test_decode_burst_defers_prefill_when_decode_ready_is_high(self):
        if FakeLLM is None:
            self.skipTest("OpenAI API fake test utilities unavailable.")

        llm = FakeLLM("/models/rwkv-test.pth")
        decode_ready = [Sequence([1, 2, 3, 4]) for _ in range(4)]
        for seq in decode_ready:
            seq.num_cached_tokens = seq.num_prompt_tokens
        waiting_seq = Sequence([1, 2, 3, 4, 5])
        llm.scheduler = _BurstScheduler(running=decode_ready, waiting=[waiting_seq])
        batcher = RequestBatcher(llm)
        batcher._decode_burst_steps = 4
        batcher._decode_burst_min_ready = 4
        batcher._prefill_waiting_max_delay_s = 0.200
        batcher._decode_steps_since_prefill = 1
        batcher._active[waiting_seq.seq_id] = BatchedRequest(
            request_id="req_waiting",
            endpoint="completion",
            prompt_text="hello",
            sampling_params=_validated_sampling_params(
                temperature=0.0,
                top_p=1.0,
                max_tokens=4,
            ),
            requested_max_tokens=4,
            prompt_token_ids_input=[104, 101, 108, 108, 111],
            capture_logprobs=False,
            top_logprobs=0,
            echo=False,
            created=0,
            http_received_at=9.950,
            handler_started_at=9.950,
            http_started_at=9.950,
            stream=False,
        )

        self.assertFalse(batcher._should_run_prefill_after_decode(10.000))

    def test_decode_burst_allows_prefill_when_waiting_request_is_old(self):
        if FakeLLM is None:
            self.skipTest("OpenAI API fake test utilities unavailable.")

        llm = FakeLLM("/models/rwkv-test.pth")
        decode_ready = [Sequence([1, 2, 3, 4]) for _ in range(4)]
        for seq in decode_ready:
            seq.num_cached_tokens = seq.num_prompt_tokens
        waiting_seq = Sequence([1, 2, 3, 4, 5])
        llm.scheduler = _BurstScheduler(running=decode_ready, waiting=[waiting_seq])
        batcher = RequestBatcher(llm)
        batcher._decode_burst_steps = 4
        batcher._decode_burst_min_ready = 4
        batcher._prefill_waiting_max_delay_s = 0.020
        batcher._decode_steps_since_prefill = 1
        batcher._active[waiting_seq.seq_id] = BatchedRequest(
            request_id="req_waiting",
            endpoint="completion",
            prompt_text="hello",
            sampling_params=_validated_sampling_params(
                temperature=0.0,
                top_p=1.0,
                max_tokens=4,
            ),
            requested_max_tokens=4,
            prompt_token_ids_input=[104, 101, 108, 108, 111],
            capture_logprobs=False,
            top_logprobs=0,
            echo=False,
            created=0,
            http_received_at=9.950,
            handler_started_at=9.950,
            http_started_at=9.950,
            stream=False,
        )

        self.assertTrue(batcher._should_run_prefill_after_decode(10.000))

    def test_pending_admission_quota_respects_max_prefill_inflight(self):
        if FakeLLM is None:
            self.skipTest("OpenAI API fake test utilities unavailable.")

        llm = FakeLLM("/models/rwkv-test.pth", llm_kwargs={"max_num_seqs": 128})
        decode_ready = [Sequence([1, 2, 3, 4]) for _ in range(4)]
        for seq in decode_ready:
            seq.num_cached_tokens = seq.num_prompt_tokens
        prefill_waiting = [Sequence([1, 2, 3, 4, 5]) for _ in range(6)]
        llm.scheduler = _BurstScheduler(running=decode_ready, waiting=prefill_waiting, max_num_seqs=128)
        batcher = RequestBatcher(llm)
        batcher._max_prefill_inflight = 8
        batcher._prefill_admission_reserve_slots = 0
        batcher._pending.extend(
            [
                BatchedRequest(
                    request_id=f"req_{idx}",
                    endpoint="completion",
                    prompt_text="hello",
                    sampling_params=_validated_sampling_params(
                        temperature=0.0,
                        top_p=1.0,
                        max_tokens=4,
                    ),
                    requested_max_tokens=4,
                    prompt_token_ids_input=[104, 101, 108, 108, 111],
                    capture_logprobs=False,
                    top_logprobs=0,
                    echo=False,
                    created=0,
                    http_received_at=0.0,
                    handler_started_at=0.0,
                    http_started_at=0.0,
                    stream=False,
                )
                for idx in range(16)
            ]
        )

        self.assertEqual(batcher._prefill_inflight_count(), 6)
        self.assertEqual(batcher._pending_admission_quota(10.0, admission_capacity=32), 2)

    def test_pending_admission_quota_can_stage_small_prefill_batch_under_decode_pressure(self):
        if FakeLLM is None:
            self.skipTest("OpenAI API fake test utilities unavailable.")

        llm = FakeLLM("/models/rwkv-test.pth", llm_kwargs={"max_num_seqs": 128})
        decode_ready = [Sequence([1, 2, 3, 4]) for _ in range(64)]
        for seq in decode_ready:
            seq.num_cached_tokens = seq.num_prompt_tokens
        llm.scheduler = _BurstScheduler(running=decode_ready, waiting=[], max_num_seqs=128)
        batcher = RequestBatcher(llm)
        batcher._prefill_admission_reserve_slots = 0
        batcher._max_prefill_inflight = 64
        batcher._prefill_stage_min_batch = 16
        batcher._prefill_stage_max_delay_s = 0.200
        batcher._decode_burst_min_ready = 32
        batcher._active = {idx: object() for idx in range(64)}
        batcher._pending.extend(
            [
                BatchedRequest(
                    request_id=f"req_{idx}",
                    endpoint="completion",
                    prompt_text="hello",
                    sampling_params=_validated_sampling_params(
                        temperature=0.0,
                        top_p=1.0,
                        max_tokens=4,
                    ),
                    requested_max_tokens=4,
                    prompt_token_ids_input=[104, 101, 108, 108, 111],
                    capture_logprobs=False,
                    top_logprobs=0,
                    echo=False,
                    created=0,
                    http_received_at=9.900,
                    handler_started_at=9.900,
                    http_started_at=9.900,
                    stream=False,
                )
                for idx in range(8)
            ]
        )

        self.assertEqual(batcher._pending_admission_quota(10.000, admission_capacity=32), 0)
        self.assertEqual(batcher._pending_admission_quota(10.200, admission_capacity=32), 8)

    def test_can_multi_step_decode_rejects_streaming_and_logprob_requests(self):
        if FakeLLM is None:
            self.skipTest("OpenAI API fake test utilities unavailable.")

        llm = FakeLLM("/models/rwkv-test.pth", llm_kwargs={"max_num_seqs": 128})
        batcher = RequestBatcher(llm)
        batcher._multi_step_decode_tokens = 4
        streaming_seq = Sequence([1, 2, 3, 4])
        logprob_seq = Sequence([1, 2, 3, 4])
        batcher._active[streaming_seq.seq_id] = BatchedRequest(
            request_id="req_stream",
            endpoint="completion",
            prompt_text="hello",
            sampling_params=_validated_sampling_params(temperature=0.0, top_p=1.0, max_tokens=4),
            requested_max_tokens=4,
            prompt_token_ids_input=[104, 101, 108, 108, 111],
            capture_logprobs=False,
            top_logprobs=0,
            echo=False,
            created=0,
            http_received_at=0.0,
            handler_started_at=0.0,
            http_started_at=0.0,
            stream=True,
        )
        batcher._active[logprob_seq.seq_id] = BatchedRequest(
            request_id="req_logprobs",
            endpoint="completion",
            prompt_text="hello",
            sampling_params=_validated_sampling_params(temperature=0.0, top_p=1.0, max_tokens=4),
            requested_max_tokens=4,
            prompt_token_ids_input=[104, 101, 108, 108, 111],
            capture_logprobs=True,
            top_logprobs=1,
            echo=False,
            created=0,
            http_received_at=0.0,
            handler_started_at=0.0,
            http_started_at=0.0,
            stream=False,
        )

        self.assertFalse(batcher._can_multi_step_decode([streaming_seq]))
        self.assertFalse(batcher._can_multi_step_decode([logprob_seq]))

    def test_default_multi_step_decode_tokens_stays_enabled_for_smaller_server_batches(self):
        if FakeLLM is None:
            self.skipTest("OpenAI API fake test utilities unavailable.")

        small_llm = FakeLLM("/models/rwkv-test.pth", llm_kwargs={"max_num_seqs": 128})

        self.assertEqual(RequestBatcher(small_llm)._multi_step_decode_tokens, 4)

    def test_default_multi_step_decode_tokens_is_conservative_for_large_server_batches(self):
        if FakeLLM is None:
            self.skipTest("OpenAI API fake test utilities unavailable.")

        large_llm = FakeLLM("/models/rwkv-test.pth", llm_kwargs={"max_num_seqs": 960})

        self.assertEqual(RequestBatcher(large_llm)._multi_step_decode_tokens, 1)

    def test_decode_burst_defaults_can_be_overridden_by_env(self):
        if FakeLLM is None:
            self.skipTest("OpenAI API fake test utilities unavailable.")

        llm = FakeLLM("/models/rwkv-test.pth", llm_kwargs={"max_num_seqs": 960})
        with mock.patch.dict(
            os.environ,
            {
                "NANOVLLM_DECODE_BURST_STEPS": "8",
                "NANOVLLM_DECODE_BURST_MIN_READY": "128",
            },
            clear=False,
        ):
            batcher = RequestBatcher(llm)
        self.assertEqual(batcher._decode_burst_steps, 8)
        self.assertEqual(batcher._decode_burst_min_ready, 128)

    def test_prefill_waiting_delay_can_be_overridden_by_env(self):
        if FakeLLM is None:
            self.skipTest("OpenAI API fake test utilities unavailable.")

        llm = FakeLLM("/models/rwkv-test.pth", llm_kwargs={"max_num_seqs": 960})
        with mock.patch.dict(
            os.environ,
            {
                "NANOVLLM_PREFILL_WAITING_MAX_DELAY_S": "0.35",
            },
            clear=False,
        ):
            batcher = RequestBatcher(llm)
        self.assertAlmostEqual(batcher._prefill_waiting_max_delay_s, 0.35)

    def test_decode_schedule_max_batch_defaults_for_state_cache(self):
        if FakeLLM is None:
            self.skipTest("OpenAI API fake test utilities unavailable.")

        llm = FakeLLM(
            "/models/rwkv-test.pth",
            llm_kwargs={"max_num_seqs": 960, "rwkv_state_cache_enable": True},
        )
        batcher = RequestBatcher(llm)
        self.assertEqual(batcher._decode_schedule_max_batch, 256)

    def test_max_prefill_inflight_scales_up_for_large_non_cache_server_batches(self):
        if FakeLLM is None:
            self.skipTest("OpenAI API fake test utilities unavailable.")

        llm = FakeLLM("/models/rwkv-test.pth", llm_kwargs={"max_num_seqs": 960})
        batcher = RequestBatcher(llm)
        self.assertEqual(batcher._max_prefill_inflight, 240)

    def test_max_prefill_inflight_stays_conservative_for_state_cache(self):
        if FakeLLM is None:
            self.skipTest("OpenAI API fake test utilities unavailable.")

        llm = FakeLLM(
            "/models/rwkv-test.pth",
            llm_kwargs={"max_num_seqs": 960, "rwkv_state_cache_enable": True},
        )
        batcher = RequestBatcher(llm)
        self.assertEqual(batcher._max_prefill_inflight, 64)

    def test_schedule_split_batch_clamps_decode_batch_size(self):
        if FakeLLM is None:
            self.skipTest("OpenAI API fake test utilities unavailable.")

        llm = FakeLLM("/models/rwkv-test.pth", llm_kwargs={"max_num_seqs": 960})
        batcher = RequestBatcher(llm)
        batcher._decode_schedule_max_batch = 3

        seqs = []
        for _ in range(5):
            seq = llm.add_request(
                [97, 98, 99],
                _validated_sampling_params(temperature=0.0, top_p=1.0, max_tokens=2),
            )
            seq.num_cached_tokens = seq.num_prompt_tokens
            seq.status = SequenceStatus.RUNNING
            seqs.append(seq)
        llm.scheduler.waiting.clear()
        llm.scheduler.running.extend(seqs)

        scheduled = batcher._schedule_split_batch(is_prefill=False)
        self.assertIsNotNone(scheduled)
        self.assertEqual(len(scheduled), 3)
        self.assertEqual([seq.seq_id for seq in scheduled], [seq.seq_id for seq in seqs[:3]])

    def test_run_split_steps_can_decode_multiple_tokens_for_non_streaming_request(self):
        if FakeLLM is None:
            self.skipTest("OpenAI API fake test utilities unavailable.")

        llm = FakeLLM("/models/rwkv-test.pth", completion_text="OK", llm_kwargs={"max_num_seqs": 128})
        batcher = RequestBatcher(llm)
        batcher._multi_step_decode_tokens = 4
        requests = []
        seqs = []
        for idx in range(8):
            seq = llm.add_request(
                [97, 98, 99],
                _validated_sampling_params(temperature=0.0, top_p=1.0, max_tokens=2),
            )
            request = BatchedRequest(
                request_id=f"req_non_stream_{idx}",
                endpoint="completion",
                prompt_text="abc",
                sampling_params=_validated_sampling_params(temperature=0.0, top_p=1.0, max_tokens=2),
                requested_max_tokens=2,
                prompt_token_ids_input=[97, 98, 99],
                capture_logprobs=False,
                top_logprobs=0,
                echo=False,
                created=0,
                http_received_at=0.0,
                handler_started_at=0.0,
                http_started_at=0.0,
                stream=False,
            )
            request.prompt_token_ids = [97, 98, 99]
            request.seq_id = seq.seq_id
            batcher._active[seq.seq_id] = request
            seq.num_cached_tokens = seq.num_prompt_tokens
            requests.append(request)
            seqs.append(seq)
        llm.scheduler.waiting.clear()
        llm.scheduler.running.extend(seqs)

        self.assertTrue(batcher._run_split_steps())
        self.assertEqual([call[0] for call in llm.model_runner.calls], ["run", "run"])
        self.assertTrue(all(request.completion_token_ids == [79, 75] for request in requests))
        self.assertTrue(all(request.finish_reason == "length" for request in requests))

    def test_run_split_steps_keeps_streaming_request_single_step(self):
        if FakeLLM is None:
            self.skipTest("OpenAI API fake test utilities unavailable.")

        llm = FakeLLM("/models/rwkv-test.pth", completion_text="OK", llm_kwargs={"max_num_seqs": 128})
        batcher = RequestBatcher(llm)
        batcher._multi_step_decode_tokens = 4
        requests = []
        seqs = []
        for idx in range(8):
            seq = llm.add_request(
                [97, 98, 99],
                _validated_sampling_params(temperature=0.0, top_p=1.0, max_tokens=2),
            )
            request = BatchedRequest(
                request_id=f"req_stream_{idx}",
                endpoint="completion",
                prompt_text="abc",
                sampling_params=_validated_sampling_params(temperature=0.0, top_p=1.0, max_tokens=2),
                requested_max_tokens=2,
                prompt_token_ids_input=[97, 98, 99],
                capture_logprobs=False,
                top_logprobs=0,
                echo=False,
                created=0,
                http_received_at=0.0,
                handler_started_at=0.0,
                http_started_at=0.0,
                stream=True,
                stream_queue=asyncio.Queue(),
            )
            request.prompt_token_ids = [97, 98, 99]
            request.seq_id = seq.seq_id
            batcher._active[seq.seq_id] = request
            seq.num_cached_tokens = seq.num_prompt_tokens
            requests.append(request)
            seqs.append(seq)
        llm.scheduler.waiting.clear()
        llm.scheduler.running.extend(seqs)

        self.assertTrue(batcher._run_split_steps())
        self.assertEqual([call[0] for call in llm.model_runner.calls], ["run"])
        self.assertTrue(all(request.completion_token_ids == [79] for request in requests))
        self.assertTrue(all(request.finish_reason is None for request in requests))

    def test_prefill_stage_target_grows_with_large_decode_ready_batch(self):
        if FakeLLM is None:
            self.skipTest("OpenAI API fake test utilities unavailable.")

        llm = FakeLLM("/models/rwkv-test.pth", llm_kwargs={"max_num_seqs": 512})
        decode_ready = [Sequence([1, 2, 3, 4]) for _ in range(384)]
        for seq in decode_ready:
            seq.num_cached_tokens = seq.num_prompt_tokens
        llm.scheduler = _BurstScheduler(running=decode_ready, waiting=[])
        batcher = RequestBatcher(llm)
        batcher._prefill_stage_min_batch = 16
        batcher._decode_burst_min_ready = 64

        self.assertEqual(batcher._prefill_stage_target(), 38)

    def test_pop_staged_requests_prefers_prefill_stage_before_new_pending(self):
        if FakeLLM is None:
            self.skipTest("OpenAI API fake test utilities unavailable.")

        llm = FakeLLM("/models/rwkv-test.pth", llm_kwargs={"max_num_seqs": 128})
        batcher = RequestBatcher(llm)
        staged_request = BatchedRequest(
            request_id="req_staged",
            endpoint="completion",
            prompt_text="old",
            sampling_params=_validated_sampling_params(
                temperature=0.0,
                top_p=1.0,
                max_tokens=4,
            ),
            requested_max_tokens=4,
            prompt_token_ids_input=[111, 108, 100],
            capture_logprobs=False,
            top_logprobs=0,
            echo=False,
            created=0,
            http_received_at=9.800,
            handler_started_at=9.800,
            http_started_at=9.800,
            stream=False,
        )
        fresh_request = BatchedRequest(
            request_id="req_pending",
            endpoint="completion",
            prompt_text="new",
            sampling_params=_validated_sampling_params(
                temperature=0.0,
                top_p=1.0,
                max_tokens=4,
            ),
            requested_max_tokens=4,
            prompt_token_ids_input=[110, 101, 119],
            capture_logprobs=False,
            top_logprobs=0,
            echo=False,
            created=0,
            http_received_at=9.900,
            handler_started_at=9.900,
            http_started_at=9.900,
            stream=False,
        )
        batcher._prefill_stage.append(staged_request)
        batcher._pending.append(fresh_request)

        popped = batcher._pop_staged_requests(2)

        self.assertEqual([request.request_id for request in popped], ["req_staged", "req_pending"])

    def test_cancelled_pending_request_is_dropped_before_admission(self):
        if FakeLLM is None:
            self.skipTest("OpenAI API fake test utilities unavailable.")

        llm = FakeLLM("/models/rwkv-test.pth", llm_kwargs={"max_num_seqs": 128})
        batcher = RequestBatcher(llm)
        request = BatchedRequest(
            request_id="req_cancel_pending",
            endpoint="completion",
            prompt_text="abc",
            sampling_params=_validated_sampling_params(
                temperature=0.0,
                top_p=1.0,
                max_tokens=4,
            ),
            requested_max_tokens=4,
            prompt_token_ids_input=[97, 98, 99],
            capture_logprobs=False,
            top_logprobs=0,
            echo=False,
            created=0,
            http_received_at=0.0,
            handler_started_at=0.0,
            http_started_at=0.0,
            stream=False,
        )
        batcher._pending.append(request)
        batcher.cancel(request)
        batcher._stage_pending_requests()
        batcher._drop_cancelled_pending_requests()

        self.assertFalse(batcher._pending)
        self.assertFalse(batcher._prefill_stage)
        self.assertEqual(request.finish_reason, "cancelled")
        self.assertEqual(llm.received_requests, [])

    def test_cancelled_active_request_aborts_scheduler_sequence(self):
        if FakeLLM is None:
            self.skipTest("OpenAI API fake test utilities unavailable.")

        llm = FakeLLM("/models/rwkv-test.pth", completion_text="OK", llm_kwargs={"max_num_seqs": 128})
        batcher = RequestBatcher(llm)
        seq = llm.add_request(
            [97, 98, 99],
            _validated_sampling_params(temperature=0.0, top_p=1.0, max_tokens=4),
        )
        request = BatchedRequest(
            request_id="req_cancel_active",
            endpoint="completion",
            prompt_text="abc",
            sampling_params=_validated_sampling_params(
                temperature=0.0,
                top_p=1.0,
                max_tokens=4,
            ),
            requested_max_tokens=4,
            prompt_token_ids_input=[97, 98, 99],
            capture_logprobs=False,
            top_logprobs=0,
            echo=False,
            created=0,
            http_received_at=0.0,
            handler_started_at=0.0,
            http_started_at=0.0,
            stream=True,
            stream_queue=asyncio.Queue(),
        )
        request.prompt_token_ids = [97, 98, 99]
        request.seq_id = seq.seq_id
        batcher._active[seq.seq_id] = request
        llm.scheduler.waiting.clear()
        llm.scheduler.running.append(seq)

        batcher.cancel(request)
        batcher._abort_cancelled_active_requests()

        self.assertNotIn(seq.seq_id, batcher._active)
        self.assertFalse(llm.scheduler.running)
        self.assertIsNone(llm._pending_token_ids_by_seq.get(seq.seq_id))
        self.assertEqual(request.finish_reason, "cancelled")

    def test_resolve_frontend_runtime_prefers_shared_threads_in_auto_mode(self):
        args = SimpleNamespace(
            frontend_mode="auto",
            listener_threads=1,
            frontend_workers=4,
            tensor_parallel_size=1,
        )
        mode, listener_threads = _resolve_frontend_runtime(args)
        self.assertEqual(mode, "shared")
        self.assertEqual(listener_threads, 4)

    def test_resolve_frontend_runtime_keeps_queue_mode_when_forced(self):
        args = SimpleNamespace(
            frontend_mode="queue",
            listener_threads=1,
            frontend_workers=4,
            tensor_parallel_size=1,
        )
        mode, listener_threads = _resolve_frontend_runtime(args)
        self.assertEqual(mode, "queue")
        self.assertEqual(listener_threads, 1)


if __name__ == "__main__":
    unittest.main()
