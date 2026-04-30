from __future__ import annotations

import argparse
import asyncio
import json
import math
import multiprocessing as mp
import os
import queue
import signal
import socket
import threading
import time
import uuid
from collections import OrderedDict, deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Literal

import httpx
import msgspec
import torch
import uvicorn
from aiohttp import web
from fastapi import FastAPI, Header, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, ConfigDict

from nanovllm import LLM, SamplingParams
from nanovllm.entrypoints.openai.streaming_markdown_restorer import StreamingMarkdownRestorer
from nanovllm.entrypoints.openai.streaming_string_parser import StreamingStringParser, TRIE_THINK_NO_TRIGGER
from nanovllm.tokenizers import RWKVTokenizer, get_rwkv_tokenizer
from nanovllm.utils.rwkv_int8 import (
    add_rwkv_int8_cli_args,
)

try:
    import uvloop
except ModuleNotFoundError:  # pragma: no cover
    uvloop = None


def _format_public_ready_urls(host: str | None, port: int | None) -> list[str]:
    display_host = host or "127.0.0.1"
    display_port = int(port or 8000)
    host_variants = [display_host]
    if display_host == "0.0.0.0":
        host_variants = ["127.0.0.1", "0.0.0.0"]
    elif display_host == "::":
        host_variants = ["[::1]", "[::]"]

    urls: list[str] = []
    seen: set[str] = set()
    for variant in host_variants:
        if ":" in variant and not variant.startswith("["):
            variant = f"[{variant}]"
        url = f"http://{variant}:{display_port}/v1"
        if url in seen:
            continue
        seen.add(url)
        urls.append(url)
    return urls


def _print_public_ready_banner(host: str | None, port: int | None) -> None:
    urls = _format_public_ready_urls(host, port)
    print("[nano-vllm] backend warmup complete; OpenAI API ready:", flush=True)
    for url in urls:
        print(f"[nano-vllm] {url}", flush=True)


def _make_once_callback(callback: Callable[[], None]) -> Callable[[], None]:
    fired = False
    lock = threading.Lock()

    def _wrapped() -> None:
        nonlocal fired
        with lock:
            if fired:
                return
            fired = True
        callback()

    return _wrapped


def _set_ready_event(ready_event: Any | None) -> None:
    if ready_event is not None:
        ready_event.set()


class _ReadyAwareUvicornServer(uvicorn.Server):
    def __init__(self, config: uvicorn.Config, ready_callback: Callable[[], None] | None = None):
        super().__init__(config)
        self._ready_callback = ready_callback
        self._ready_callback_fired = False

    async def startup(self, sockets: list[socket.socket] | None = None) -> None:
        await super().startup(sockets)
        if self._ready_callback is None or self._ready_callback_fired or self.should_exit:
            return
        self._ready_callback_fired = True
        self._ready_callback()


def _default_model_name(model_path: str) -> str:
    base = os.path.basename(model_path.rstrip("/"))
    if base.endswith(".pth"):
        return base[:-4]
    return base or "nano-vllm"


class OpenAIAPIError(Exception):
    def __init__(
        self,
        status_code: int,
        message: str,
        *,
        error_type: str = "invalid_request_error",
        param: str | None = None,
        code: str | None = None,
    ):
        self.status_code = status_code
        self.message = message
        self.error_type = error_type
        self.param = param
        self.code = code


class TextPart(BaseModel):
    type: str
    text: str | None = None


class ChatMessage(BaseModel):
    role: Literal["system", "developer", "user", "assistant", "tool"]
    content: str | list[TextPart] | None


DEFAULT_OPENAI_MAX_TOKENS = 2048
DEFAULT_OPENAI_PENALTY_DECAY = 0.996


class CompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str
    prompt: str | list[str] | None = None
    prompt_token_ids: list[int] | None = None
    max_tokens: int | None = None
    temperature: float | None = None
    stream: bool | None = False
    n: int | None = 1
    top_p: float | None = None
    stop: str | list[str] | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    penalty_decay: float | None = None
    logprobs: int | bool | None = None
    echo: bool | None = None
    seed: int | None = None
    user: str | None = None
    stream_options: dict[str, Any] | None = None


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str
    messages: list[ChatMessage]
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    temperature: float | None = None
    stream: bool | None = False
    n: int | None = 1
    top_p: float | None = None
    stop: str | list[str] | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    penalty_decay: float | None = None
    logprobs: bool | None = None
    top_logprobs: int | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None
    parallel_tool_calls: bool | None = None
    response_format: dict[str, Any] | None = None
    seed: int | None = None
    user: str | None = None
    stream_options: dict[str, Any] | None = None


@dataclass(slots=True)
class ParsedCompletionRequest:
    model: str
    prompt: str | list[str] | None = None
    prompt_token_ids: list[int] | None = None
    max_tokens: int | None = None
    temperature: float | None = None
    stream: bool | None = False
    n: int | None = 1
    top_p: float | None = None
    stop: str | list[str] | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    penalty_decay: float | None = None
    logprobs: int | bool | None = None
    echo: bool | None = None
    seed: int | None = None
    user: str | None = None
    stream_options: dict[str, Any] | None = None
    model_extra: dict[str, Any] | None = None


@dataclass(slots=True)
class ParsedChatCompletionRequest:
    model: str
    messages: list[dict[str, Any]]
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    temperature: float | None = None
    stream: bool | None = False
    n: int | None = 1
    top_p: float | None = None
    stop: str | list[str] | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    penalty_decay: float | None = None
    logprobs: bool | None = None
    top_logprobs: int | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None
    parallel_tool_calls: bool | None = None
    response_format: dict[str, Any] | None = None
    seed: int | None = None
    user: str | None = None
    stream_options: dict[str, Any] | None = None
    model_extra: dict[str, Any] | None = None


@dataclass
class ServerState:
    llm: LLM
    model_id: str
    created: int
    api_key: str | None
    lock: threading.Lock
    prompt_token_cache: "PromptTokenCache"
    batcher: "RequestBatcher | None" = None


@dataclass
class IPCFrontendState:
    model_id: str
    created: int
    api_key: str | None
    backend_uds: str
    backend_channel_count: int
    prompt_token_cache: "PromptTokenCache"


@dataclass
class QueueFrontendState:
    model_id: str
    created: int
    api_key: str | None
    frontend_id: int
    request_queue: Any
    response_queue: Any
    tokenizer: Any
    prompt_token_cache: "PromptTokenCache"


@dataclass(frozen=True)
class PreparedOpenAIRequest:
    prompt_text: str | None
    sampling_params: SamplingParams
    requested_max_tokens: int
    prompt_token_ids: list[int] | None = None
    capture_logprobs: bool = False
    top_logprobs: int = 0
    echo: bool = False
    stop_token_seqs: tuple[tuple[int, ...], ...] = ()


class FrontendResponseBridge:
    def __init__(self, response_queue: Any):
        self._response_queue = response_queue
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._pending: dict[str, asyncio.Queue[dict[str, Any]]] = {}

    def start(self):
        if self._thread is not None:
            return
        self._loop = asyncio.get_running_loop()
        self._thread = threading.Thread(
            target=self._run_loop,
            name="nanovllm-openai-frontend-response-bridge",
            daemon=True,
        )
        self._thread.start()

    def stop(self):
        thread = self._thread
        if thread is None:
            return
        self._stop.set()
        try:
            self._response_queue.put({"kind": "__frontend_stop__"})
        except Exception:
            pass
        thread.join(timeout=5.0)
        self._thread = None

    def register(self, request_id: str) -> asyncio.Queue[dict[str, Any]]:
        pending: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        with self._lock:
            self._pending[request_id] = pending
        return pending

    def unregister(self, request_id: str):
        with self._lock:
            self._pending.pop(request_id, None)

    def _dispatch(self, frame: dict[str, Any]):
        request_id = frame.get("request_id")
        if not isinstance(request_id, str):
            return
        with self._lock:
            queue = self._pending.get(request_id)
        if queue is not None:
            queue.put_nowait(frame)

    def _run_loop(self):
        assert self._loop is not None
        while not self._stop.is_set():
            try:
                frame = self._response_queue.get()
            except (EOFError, OSError):
                return
            if not isinstance(frame, dict):
                continue
            if frame.get("kind") == "__frontend_stop__":
                return
            self._loop.call_soon_threadsafe(self._dispatch, frame)


class PromptTokenCache:
    def __init__(self, max_entries: int = 4096):
        self._max_entries = max(1, int(max_entries))
        self._lock = threading.Lock()
        self._entries: OrderedDict[str, tuple[int, ...]] = OrderedDict()

    def encode(self, text: str, encode_fn) -> list[int]:
        with self._lock:
            cached = self._entries.get(text)
            if cached is not None:
                self._entries.move_to_end(text)
                return list(cached)
        encoded = tuple(int(token_id) for token_id in encode_fn(text))
        with self._lock:
            self._entries[text] = encoded
            self._entries.move_to_end(text)
            while len(self._entries) > self._max_entries:
                self._entries.popitem(last=False)
        return list(encoded)


@dataclass
class RequestResult:
    prompt_token_ids: list[int]
    completion_token_ids: list[int]
    text: str
    finish_reason: str
    ttft_s: float | None
    generation_s: float


@dataclass(slots=True)
class _ChatOutputFilter:
    parser: StreamingStringParser = field(default_factory=lambda: StreamingStringParser(tries=TRIE_THINK_NO_TRIGGER))
    content_restorer: StreamingMarkdownRestorer = field(default_factory=StreamingMarkdownRestorer)
    reasoning_restorer: StreamingMarkdownRestorer = field(default_factory=StreamingMarkdownRestorer)


def _assistant_message_payload(content: str, reasoning_content: str | None = None) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if reasoning_content:
        message["reasoning_content"] = reasoning_content
    return message


def _chat_stream_delta_payload(
    *,
    role: str | None = None,
    content: str | None = None,
    reasoning_content: str | None = None,
) -> dict[str, Any]:
    delta: dict[str, Any] = {}
    if role is not None:
        delta["role"] = role
    if content is not None:
        delta["content"] = content
    if reasoning_content is not None:
        delta["reasoning_content"] = reasoning_content
    return delta


def _filter_chat_segments(
    filter_state: _ChatOutputFilter,
    segments: list[tuple[str, str]],
) -> list[tuple[str, str]]:
    events: list[tuple[str, str]] = []
    for text, state in segments:
        if not text:
            continue
        if state == "reasoning_content":
            rendered = filter_state.reasoning_restorer.parse(text)
            if rendered:
                events.append(("reasoning_content", rendered))
            continue
        if state == "content":
            rendered = filter_state.content_restorer.parse(text)
            if rendered:
                events.append(("content", rendered))
    return events


def _filter_chat_delta(filter_state: _ChatOutputFilter, delta: str) -> list[tuple[str, str]]:
    if not delta:
        return []
    return _filter_chat_segments(filter_state, filter_state.parser.parse(delta))


def _flush_chat_delta_filter(filter_state: _ChatOutputFilter) -> list[tuple[str, str]]:
    events = _filter_chat_segments(filter_state, filter_state.parser.flush())
    reasoning_tail = filter_state.reasoning_restorer.flush()
    if reasoning_tail:
        events.append(("reasoning_content", reasoning_tail))
    content_tail = filter_state.content_restorer.flush()
    if content_tail:
        events.append(("content", content_tail))
    return events


def _filter_chat_text(text: str, *, mode: Literal["default", "thinking", "raw"] = "default") -> tuple[str, str | None]:
    filter_state = _make_chat_output_filter(mode)
    reasoning_parts: list[str] = []
    content_parts: list[str] = []
    for kind, rendered in _filter_chat_delta(filter_state, text):
        if kind == "reasoning_content":
            reasoning_parts.append(rendered)
        else:
            content_parts.append(rendered)
    for kind, rendered in _flush_chat_delta_filter(filter_state):
        if kind == "reasoning_content":
            reasoning_parts.append(rendered)
        else:
            content_parts.append(rendered)
    reasoning_text = "".join(reasoning_parts)
    return "".join(content_parts), reasoning_text or None

TRIE_RAW_NO_TRIGGER = StreamingStringParser.build_trie(
    [
        ("content", "\n\n", "end", "right"),
    ]
)

def _chat_output_mode_from_prompt(prompt_text: str | None) -> Literal["default", "thinking", "raw"]:
    normalized = (prompt_text or "").rstrip()
    if normalized.endswith("Assistant: <think"):
        return "thinking"
    if normalized.endswith("Assistant:"):
        return "raw"
    return "default"

def _make_chat_output_filter(mode: Literal["default", "thinking", "raw"] = "default") -> _ChatOutputFilter:
    if mode == "thinking":
        parser = StreamingStringParser(tries=TRIE_THINK_NO_TRIGGER, start_state="reasoning_content")
    elif mode == "raw":
        parser = StreamingStringParser(tries=TRIE_RAW_NO_TRIGGER)
    else:
        parser = StreamingStringParser(tries=TRIE_THINK_NO_TRIGGER)
    return _ChatOutputFilter(parser=parser)


def _apply_disable_cors(app: FastAPI, disable_cors: bool) -> None:
    if not disable_cors:
        return
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["*"],
    )


@dataclass
class TokenLogprobRecord:
    token_id: int
    logprob: float | None
    top_logprobs: list[tuple[int, float]] | None = None


class BatcherProfiler:
    def __init__(self, label: str = ""):
        now = time.perf_counter()
        self.label = label or "default"
        self.started_at = now
        self.pending_peak = 0
        self.active_peak = 0
        self.admitted_requests = 0
        self.tokenize_s = 0.0
        self.add_request_s = 0.0
        self.schedule_calls = {"decode": 0, "prefill": 0, "fallback": 0}
        self.schedule_hits = {"decode": 0, "prefill": 0, "fallback": 0}
        self.schedule_s = {"decode": 0.0, "prefill": 0.0, "fallback": 0.0}
        self.custom_check_calls = 0
        self.custom_check_hits = 0
        self.custom_check_s = 0.0
        self.state_observations = 0
        self.decode_ready_total = 0
        self.prefill_inflight_total = 0
        self.scheduler_waiting_total = 0
        self.prefill_stage_total = 0
        self.step_counts = {"decode": 0, "prefill": 0}
        self.step_seq_total = {"decode": 0, "prefill": 0}
        self.step_total_s = {"decode": 0.0, "prefill": 0.0}
        self.step_model_s = {"decode": 0.0, "prefill": 0.0}
        self.step_post_s = {"decode": 0.0, "prefill": 0.0}
        self.step_emit_s = {"decode": 0.0, "prefill": 0.0}

    def observe_depths(self, pending: int, active: int) -> None:
        self.pending_peak = max(self.pending_peak, pending)
        self.active_peak = max(self.active_peak, active)

    def observe_state(
        self,
        *,
        decode_ready: int,
        prefill_inflight: int,
        scheduler_waiting: int,
        prefill_stage: int,
    ) -> None:
        self.state_observations += 1
        self.decode_ready_total += decode_ready
        self.prefill_inflight_total += prefill_inflight
        self.scheduler_waiting_total += scheduler_waiting
        self.prefill_stage_total += prefill_stage

    def record_admission(self, *, tokenize_s: float, add_request_s: float, count: int = 1) -> None:
        self.admitted_requests += count
        self.tokenize_s += tokenize_s
        self.add_request_s += add_request_s

    def record_schedule(self, kind: str, duration_s: float, hit: bool) -> None:
        self.schedule_calls[kind] += 1
        self.schedule_s[kind] += duration_s
        if hit:
            self.schedule_hits[kind] += 1

    def record_custom_check(self, duration_s: float, required: bool) -> None:
        self.custom_check_calls += 1
        self.custom_check_s += duration_s
        if required:
            self.custom_check_hits += 1

    def record_step(
        self,
        *,
        kind: str,
        seq_count: int,
        total_s: float,
        model_s: float,
        post_s: float,
        emit_s: float,
    ) -> None:
        self.step_counts[kind] += 1
        self.step_seq_total[kind] += seq_count
        self.step_total_s[kind] += total_s
        self.step_model_s[kind] += model_s
        self.step_post_s[kind] += post_s
        self.step_emit_s[kind] += emit_s

    def emit_report(self) -> None:
        wall_s = max(1e-9, time.perf_counter() - self.started_at)

        def _avg_ms(total_s: float, count: int) -> float:
            if count <= 0:
                return 0.0
            return total_s * 1000.0 / count

        def _avg_us(total_s: float, count: int) -> float:
            if count <= 0:
                return 0.0
            return total_s * 1_000_000.0 / count

        def _avg_bsz(kind: str) -> float:
            count = self.step_counts[kind]
            if count <= 0:
                return 0.0
            return self.step_seq_total[kind] / count

        print(
            "[batcher-profile] "
            f"label={self.label} wall_s={wall_s:.3f} admitted={self.admitted_requests} "
            f"pending_peak={self.pending_peak} active_peak={self.active_peak}",
            flush=True,
        )
        if self.admitted_requests > 0:
            print(
                "[batcher-profile] "
                f"admission tokenize_ms_per_req={self.tokenize_s * 1000.0 / self.admitted_requests:.3f} "
                f"add_request_ms_per_req={self.add_request_s * 1000.0 / self.admitted_requests:.3f}",
                flush=True,
            )
        print(
            "[batcher-profile] "
            f"schedule_decode calls={self.schedule_calls['decode']} hits={self.schedule_hits['decode']} "
            f"avg_ms={_avg_ms(self.schedule_s['decode'], self.schedule_calls['decode']):.3f} "
            f"schedule_prefill calls={self.schedule_calls['prefill']} hits={self.schedule_hits['prefill']} "
            f"avg_ms={_avg_ms(self.schedule_s['prefill'], self.schedule_calls['prefill']):.3f} "
            f"schedule_fallback calls={self.schedule_calls['fallback']} hits={self.schedule_hits['fallback']} "
            f"avg_ms={_avg_ms(self.schedule_s['fallback'], self.schedule_calls['fallback']):.3f}",
            flush=True,
        )
        if self.state_observations > 0:
            print(
                "[batcher-profile] "
                f"state avg_decode_ready={self.decode_ready_total / self.state_observations:.2f} "
                f"avg_prefill_inflight={self.prefill_inflight_total / self.state_observations:.2f} "
                f"avg_scheduler_waiting={self.scheduler_waiting_total / self.state_observations:.2f} "
                f"avg_prefill_stage={self.prefill_stage_total / self.state_observations:.2f}",
                flush=True,
            )
        for kind in ("decode", "prefill"):
            count = self.step_counts[kind]
            other_s = self.step_total_s[kind] - self.step_model_s[kind] - self.step_post_s[kind] - self.step_emit_s[kind]
            print(
                "[batcher-profile] "
                f"{kind}_steps count={count} avg_bsz={_avg_bsz(kind):.2f} "
                f"total_ms_per_step={_avg_ms(self.step_total_s[kind], count):.3f} "
                f"model_ms_per_step={_avg_ms(self.step_model_s[kind], count):.3f} "
                f"post_ms_per_step={_avg_ms(self.step_post_s[kind], count):.3f} "
                f"emit_ms_per_step={_avg_ms(self.step_emit_s[kind], count):.3f} "
                f"other_ms_per_step={_avg_ms(other_s, count):.3f}",
                flush=True,
            )
        print(
            "[batcher-profile] "
            f"custom_check calls={self.custom_check_calls} hits={self.custom_check_hits} "
            f"avg_us={_avg_us(self.custom_check_s, self.custom_check_calls):.3f}",
            flush=True,
        )


@dataclass
class BatchedRequest:
    request_id: str
    endpoint: Literal["completion", "chat"]
    prompt_text: str | None
    sampling_params: SamplingParams
    requested_max_tokens: int
    prompt_token_ids_input: list[int] | None
    capture_logprobs: bool
    top_logprobs: int
    echo: bool
    created: int
    http_received_at: float
    handler_started_at: float
    http_started_at: float
    stream: bool
    prompt_token_ids: list[int] | None = None
    completion_token_ids: list[int] = field(default_factory=list)
    visible_text: str = ""
    seq_id: int | None = None
    first_scheduled_at: float | None = None
    first_token_at: float | None = None
    finished_at: float | None = None
    response_built_at: float | None = None
    finish_reason: str | None = None
    error: BaseException | None = None
    loop: asyncio.AbstractEventLoop | None = None
    ready_event: asyncio.Event | None = None
    done_event: asyncio.Event | None = None
    stream_queue: asyncio.Queue[tuple[str, str | None]] | None = None
    completion_notify: Any = None
    frontend_id: int | None = None
    prompt_scoring_token_ids: list[int] = field(default_factory=list)
    prompt_logprob_records: list[TokenLogprobRecord] = field(default_factory=list)
    completion_logprob_records: list[TokenLogprobRecord] = field(default_factory=list)
    cancelled: bool = False
    stop_token_seqs: tuple[tuple[int, ...], ...] = ()

    @property
    def ready_at(self) -> float | None:
        return self.first_token_at or self.finished_at

    @property
    def queue_wait_s(self) -> float:
        if self.first_scheduled_at is None:
            return 0.0
        return self.first_scheduled_at - self.http_started_at

    @property
    def ttft_s(self) -> float | None:
        if self.first_token_at is None or self.first_scheduled_at is None:
            return None
        return self.first_token_at - self.first_scheduled_at

    @property
    def generation_s(self) -> float | None:
        if self.finished_at is None or self.first_scheduled_at is None:
            return None
        return self.finished_at - self.first_scheduled_at

    @property
    def total_s(self) -> float | None:
        if self.finished_at is None:
            return None
        return self.finished_at - self.http_started_at

    @property
    def processing_s(self) -> float | None:
        ready_at = self.ready_at
        if ready_at is None:
            return None
        return ready_at - self.http_started_at

    @property
    def request_parse_s(self) -> float:
        return self.handler_started_at - self.http_received_at

    @property
    def request_setup_s(self) -> float:
        return self.http_started_at - self.handler_started_at

    @property
    def response_build_s(self) -> float | None:
        completed_at = self.finished_at or self.ready_at
        if self.response_built_at is None or completed_at is None:
            return None
        return self.response_built_at - completed_at

    @property
    def server_app_s(self) -> float | None:
        if self.response_built_at is None:
            return None
        return self.response_built_at - self.http_received_at

    def result(self) -> RequestResult:
        assert self.prompt_token_ids is not None
        assert self.finish_reason is not None
        return RequestResult(
            prompt_token_ids=list(self.prompt_token_ids),
            completion_token_ids=list(self.completion_token_ids),
            text=self.visible_text,
            finish_reason=self.finish_reason,
            ttft_s=self.ttft_s,
            generation_s=0.0 if self.generation_s is None else self.generation_s,
        )


class RequestBatcher:
    def __init__(self, llm: LLM):
        self.llm = llm
        self._cv = threading.Condition()
        self._thread_lock = threading.Lock()
        self._pending: deque[BatchedRequest] = deque()
        self._prefill_stage: deque[BatchedRequest] = deque()
        self._active: dict[int, BatchedRequest] = {}
        self._shutdown = False
        self._failure: BaseException | None = None
        self._thread: threading.Thread | None = None
        self._cold_start_batch_wait_s = 0.010
        self._prefill_admission_reserve_slots = self._default_prefill_admission_reserve_slots()
        self._prefill_admission_max_delay_s = 0.250
        self._max_prefill_inflight = self._default_max_prefill_inflight()
        self._prefill_stage_min_batch = self._default_prefill_stage_min_batch()
        self._prefill_stage_max_delay_s = self._default_prefill_stage_max_delay_s()
        self._decode_burst_steps = self._default_decode_burst_steps()
        self._decode_burst_min_ready = self._default_decode_burst_min_ready()
        self._decode_schedule_max_batch = self._default_decode_schedule_max_batch()
        self._prefill_waiting_max_delay_s = self._default_prefill_waiting_max_delay_s()
        self._multi_step_decode_tokens = self._default_multi_step_decode_tokens()
        self._decode_steps_since_prefill = 0
        self._profile = self._create_profiler()

    def _create_profiler(self) -> BatcherProfiler | None:
        raw = os.getenv("NANOVLLM_BATCHER_PROFILE", "")
        if raw.lower() in ("", "0", "false", "off", "no"):
            return None
        return BatcherProfiler(label=os.getenv("NANOVLLM_BATCHER_PROFILE_LABEL", ""))

    def start(self):
        with self._thread_lock:
            if self._thread is not None:
                return
            self._shutdown = False
            self._thread = threading.Thread(
                target=self._run_loop,
                name="nanovllm-openai-batcher",
                daemon=True,
            )
            self._thread.start()

    def stop(self):
        with self._thread_lock:
            thread = self._thread
            if thread is None:
                return
            with self._cv:
                self._shutdown = True
                self._cv.notify_all()
            thread.join(timeout=5.0)
            self._thread = None

    def submit(self, request: BatchedRequest):
        if self._thread is None:
            self.start()
        with self._cv:
            if self._failure is not None:
                raise RuntimeError("OpenAI API batch worker is unavailable.") from self._failure
            self._pending.append(request)
            self._cv.notify()

    def cancel(self, request: BatchedRequest):
        with self._cv:
            request.cancelled = True
            self._cv.notify()

    def _notify_ready(self, request: BatchedRequest):
        if request.ready_event is not None and not request.ready_event.is_set():
            request.ready_event.set()

    def _notify_done(self, request: BatchedRequest):
        if request.done_event is not None and not request.done_event.is_set():
            request.done_event.set()

    def _signal_ready(self, request: BatchedRequest):
        if request.ready_event is not None:
            self._call_on_loop(request, self._notify_ready, request)

    def _signal_done(self, request: BatchedRequest):
        if request.done_event is not None:
            self._call_on_loop(request, self._notify_done, request)

    def _push_stream_item(self, request: BatchedRequest, kind: str, value: str | None):
        if request.stream_queue is not None:
            request.stream_queue.put_nowait((kind, value))

    def _call_on_loop(self, request: BatchedRequest, fn, *args):
        if request.loop is None:
            fn(*args)
            return
        request.loop.call_soon_threadsafe(fn, *args)

    def _fail_request(self, request: BatchedRequest, exc: BaseException):
        request.error = exc
        if request.completion_notify is not None and not request.stream:
            request.completion_notify(request)
            return
        self._signal_ready(request)
        self._signal_done(request)
        self._call_on_loop(request, self._push_stream_item, request, "error", None)

    def _fail_all(self, exc: BaseException):
        with self._cv:
            pending = list(self._prefill_stage) + list(self._pending)
            self._prefill_stage.clear()
            self._pending.clear()
            active = list(self._active.values())
            self._active.clear()
            self._failure = exc
        for request in pending + active:
            self._fail_request(request, exc)

    def _complete_request_without_sequence(self, request: BatchedRequest) -> None:
        now = time.perf_counter()
        request.finished_at = now
        request.finish_reason = "stop"
        if request.completion_notify is not None and not request.stream:
            request.completion_notify(request)
            return
        self._signal_ready(request)
        self._signal_done(request)

    def _finalize_cancelled_request(self, request: BatchedRequest) -> None:
        if request.finished_at is None:
            request.finished_at = time.perf_counter()
        if request.finish_reason is None:
            request.finish_reason = "cancelled"
        self._signal_done(request)

    def _drop_cancelled_pending_requests(self) -> None:
        removed: list[BatchedRequest] = []
        if self._prefill_stage:
            kept = deque()
            while self._prefill_stage:
                request = self._prefill_stage.popleft()
                if request.cancelled:
                    removed.append(request)
                else:
                    kept.append(request)
            self._prefill_stage = kept
        if self._pending:
            kept = deque()
            while self._pending:
                request = self._pending.popleft()
                if request.cancelled:
                    removed.append(request)
                else:
                    kept.append(request)
            self._pending = kept
        for request in removed:
            self._finalize_cancelled_request(request)

    def _abort_cancelled_active_requests(self) -> None:
        cancelled = [
            (seq_id, request)
            for seq_id, request in list(self._active.items())
            if request.cancelled
        ]
        if not cancelled:
            return
        for seq_id, request in cancelled:
            abort = getattr(self.llm, "abort", None)
            if callable(abort):
                abort(seq_id)
            self._active.pop(seq_id, None)
            self._finalize_cancelled_request(request)

    def _admission_capacity(self) -> int:
        max_active = getattr(self.llm.scheduler, "max_num_seqs", 0)
        if not isinstance(max_active, int) or max_active <= 0:
            return self._pending_count()
        return max(0, max_active - len(self._active))

    def _pending_count(self) -> int:
        return len(self._prefill_stage) + len(self._pending)

    def _has_pending_requests(self) -> bool:
        return bool(self._prefill_stage or self._pending)

    def _stage_pending_requests(self) -> None:
        if not self._pending:
            return
        self._prefill_stage.extend(self._pending)
        self._pending.clear()

    def _rwkv_state_cache_enabled(self) -> bool:
        scheduler = getattr(self.llm, "scheduler", None)
        config = None if scheduler is None else getattr(scheduler, "config", None)
        if config is not None:
            return bool(getattr(config, "rwkv_state_cache_enable", False))
        if scheduler is not None and hasattr(scheduler, "rwkv_state_cache_enable"):
            return bool(getattr(scheduler, "rwkv_state_cache_enable"))
        llm_kwargs = getattr(self.llm, "llm_kwargs", None)
        if isinstance(llm_kwargs, dict):
            return bool(llm_kwargs.get("rwkv_state_cache_enable", False))
        return False

    def _default_prefill_admission_reserve_slots(self) -> int:
        max_active = getattr(self.llm.scheduler, "max_num_seqs", 0)
        if not isinstance(max_active, int) or max_active <= 64:
            return 0
        return min(32, max(8, max_active // 32))

    def _default_decode_burst_steps(self) -> int:
        env_value = os.getenv("NANOVLLM_DECODE_BURST_STEPS")
        if env_value is not None and env_value != "":
            try:
                parsed = int(env_value)
            except ValueError:
                parsed = -1
            if parsed >= 0:
                return parsed
        max_active = getattr(self.llm.scheduler, "max_num_seqs", 0)
        if not isinstance(max_active, int) or max_active <= 64:
            return 0
        return 4

    def _default_max_prefill_inflight(self) -> int:
        env_value = os.getenv("NANOVLLM_MAX_PREFILL_INFLIGHT")
        if env_value is not None and env_value != "":
            try:
                parsed = int(env_value)
            except ValueError:
                parsed = -1
            if parsed > 0:
                return parsed
        max_active = getattr(self.llm.scheduler, "max_num_seqs", 0)
        if not isinstance(max_active, int) or max_active <= 0:
            return 0
        if max_active <= 64:
            return max_active
        if self._rwkv_state_cache_enabled() or max_active <= 256:
            return min(max_active, 64)
        return min(max_active, max(64, min(256, max_active // 4)))

    def _default_prefill_stage_min_batch(self) -> int:
        env_value = os.getenv("NANOVLLM_PREFILL_STAGE_MIN_BATCH")
        if env_value is not None and env_value != "":
            try:
                parsed = int(env_value)
            except ValueError:
                parsed = -1
            if parsed > 0:
                return parsed
        max_active = getattr(self.llm.scheduler, "max_num_seqs", 0)
        if not isinstance(max_active, int) or max_active <= 64:
            return 1
        if max_active <= 256:
            return 16
        return 24

    def _default_prefill_stage_max_delay_s(self) -> float:
        env_value = os.getenv("NANOVLLM_PREFILL_STAGE_MAX_DELAY_S")
        if env_value is not None and env_value != "":
            try:
                parsed = float(env_value)
            except ValueError:
                parsed = -1.0
            if parsed > 0:
                return parsed
        return 0.200

    def _default_decode_burst_min_ready(self) -> int:
        env_value = os.getenv("NANOVLLM_DECODE_BURST_MIN_READY")
        if env_value is not None and env_value != "":
            try:
                parsed = int(env_value)
            except ValueError:
                parsed = -1
            if parsed >= 0:
                return parsed
        max_active = getattr(self.llm.scheduler, "max_num_seqs", 0)
        if not isinstance(max_active, int) or max_active <= 0:
            return 0
        return min(max_active, max(64, max_active // 4))

    def _default_prefill_waiting_max_delay_s(self) -> float:
        env_value = os.getenv("NANOVLLM_PREFILL_WAITING_MAX_DELAY_S")
        if env_value is not None and env_value != "":
            try:
                parsed = float(env_value)
            except ValueError:
                parsed = -1.0
            if parsed > 0:
                return parsed
        return 0.120

    def _default_decode_schedule_max_batch(self) -> int:
        env_value = os.getenv("NANOVLLM_DECODE_SCHEDULE_MAX_BATCH")
        if env_value is not None and env_value != "":
            try:
                parsed = int(env_value)
            except ValueError:
                parsed = -1
            if parsed > 0:
                return parsed
            if parsed == 0:
                return 0
        scheduler = getattr(self.llm, "scheduler", None)
        if self._rwkv_state_cache_enabled():
            max_active = getattr(scheduler, "max_num_seqs", 0)
            if isinstance(max_active, int) and max_active > 0:
                return min(max_active, 256)
        return 0

    def _default_multi_step_decode_tokens(self) -> int:
        env_value = os.getenv("NANOVLLM_MULTI_STEP_DECODE_TOKENS")
        if env_value is not None and env_value != "":
            try:
                parsed = int(env_value)
            except ValueError:
                parsed = -1
            if parsed > 1:
                return parsed
        max_active = getattr(self.llm.scheduler, "max_num_seqs", 0)
        if not isinstance(max_active, int) or max_active <= 64:
            return 1
        if max_active <= 256:
            return 4
        if max_active <= 512:
            return 2
        return 1

    def _oldest_pending_age_s(self, now: float) -> float:
        pending = self._prefill_stage[0] if self._prefill_stage else (self._pending[0] if self._pending else None)
        if pending is None:
            return 0.0
        return max(0.0, now - pending.http_started_at)

    def _decode_ready_count(self) -> int:
        scheduler = getattr(self.llm, "scheduler", None)
        if scheduler is None:
            return 0
        running = getattr(scheduler, "running", None)
        if not running:
            return 0
        count = 0
        for seq in running:
            if scheduler._prefill_step_tokens(seq) <= 0:
                count += 1
        return count

    def _prefill_inflight_count(self) -> int:
        scheduler = getattr(self.llm, "scheduler", None)
        if scheduler is None:
            return 0
        count = 0
        waiting = getattr(scheduler, "waiting", None)
        if waiting:
            count += len(waiting)
        running = getattr(scheduler, "running", None)
        if running:
            for seq in running:
                if scheduler._prefill_step_tokens(seq) > 0:
                    count += 1
        return count

    def _decode_supply_counts(self) -> tuple[int, int, int]:
        decode_ready = self._decode_ready_count()
        prefill_inflight = self._prefill_inflight_count()
        return decode_ready, prefill_inflight, decode_ready + prefill_inflight

    def _scheduler_waiting_count(self) -> int:
        scheduler = getattr(self.llm, "scheduler", None)
        if scheduler is None:
            return 0
        waiting = getattr(scheduler, "waiting", None)
        if not waiting:
            return 0
        return len(waiting)

    def _oldest_scheduler_waiting_age_s(self, now: float) -> float:
        scheduler = getattr(self.llm, "scheduler", None)
        if scheduler is None:
            return 0.0
        waiting = getattr(scheduler, "waiting", None)
        if not waiting:
            return 0.0
        oldest_started_at = None
        for seq in waiting:
            request = self._active.get(seq.seq_id)
            if request is None:
                continue
            if oldest_started_at is None or request.http_started_at < oldest_started_at:
                oldest_started_at = request.http_started_at
        if oldest_started_at is None:
            return 0.0
        return max(0.0, now - oldest_started_at)

    def _should_run_prefill_after_decode(self, now: float) -> bool:
        scheduler = getattr(self.llm, "scheduler", None)
        waiting = None if scheduler is None else getattr(scheduler, "waiting", None)
        if not waiting:
            return False
        if self._decode_burst_steps <= 0:
            return True
        oldest_waiting_age_s = self._oldest_scheduler_waiting_age_s(now)
        if oldest_waiting_age_s >= self._prefill_waiting_max_delay_s:
            return True
        decode_ready_count = self._decode_ready_count()
        if decode_ready_count < self._decode_burst_min_ready:
            return True
        if self._decode_steps_since_prefill >= self._decode_burst_steps:
            return True
        return False

    def _should_admit_pending_requests(self, now: float, admission_capacity: int) -> bool:
        if admission_capacity <= 0 or not self._has_pending_requests():
            return False
        scheduler = getattr(self.llm, "scheduler", None)
        waiting = None if scheduler is None else getattr(scheduler, "waiting", None)
        if (
            self._active
            and self._decode_burst_steps > 0
            and not waiting
            and self._decode_steps_since_prefill < self._decode_burst_steps
            and self._decode_ready_count() >= self._decode_burst_min_ready
            and self._oldest_pending_age_s(now) < self._prefill_stage_max_delay_s
        ):
            return False
        max_active = getattr(self.llm.scheduler, "max_num_seqs", 0)
        if not isinstance(max_active, int) or max_active <= 0:
            return True
        active_count = len(self._active)
        if active_count <= 0:
            return True
        reserve_slots = min(self._prefill_admission_reserve_slots, max_active)
        if reserve_slots <= 0:
            return True
        if active_count <= max(0, max_active - reserve_slots):
            return True
        if admission_capacity >= reserve_slots:
            return True
        return self._oldest_pending_age_s(now) >= self._prefill_admission_max_delay_s

    def _prefill_stage_target(self) -> int:
        target = self._prefill_stage_min_batch
        if target <= 1:
            return 1
        max_active = getattr(self.llm.scheduler, "max_num_seqs", 0)
        if not isinstance(max_active, int) or max_active <= 256:
            return target
        decode_ready_count = self._decode_ready_count()
        if self._decode_burst_min_ready <= 0 or decode_ready_count < self._decode_burst_min_ready:
            return target
        return max(target, min(64, max(target, decode_ready_count // 10)))

    def _pending_admission_quota(self, now: float, admission_capacity: int) -> int:
        if not self._should_admit_pending_requests(now, admission_capacity):
            return 0
        pending_count = self._pending_count()
        decode_ready_count, prefill_inflight, _decode_supply = self._decode_supply_counts()
        if self._max_prefill_inflight <= 0:
            quota = min(admission_capacity, pending_count)
        else:
            remaining_prefill_capacity = self._max_prefill_inflight - prefill_inflight
            if remaining_prefill_capacity <= 0:
                return 0
            quota = min(admission_capacity, remaining_prefill_capacity, pending_count)
        if quota <= 0:
            return 0
        if (
            quota < self._prefill_stage_target()
            and decode_ready_count >= self._decode_burst_min_ready
            and self._oldest_pending_age_s(now) < self._prefill_stage_max_delay_s
        ):
            return 0
        return quota

    def _pop_staged_requests(self, count: int) -> list[BatchedRequest]:
        if count <= 0:
            return []
        requests: list[BatchedRequest] = []
        while self._prefill_stage and len(requests) < count:
            requests.append(self._prefill_stage.popleft())
        while self._pending and len(requests) < count:
            requests.append(self._pending.popleft())
        return requests

    def _custom_token_selection_required(self) -> bool:
        return any(request.capture_logprobs or request.prompt_scoring_token_ids for request in self._active.values())

    def _can_multi_step_decode(self, seqs) -> bool:
        if self._multi_step_decode_tokens <= 1 or not seqs:
            return False
        if len(seqs) < 8:
            return False
        if self._has_pending_requests():
            return False
        if self._custom_token_selection_required():
            return False
        for seq in seqs:
            request = self._active.get(seq.seq_id)
            if request is None:
                continue
            if request.stream or request.capture_logprobs or request.prompt_scoring_token_ids:
                return False
        return True

    def _can_emit_token(self, seq: Any, is_prefill: bool) -> bool:
        if not is_prefill:
            return True
        scheduler_config = getattr(self.llm.scheduler, "config", None)
        chunk_size = -1 if scheduler_config is None else scheduler_config.rwkv_prefill_chunk_size
        step_tokens = seq.prefill_step_tokens(chunk_size)
        return seq.num_cached_tokens + step_tokens >= seq.num_prompt_tokens

    def _record_logprob(self, request: BatchedRequest, token_id: int, logits_row: torch.Tensor, *, prompt_token: bool) -> None:
        if not request.capture_logprobs:
            return
        log_probs = torch.log_softmax(logits_row.float(), dim=-1)
        top_entries = None
        if request.top_logprobs > 0:
            k = min(request.top_logprobs, int(log_probs.numel()))
            values, indices = torch.topk(log_probs, k)
            top_entries = [
                (int(index.item()), float(value.item()))
                for value, index in zip(values, indices, strict=True)
            ]
        record = TokenLogprobRecord(
            token_id=token_id,
            logprob=float(log_probs[token_id].item()),
            top_logprobs=top_entries,
        )
        if prompt_token:
            request.prompt_logprob_records.append(record)
        else:
            request.completion_logprob_records.append(record)

    def _run_custom_step(self, seqs, is_prefill: bool) -> None:
        step_started_at = time.perf_counter()
        for seq in seqs:
            request = self._active.get(seq.seq_id)
            if request is not None and request.first_scheduled_at is None:
                request.first_scheduled_at = step_started_at
        model_started_at = time.perf_counter()
        logits = self.llm.model_runner.call("run_logits", seqs, is_prefill)
        model_finished_at = time.perf_counter()
        token_ids: list[int | None] = [None] * len(seqs)
        emit_kinds: list[str | None] = [None] * len(seqs)
        sample_indices: list[int] = []
        sample_seqs = []
        restore_ignore_eos: list[tuple[Any, BatchedRequest]] = []
        for index, seq in enumerate(seqs):
            request = self._active.get(seq.seq_id)
            if request is None or not self._can_emit_token(seq, is_prefill):
                continue
            if getattr(seq, "pending_hidden_finalize", False):
                continue
            if request.prompt_scoring_token_ids:
                token_id = request.prompt_scoring_token_ids.pop(0)
                token_ids[index] = token_id
                emit_kinds[index] = "prompt"
                self._record_logprob(request, token_id, logits[index], prompt_token=True)
                if not request.prompt_scoring_token_ids:
                    restore_ignore_eos.append((seq, request))
                continue
            sample_indices.append(index)
            sample_seqs.append(seq)
        if sample_indices:
            index_tensor = torch.tensor(sample_indices, dtype=torch.int64, device=logits.device)
            sample_logits = logits.index_select(0, index_tensor) if len(sample_indices) != len(seqs) else logits
            sampled_tokens = self.llm.model_runner.sampler(sample_logits, sample_seqs).tolist()
            for row, token_id in zip(sample_indices, sampled_tokens, strict=True):
                token_ids[row] = int(token_id)
                emit_kinds[row] = "completion"
                request = self._active.get(seqs[row].seq_id)
                if request is not None:
                    self._record_logprob(request, int(token_id), logits[row], prompt_token=False)
        post_started_at = time.perf_counter()
        self.llm.model_runner.call("prepare_postprocess", seqs, token_ids)
        self.llm.scheduler.postprocess(seqs, token_ids)
        post_finished_at = time.perf_counter()
        for seq, request in restore_ignore_eos:
            if not seq.is_finished:
                seq.ignore_eos = request.sampling_params.ignore_eos
        emit_started_at = post_finished_at
        step_finished_at = time.perf_counter()
        for seq, token_id, emit_kind in zip(seqs, token_ids, emit_kinds, strict=True):
            request = self._active.get(seq.seq_id)
            if request is None:
                continue
            suppress_output_token = bool(getattr(seq, "last_token_hidden_from_output", False))
            if token_id is not None and emit_kind == "completion":
                if suppress_output_token:
                    if request.completion_logprob_records:
                        request.completion_logprob_records.pop()
                else:
                    request.completion_token_ids.append(token_id)
                    if request.first_token_at is None:
                        request.first_token_at = step_finished_at
                        self._signal_ready(request)
                    if request.stream:
                        new_visible_text = _decode_visible_text(self.llm.tokenizer, request.completion_token_ids)
                        delta = (
                            new_visible_text[len(request.visible_text):]
                            if new_visible_text.startswith(request.visible_text)
                            else new_visible_text
                        )
                        request.visible_text = new_visible_text
                        if delta:
                            self._call_on_loop(request, self._push_stream_item, request, "delta", delta)
            if seq.is_finished:
                request.finished_at = step_finished_at
                request.finish_reason = (
                    "length"
                    if request.requested_max_tokens > 0 and len(request.completion_token_ids) >= request.requested_max_tokens
                    else "stop"
                )
                if not request.stream:
                    request.visible_text = _decode_visible_text(
                        self.llm.tokenizer,
                        request.completion_token_ids,
                    )
                if request.completion_notify is not None and not request.stream:
                    request.completion_notify(request)
                    self._active.pop(seq.seq_id, None)
                    continue
                self._signal_ready(request)
                self._signal_done(request)
                if request.stream:
                    self._call_on_loop(
                        request,
                        self._push_stream_item,
                        request,
                        "finish",
                        request.finish_reason,
                    )
                    self._call_on_loop(request, self._push_stream_item, request, "done", None)
                self._active.pop(seq.seq_id, None)
        if self._profile is not None:
            self._profile.record_step(
                kind="prefill" if is_prefill else "decode",
                seq_count=len(seqs),
                total_s=step_finished_at - step_started_at,
                model_s=model_finished_at - model_started_at,
                post_s=post_finished_at - post_started_at,
                emit_s=step_finished_at - emit_started_at,
            )

    def _run_standard_step(self, seqs, is_prefill: bool) -> None:
        step_started_at = time.perf_counter()
        for seq in seqs:
            request = self._active.get(seq.seq_id)
            if request is not None and request.first_scheduled_at is None:
                request.first_scheduled_at = step_started_at
        model_started_at = time.perf_counter()
        token_ids = self.llm.model_runner.call("run", seqs, is_prefill)
        model_finished_at = time.perf_counter()
        post_started_at = time.perf_counter()
        self.llm.scheduler.postprocess(seqs, token_ids)
        post_finished_at = time.perf_counter()
        emit_started_at = post_finished_at
        step_finished_at = time.perf_counter()
        for seq, token_id in zip(seqs, token_ids):
            request = self._active.get(seq.seq_id)
            if request is None:
                continue
            if token_id is not None:
                suppress_output_token = bool(getattr(seq, "last_token_hidden_from_output", False))
                if not suppress_output_token:
                    request.completion_token_ids.append(token_id)
                    if request.first_token_at is None:
                        request.first_token_at = step_finished_at
                        self._signal_ready(request)
                    if request.stream:
                        new_visible_text = _decode_visible_text(self.llm.tokenizer, request.completion_token_ids)
                        delta = (
                            new_visible_text[len(request.visible_text):]
                            if new_visible_text.startswith(request.visible_text)
                            else new_visible_text
                        )
                        request.visible_text = new_visible_text
                        if delta:
                            self._call_on_loop(request, self._push_stream_item, request, "delta", delta)
            if seq.is_finished:
                request.finished_at = step_finished_at
                request.finish_reason = _finish_reason(
                    len(request.completion_token_ids),
                    request.sampling_params.max_tokens,
                )
                if not request.stream:
                    request.visible_text = _decode_visible_text(
                        self.llm.tokenizer,
                        request.completion_token_ids,
                    )
                if request.completion_notify is not None and not request.stream:
                    request.completion_notify(request)
                    self._active.pop(seq.seq_id, None)
                    continue
                self._signal_ready(request)
                self._signal_done(request)
                if request.stream:
                    self._call_on_loop(
                        request,
                        self._push_stream_item,
                        request,
                        "finish",
                        request.finish_reason,
                    )
                    self._call_on_loop(request, self._push_stream_item, request, "done", None)
                self._active.pop(seq.seq_id, None)
        if self._profile is not None:
            self._profile.record_step(
                kind="prefill" if is_prefill else "decode",
                seq_count=len(seqs),
                total_s=step_finished_at - step_started_at,
                model_s=model_finished_at - model_started_at,
                post_s=post_finished_at - post_started_at,
                emit_s=step_finished_at - emit_started_at,
            )

    def _run_scheduled_step(self, seqs, is_prefill: bool) -> None:
        custom_check_started_at = time.perf_counter()
        requires_custom_token_selection = self._custom_token_selection_required()
        custom_check_finished_at = time.perf_counter()
        if self._profile is not None:
            self._profile.record_custom_check(
                duration_s=custom_check_finished_at - custom_check_started_at,
                required=requires_custom_token_selection,
            )
        if requires_custom_token_selection:
            self._run_custom_step(seqs, is_prefill)
        else:
            self._run_standard_step(seqs, is_prefill)

    def _schedule_split_batch(self, *, is_prefill: bool):
        method_name = "schedule_prefill_only" if is_prefill else "schedule_decode_only"
        schedule_fn = getattr(self.llm.scheduler, method_name, None)
        if schedule_fn is None:
            return None
        schedule_started_at = time.perf_counter()
        seqs = schedule_fn()
        schedule_finished_at = time.perf_counter()
        if self._profile is not None:
            self._profile.record_schedule(
                "prefill" if is_prefill else "decode",
                schedule_finished_at - schedule_started_at,
                bool(seqs),
            )
        if not seqs:
            return None
        if not is_prefill and self._decode_schedule_max_batch > 0 and len(seqs) > self._decode_schedule_max_batch:
            seqs = seqs[: self._decode_schedule_max_batch]
        return seqs

    def _run_split_steps(self) -> bool:
        ran_step = False
        if not self._active:
            return False
        decode_seqs = self._schedule_split_batch(is_prefill=False)
        if decode_seqs:
            remaining_decode_seqs = decode_seqs
            decode_steps_run = 0
            while remaining_decode_seqs:
                self._run_scheduled_step(remaining_decode_seqs, False)
                self._decode_steps_since_prefill += 1
                decode_steps_run += 1
                ran_step = True
                if decode_steps_run >= self._multi_step_decode_tokens:
                    break
                if not self._can_multi_step_decode(remaining_decode_seqs):
                    break
                remaining_decode_seqs = [seq for seq in remaining_decode_seqs if not seq.is_finished]
                if not remaining_decode_seqs:
                    break
            ran_step = True
            if not self._should_run_prefill_after_decode(time.perf_counter()):
                return True
        if not self._active:
            return ran_step
        prefill_seqs = self._schedule_split_batch(is_prefill=True)
        if prefill_seqs:
            self._run_scheduled_step(prefill_seqs, True)
            self._decode_steps_since_prefill = 0
            ran_step = True
        return ran_step

    def _run_loop(self):
        try:
            while True:
                with self._cv:
                    while not self._shutdown and self._failure is None and not self._has_pending_requests() and not self._active:
                        self._cv.wait()
                    if self._shutdown:
                        return
                    if self._has_pending_requests() and not self._active and self._cold_start_batch_wait_s > 0:
                        deadline = time.perf_counter() + self._cold_start_batch_wait_s
                        while not self._shutdown and self._failure is None and not self._active:
                            remaining = deadline - time.perf_counter()
                            if remaining <= 0:
                                break
                            self._cv.wait(timeout=remaining)
                        if self._shutdown:
                            return
                    self._stage_pending_requests()
                    self._drop_cancelled_pending_requests()
                    self._abort_cancelled_active_requests()
                    new_requests = []
                    admission_capacity = self._admission_capacity()
                    now = time.perf_counter()
                    if self._profile is not None:
                        self._profile.observe_depths(self._pending_count(), len(self._active))
                        self._profile.observe_state(
                            decode_ready=self._decode_ready_count(),
                            prefill_inflight=self._prefill_inflight_count(),
                            scheduler_waiting=self._scheduler_waiting_count(),
                            prefill_stage=len(self._prefill_stage),
                        )
                    admission_quota = self._pending_admission_quota(now, admission_capacity)
                    if admission_quota > 0:
                        new_requests = self._pop_staged_requests(admission_quota)

                for request in new_requests:
                    if request.cancelled:
                        self._finalize_cancelled_request(request)
                        continue
                    tokenize_s = 0.0
                    add_request_s = 0.0
                    if request.prompt_token_ids_input is not None:
                        prompt_token_ids = list(request.prompt_token_ids_input)
                    else:
                        assert request.prompt_text is not None
                        tokenize_started_at = time.perf_counter()
                        prompt_token_ids = self.llm.tokenizer.encode(request.prompt_text)
                        tokenize_s = time.perf_counter() - tokenize_started_at
                    if not prompt_token_ids:
                        self._fail_request(
                            request,
                            OpenAIAPIError(400, "Prompt must not tokenize to an empty input.", param="prompt"),
                        )
                        continue
                    request.prompt_token_ids = prompt_token_ids
                    if request.capture_logprobs and request.echo:
                        request.prompt_scoring_token_ids = prompt_token_ids[1:]
                    if request.requested_max_tokens == 0 and not request.prompt_scoring_token_ids:
                        self._complete_request_without_sequence(request)
                        continue
                    engine_prompt_token_ids = prompt_token_ids[:1] if request.prompt_scoring_token_ids else prompt_token_ids
                    engine_max_tokens = len(request.prompt_scoring_token_ids) + request.requested_max_tokens
                    engine_sampling_params = request.sampling_params
                    if engine_sampling_params.max_tokens != engine_max_tokens:
                        engine_sampling_params = replace(engine_sampling_params, max_tokens=engine_max_tokens)
                    add_request_started_at = time.perf_counter()
                    seq = self.llm.add_request(engine_prompt_token_ids, engine_sampling_params)
                    add_request_s = time.perf_counter() - add_request_started_at
                    if request.prompt_scoring_token_ids:
                        seq.ignore_eos = True
                    if request.stop_token_seqs:
                        seq.stop_token_seqs = request.stop_token_seqs
                    request.seq_id = seq.seq_id
                    self._active[seq.seq_id] = request
                    if self._profile is not None:
                        self._profile.record_admission(
                            tokenize_s=tokenize_s,
                            add_request_s=add_request_s,
                        )
                        self._profile.observe_depths(self._pending_count(), len(self._active))
                        self._profile.observe_state(
                            decode_ready=self._decode_ready_count(),
                            prefill_inflight=self._prefill_inflight_count(),
                            scheduler_waiting=self._scheduler_waiting_count(),
                            prefill_stage=len(self._prefill_stage),
                        )

                if not self._active:
                    continue

                if self._run_split_steps():
                    continue

                schedule_started_at = time.perf_counter()
                seqs, is_prefill = self.llm.scheduler.schedule()
                schedule_finished_at = time.perf_counter()
                if self._profile is not None:
                    self._profile.record_schedule("fallback", schedule_finished_at - schedule_started_at, bool(seqs))
                self._run_scheduled_step(seqs, is_prefill)
        except BaseException as exc:  # pragma: no cover
            self._fail_all(exc)
        finally:
            if self._profile is not None:
                self._profile.emit_report()


def _openai_error_response(exc: OpenAIAPIError):
    return _json_response(
        {
            "error": {
                "message": exc.message,
                "type": exc.error_type,
                "param": exc.param,
                "code": exc.code,
            }
        },
        status_code=exc.status_code,
    )


def _serialize_openai_error(exc: OpenAIAPIError) -> dict[str, Any]:
    return {
        "status_code": exc.status_code,
        "message": exc.message,
        "error_type": exc.error_type,
        "param": exc.param,
        "code": exc.code,
    }


def _deserialize_openai_error(payload: dict[str, Any]) -> OpenAIAPIError:
    return OpenAIAPIError(
        int(payload["status_code"]),
        str(payload["message"]),
        error_type=str(payload.get("error_type") or "invalid_request_error"),
        param=payload.get("param"),
        code=payload.get("code"),
    )


def _coerce_text_content(content: str | list[TextPart] | None) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    chunks: list[str] = []
    for part in content:
        part_type = part.type if isinstance(part, TextPart) else part.get("type")
        part_text = part.text if isinstance(part, TextPart) else part.get("text")
        if part_type != "text":
            raise OpenAIAPIError(
                400,
                f"Only text content parts are supported in this server. Got content part type={part_type!r}.",
                param="messages",
            )
        chunks.append(part_text or "")
    return "".join(chunks)


def _render_chat_prompt(tokenizer, messages: list[ChatMessage], *, add_generation_prompt: bool = True) -> str:
    normalized_messages = []
    for msg in messages:
        role_value = msg.role if isinstance(msg, ChatMessage) else msg.get("role")
        content_value = msg.content if isinstance(msg, ChatMessage) else msg.get("content")
        role = "system" if role_value == "developer" else role_value
        normalized_messages.append(
            {
                "role": role,
                "content": _coerce_text_content(content_value),
            }
        )

    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                normalized_messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )
        except Exception:
            pass

    lines: list[str] = []
    role_names = {
        "system": "System",
        "user": "User",
        "assistant": "Assistant",
        "tool": "Tool",
    }
    for msg in normalized_messages:
        lines.append(
            RWKVTokenizer.format_role_line(
                role_names.get(msg["role"], msg["role"].title()),
                msg["content"],
            )
        )
    if add_generation_prompt and (not normalized_messages or normalized_messages[-1]["role"] != "assistant"):
        lines.append(RWKVTokenizer.format_role_line("Assistant"))
    return "\n".join(lines)


def _parse_openai_model_mode(request_model: str) -> tuple[str, Literal["default", "thinking", "raw"]]:
    base_model, sep, suffix = request_model.partition(":")
    if not sep:
        return request_model, "default"
    suffix_lower = suffix.lower()
    if "raw" in suffix_lower:
        return base_model, "raw"
    if "thinking" in suffix_lower:
        return base_model, "thinking"
    return base_model, "default"


def _validate_openai_request_model(
    request_model: str,
    state: Any,
) -> tuple[str, Literal["default", "thinking", "raw"]]:
    base_model, mode = _parse_openai_model_mode(request_model)
    _validate_model(base_model, state)
    return base_model, mode


def _openai_model_ids(model_id: str) -> list[str]:
    return [model_id, f"{model_id}:thinking", f"{model_id}:raw"]


def _openai_model_card(model_id: str, created: int) -> dict[str, Any]:
    return {
        "id": model_id,
        "object": "model",
        "created": created,
        "owned_by": "nano-vllm",
    }


def _openai_models_response(state: Any) -> dict[str, Any]:
    return {
        "object": "list",
        "data": [_openai_model_card(model_id, state.created) for model_id in _openai_model_ids(state.model_id)],
    }


def _retrieve_openai_model_response(state: Any, request_model: str) -> dict[str, Any]:
    if request_model not in _openai_model_ids(state.model_id):
        raise OpenAIAPIError(
            404,
            f"Model {request_model!r} not found. This server is serving {state.model_id!r}.",
            error_type="invalid_request_error",
            param="model",
            code="model_not_found",
        )
    return _openai_model_card(request_model, state.created)


def _render_openai_chat_prompt(
    tokenizer,
    messages: list[ChatMessage],
    *,
    mode: Literal["default", "thinking", "raw"],
) -> str:
    prompt_text = _render_chat_prompt(tokenizer, messages, add_generation_prompt=False).strip()
    prefix = f"{prompt_text}\n\n" if prompt_text else ""
    if mode == "raw":
        return f"{prefix}Assistant:"
    if mode == "thinking":
        return f"{prefix}Assistant: <think"
    return f"{prefix}Assistant: <think>\n</think>\n"


def _validate_model(request_model: str, state: ServerState):
    if request_model != state.model_id:
        raise OpenAIAPIError(
            404,
            f"Model {request_model!r} not found. This server is serving {state.model_id!r}.",
            error_type="invalid_request_error",
            param="model",
            code="model_not_found",
        )


def _reject_extra_fields(req: BaseModel):
    model_extra = getattr(req, "model_extra", None)
    if not model_extra:
        return
    unsupported = [key for key, value in model_extra.items() if value is not None]
    if unsupported:
        raise OpenAIAPIError(
            400,
            f"Unsupported request field(s): {', '.join(sorted(unsupported))}.",
        )


def _validate_common_controls(
    *,
    n: int | None,
    top_p: float | None,
    stop: str | list[str] | None,
    presence_penalty: float | None,
    frequency_penalty: float | None,
    seed: int | None,
):
    if n not in (None, 1):
        raise OpenAIAPIError(400, "Only n=1 is supported.", param="n")
    if top_p is not None and not (0.0 <= top_p <= 1.0):
        raise OpenAIAPIError(400, "top_p must be in [0, 1].", param="top_p")
    if stop not in (None, [], ""):
        raise OpenAIAPIError(400, "Stop sequences are not supported yet.", param="stop")
    if seed is not None:
        raise OpenAIAPIError(400, "seed is not supported yet.", param="seed")


def _validate_openai_penalty(
    value: float | None,
    field_name: str,
    *,
    min_value: float,
    max_value: float,
) -> float:
    if value is None:
        return 0.0
    if not math.isfinite(value):
        raise OpenAIAPIError(400, f"{field_name} must be finite.", param=field_name)
    if not (min_value <= value <= max_value):
        raise OpenAIAPIError(
            400,
            f"{field_name} must be in [{min_value:g}, {max_value:g}].",
            param=field_name,
        )
    return float(value)


def _validate_penalty_decay(value: float | None) -> float:
    if value is None:
        return DEFAULT_OPENAI_PENALTY_DECAY
    if not math.isfinite(value):
        raise OpenAIAPIError(400, "penalty_decay must be finite.", param="penalty_decay")
    if not (0.0 <= value <= 1.0):
        raise OpenAIAPIError(400, "penalty_decay must be in [0, 1].", param="penalty_decay")
    return float(value)


def _stream_options_include_usage(stream_options: dict[str, Any] | None) -> bool:
    if stream_options is None:
        return False
    if not isinstance(stream_options, dict):
        raise OpenAIAPIError(400, "stream_options must be an object.", param="stream_options")
    unsupported = [key for key, value in stream_options.items() if key != "include_usage" and value is not None]
    if unsupported:
        raise OpenAIAPIError(
            400,
            f"Unsupported stream_options field(s): {', '.join(sorted(unsupported))}.",
            param="stream_options",
        )
    include_usage = stream_options.get("include_usage")
    if include_usage is None:
        return False
    if not isinstance(include_usage, bool):
        raise OpenAIAPIError(400, "stream_options.include_usage must be a boolean.", param="stream_options")
    return include_usage


def _validated_sampling_params(
    *,
    temperature: float | None,
    top_p: float | None,
    max_tokens: int | None,
    presence_penalty: float | None = None,
    frequency_penalty: float | None = None,
    penalty_decay: float | None = None,
    allow_zero_max_tokens: bool = False,
) -> SamplingParams:
    use_temperature = 1.0 if temperature is None else temperature
    use_top_p = 1.0 if top_p is None else top_p
    use_max_tokens = DEFAULT_OPENAI_MAX_TOKENS if max_tokens is None else max_tokens
    use_presence_penalty = _validate_openai_penalty(
        presence_penalty,
        "presence_penalty",
        min_value=0.0,
        max_value=2.0,
    )
    use_frequency_penalty = _validate_openai_penalty(
        frequency_penalty,
        "frequency_penalty",
        min_value=0.0,
        max_value=1.0,
    )
    use_penalty_decay = _validate_penalty_decay(penalty_decay)
    if use_temperature < 0:
        raise OpenAIAPIError(400, "temperature must be non-negative.", param="temperature")
    if not (0.0 <= use_top_p <= 1.0):
        raise OpenAIAPIError(400, "top_p must be in [0, 1].", param="top_p")
    if use_max_tokens < 0 or (use_max_tokens == 0 and not allow_zero_max_tokens):
        raise OpenAIAPIError(400, "max_tokens must be positive.", param="max_tokens")
    engine_max_tokens = max(1, use_max_tokens)
    return SamplingParams(
        temperature=use_temperature,
        top_p=use_top_p,
        presence_penalty=use_presence_penalty,
        repetition_penalty=use_frequency_penalty,
        penalty_decay=use_penalty_decay,
        max_tokens=engine_max_tokens,
    )


def _sampling_params_to_payload(sampling_params: SamplingParams) -> dict[str, Any]:
    return {
        "temperature": float(sampling_params.temperature),
        "top_p": float(sampling_params.top_p),
        "presence_penalty": float(sampling_params.presence_penalty),
        "repetition_penalty": float(sampling_params.repetition_penalty),
        "penalty_decay": float(sampling_params.penalty_decay),
        "max_tokens": int(sampling_params.max_tokens),
    }


def _sampling_params_from_payload(payload: dict[str, Any]) -> SamplingParams:
    return SamplingParams(
        temperature=float(payload["temperature"]),
        top_p=float(payload["top_p"]),
        presence_penalty=float(payload.get("presence_penalty", 0.0)),
        repetition_penalty=float(payload.get("repetition_penalty", 0.0)),
        penalty_decay=float(payload.get("penalty_decay", DEFAULT_OPENAI_PENALTY_DECAY)),
        max_tokens=int(payload["max_tokens"]),
    )


def _state_tokenizer(state: Any):
    tokenizer = getattr(state, "tokenizer", None)
    if tokenizer is not None:
        return tokenizer
    llm = getattr(state, "llm", None)
    if llm is not None:
        return llm.tokenizer
    raise RuntimeError("State does not provide a tokenizer.")


def _state_encode_text(state: Any, text: str) -> list[int]:
    cache = getattr(state, "prompt_token_cache", None)
    tokenizer = _state_tokenizer(state)
    if cache is None:
        return tokenizer.encode(text)
    return cache.encode(text, tokenizer.encode)


def _serialize_prepared_request(prepared: PreparedOpenAIRequest) -> dict[str, Any]:
    return {
        "prompt_text": prepared.prompt_text,
        "prompt_token_ids": None if prepared.prompt_token_ids is None else list(prepared.prompt_token_ids),
        "requested_max_tokens": int(prepared.requested_max_tokens),
        "sampling": _sampling_params_to_payload(prepared.sampling_params),
        "capture_logprobs": bool(prepared.capture_logprobs),
        "top_logprobs": int(prepared.top_logprobs),
        "echo": bool(prepared.echo),
        "stop_token_seqs": [list(seq) for seq in prepared.stop_token_seqs],
    }


def _deserialize_prepared_request(payload: dict[str, Any]) -> PreparedOpenAIRequest:
    return PreparedOpenAIRequest(
        prompt_text=payload.get("prompt_text"),
        sampling_params=_sampling_params_from_payload(payload["sampling"]),
        requested_max_tokens=int(payload["requested_max_tokens"]),
        prompt_token_ids=payload.get("prompt_token_ids"),
        capture_logprobs=bool(payload.get("capture_logprobs", False)),
        top_logprobs=int(payload.get("top_logprobs", 0)),
        echo=bool(payload.get("echo", False)),
        stop_token_seqs=tuple(tuple(int(token_id) for token_id in seq) for seq in payload.get("stop_token_seqs", ())),
    )


def _completion_logprob_count(logprobs: int | bool | None) -> int:
    if logprobs in (None, False, 0):
        return 0
    if logprobs is True or isinstance(logprobs, bool) or not isinstance(logprobs, int) or logprobs < 0:
        raise OpenAIAPIError(400, "logprobs must be a non-negative integer.", param="logprobs")
    return int(logprobs)


def _sampling_params_from_completion(req: CompletionRequest) -> SamplingParams:
    _reject_extra_fields(req)
    _validate_common_controls(
        n=req.n,
        top_p=req.top_p,
        stop=req.stop,
        presence_penalty=req.presence_penalty,
        frequency_penalty=req.frequency_penalty,
        seed=req.seed,
    )
    _stream_options_include_usage(req.stream_options)
    logprob_count = _completion_logprob_count(req.logprobs)
    if req.stream and logprob_count > 0:
        raise OpenAIAPIError(400, "logprobs is not supported for streaming completions.", param="logprobs")
    if req.stream and req.echo:
        raise OpenAIAPIError(400, "echo is not supported for streaming completions.", param="echo")
    if req.max_tokens == 0 and req.echo is not True:
        raise OpenAIAPIError(
            400,
            "max_tokens=0 is only supported together with echo=true.",
            param="max_tokens",
        )
    return _validated_sampling_params(
        temperature=req.temperature,
        top_p=req.top_p,
        max_tokens=req.max_tokens,
        presence_penalty=req.presence_penalty,
        frequency_penalty=req.frequency_penalty,
        penalty_decay=req.penalty_decay,
        allow_zero_max_tokens=bool(req.echo),
    )


def _sampling_params_from_chat(req: ChatCompletionRequest) -> SamplingParams:
    _reject_extra_fields(req)
    _validate_common_controls(
        n=req.n,
        top_p=req.top_p,
        stop=req.stop,
        presence_penalty=req.presence_penalty,
        frequency_penalty=req.frequency_penalty,
        seed=req.seed,
    )
    _stream_options_include_usage(req.stream_options)
    if req.logprobs not in (None, False):
        raise OpenAIAPIError(400, "logprobs is not supported.", param="logprobs")
    if req.top_logprobs not in (None, 0):
        raise OpenAIAPIError(400, "top_logprobs is not supported.", param="top_logprobs")
    if req.tools not in (None, []):
        raise OpenAIAPIError(400, "tools are not supported yet.", param="tools")
    if req.tool_choice is not None:
        raise OpenAIAPIError(400, "tool_choice is not supported yet.", param="tool_choice")
    if req.parallel_tool_calls is not None:
        raise OpenAIAPIError(400, "parallel_tool_calls is not supported yet.", param="parallel_tool_calls")
    if req.response_format is not None:
        raise OpenAIAPIError(400, "response_format is not supported yet.", param="response_format")
    if not req.messages:
        raise OpenAIAPIError(400, "messages must not be empty.", param="messages")
    if req.max_completion_tokens is not None and req.max_tokens is not None and req.max_completion_tokens != req.max_tokens:
        raise OpenAIAPIError(
            400,
            "Provide either max_tokens or max_completion_tokens, or set them to the same value.",
            param="max_completion_tokens",
        )
    max_tokens = req.max_completion_tokens if req.max_completion_tokens is not None else req.max_tokens
    return _validated_sampling_params(
        temperature=req.temperature,
        top_p=req.top_p,
        max_tokens=max_tokens,
        presence_penalty=req.presence_penalty,
        frequency_penalty=req.frequency_penalty,
        penalty_decay=req.penalty_decay,
    )


def _normalize_prompt(prompt: str | list[str]) -> str:
    if isinstance(prompt, str):
        return prompt
    if len(prompt) != 1:
        raise OpenAIAPIError(400, "Only a single prompt is supported.", param="prompt")
    return prompt[0]


def _normalize_prompt_token_ids(prompt_token_ids: list[int]) -> list[int]:
    if not prompt_token_ids:
        raise OpenAIAPIError(400, "prompt_token_ids must not be empty.", param="prompt_token_ids")
    normalized: list[int] = []
    for token_id in prompt_token_ids:
        if isinstance(token_id, bool) or not isinstance(token_id, int):
            raise OpenAIAPIError(400, "prompt_token_ids must be a list of integers.", param="prompt_token_ids")
        normalized.append(int(token_id))
    return normalized


def _resolve_completion_prompt(
    prompt: str | list[str] | None,
    prompt_token_ids: list[int] | None,
) -> tuple[str | None, list[int] | None]:
    has_prompt = prompt is not None
    has_prompt_token_ids = prompt_token_ids is not None
    if has_prompt == has_prompt_token_ids:
        raise OpenAIAPIError(
            400,
            "Provide exactly one of prompt or prompt_token_ids.",
            param="prompt_token_ids" if has_prompt else "prompt",
        )
    if has_prompt_token_ids:
        return None, _normalize_prompt_token_ids(prompt_token_ids)
    assert prompt is not None
    return _normalize_prompt(prompt), None


def _invalid_request_body(message: str) -> OpenAIAPIError:
    return OpenAIAPIError(400, f"Invalid request body: {message}", param=None)


def _decode_json_body(body: bytes) -> dict[str, Any]:
    try:
        payload = _json_decode(body)
    except Exception as exc:
        raise _invalid_request_body(str(exc)) from exc
    if not isinstance(payload, dict):
        raise _invalid_request_body("Top-level JSON body must be an object.")
    return payload


async def _load_json_body(request: Request) -> dict[str, Any]:
    return _decode_json_body(await request.body())


def _reject_unknown_payload_fields(payload: dict[str, Any], allowed_fields: set[str]) -> dict[str, Any] | None:
    extra = {key: value for key, value in payload.items() if key not in allowed_fields}
    unsupported = [key for key, value in extra.items() if value is not None]
    if unsupported:
        raise OpenAIAPIError(
            400,
            f"Unsupported request field(s): {', '.join(sorted(unsupported))}.",
        )
    return extra or None


def _require_string(payload: dict[str, Any], field_name: str) -> str:
    if field_name not in payload:
        raise _invalid_request_body(f"Missing required field: {field_name}.")
    value = payload[field_name]
    if not isinstance(value, str):
        raise _invalid_request_body(f"Field {field_name} must be a string.")
    return value


def _optional_bool(payload: dict[str, Any], field_name: str, default=None):
    if field_name not in payload:
        return default
    value = payload[field_name]
    if value is None:
        return None
    if not isinstance(value, bool):
        raise _invalid_request_body(f"Field {field_name} must be a boolean.")
    return value


def _optional_int(payload: dict[str, Any], field_name: str, default=None):
    if field_name not in payload:
        return default
    value = payload[field_name]
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise _invalid_request_body(f"Field {field_name} must be an integer.")
    return value


def _optional_float(payload: dict[str, Any], field_name: str, default=None):
    if field_name not in payload:
        return default
    value = payload[field_name]
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _invalid_request_body(f"Field {field_name} must be a number.")
    return float(value)


def _optional_str_or_str_list(payload: dict[str, Any], field_name: str):
    if field_name not in payload:
        return None
    value = payload[field_name]
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return value
    raise _invalid_request_body(f"Field {field_name} must be a string or list of strings.")


def _optional_int_list(payload: dict[str, Any], field_name: str):
    if field_name not in payload:
        return None
    value = payload[field_name]
    if value is None:
        return None
    if not isinstance(value, list):
        raise _invalid_request_body(f"Field {field_name} must be a list of integers.")
    token_ids: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int):
            raise _invalid_request_body(f"Field {field_name} must be a list of integers.")
        token_ids.append(int(item))
    return token_ids


def _parse_completion_request_payload(payload: dict[str, Any]) -> ParsedCompletionRequest:
    allowed_fields = {
        "model",
        "prompt",
        "prompt_token_ids",
        "max_tokens",
        "temperature",
        "stream",
        "n",
        "top_p",
        "stop",
        "presence_penalty",
        "frequency_penalty",
        "penalty_decay",
        "logprobs",
        "echo",
        "seed",
        "user",
        "stream_options",
    }
    extra = _reject_unknown_payload_fields(payload, allowed_fields)
    prompt = payload.get("prompt")
    prompt_token_ids = _optional_int_list(payload, "prompt_token_ids")
    if prompt is not None and not isinstance(prompt, str):
        if not (isinstance(prompt, list) and all(isinstance(item, str) for item in prompt)):
            raise _invalid_request_body("Field prompt must be a string or list of strings.")
    if (prompt is None) == (prompt_token_ids is None):
        raise _invalid_request_body("Provide exactly one of prompt or prompt_token_ids.")
    return ParsedCompletionRequest(
        model=_require_string(payload, "model"),
        prompt=prompt,
        prompt_token_ids=prompt_token_ids,
        max_tokens=_optional_int(payload, "max_tokens"),
        temperature=_optional_float(payload, "temperature"),
        stream=_optional_bool(payload, "stream", False),
        n=_optional_int(payload, "n", 1),
        top_p=_optional_float(payload, "top_p"),
        stop=_optional_str_or_str_list(payload, "stop"),
        presence_penalty=_optional_float(payload, "presence_penalty"),
        frequency_penalty=_optional_float(payload, "frequency_penalty"),
        penalty_decay=_optional_float(payload, "penalty_decay"),
        logprobs=payload.get("logprobs"),
        echo=_optional_bool(payload, "echo"),
        seed=_optional_int(payload, "seed"),
        user=payload.get("user"),
        stream_options=payload.get("stream_options"),
        model_extra=extra,
    )


def _parse_chat_messages(payload: dict[str, Any]) -> list[dict[str, Any]]:
    if "messages" not in payload:
        raise _invalid_request_body("Missing required field: messages.")
    messages = payload["messages"]
    if not isinstance(messages, list):
        raise _invalid_request_body("Field messages must be a list.")
    parsed_messages: list[dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, dict):
            raise _invalid_request_body("Each message must be an object.")
        role = message.get("role")
        if not isinstance(role, str):
            raise _invalid_request_body("Each message.role must be a string.")
        content = message.get("content")
        if content is not None and not isinstance(content, str):
            if not isinstance(content, list):
                raise _invalid_request_body("Each message.content must be a string, list, or null.")
            for part in content:
                if not isinstance(part, dict):
                    raise _invalid_request_body("Each content part must be an object.")
        parsed_messages.append({"role": role, "content": content})
    return parsed_messages


def _parse_chat_request_payload(payload: dict[str, Any]) -> ParsedChatCompletionRequest:
    allowed_fields = {
        "model",
        "messages",
        "max_tokens",
        "max_completion_tokens",
        "temperature",
        "stream",
        "n",
        "top_p",
        "stop",
        "presence_penalty",
        "frequency_penalty",
        "penalty_decay",
        "logprobs",
        "top_logprobs",
        "tools",
        "tool_choice",
        "parallel_tool_calls",
        "response_format",
        "seed",
        "user",
        "stream_options",
    }
    extra = _reject_unknown_payload_fields(payload, allowed_fields)
    return ParsedChatCompletionRequest(
        model=_require_string(payload, "model"),
        messages=_parse_chat_messages(payload),
        max_tokens=_optional_int(payload, "max_tokens"),
        max_completion_tokens=_optional_int(payload, "max_completion_tokens"),
        temperature=_optional_float(payload, "temperature"),
        stream=_optional_bool(payload, "stream", False),
        n=_optional_int(payload, "n", 1),
        top_p=_optional_float(payload, "top_p"),
        stop=_optional_str_or_str_list(payload, "stop"),
        presence_penalty=_optional_float(payload, "presence_penalty"),
        frequency_penalty=_optional_float(payload, "frequency_penalty"),
        penalty_decay=_optional_float(payload, "penalty_decay"),
        logprobs=_optional_bool(payload, "logprobs"),
        top_logprobs=_optional_int(payload, "top_logprobs"),
        tools=payload.get("tools"),
        tool_choice=payload.get("tool_choice"),
        parallel_tool_calls=_optional_bool(payload, "parallel_tool_calls"),
        response_format=payload.get("response_format"),
        seed=_optional_int(payload, "seed"),
        user=payload.get("user"),
        stream_options=payload.get("stream_options"),
        model_extra=extra,
    )


def _usage_dict(prompt_token_count: int, completion_token_count: int):
    return {
        "prompt_tokens": prompt_token_count,
        "completion_tokens": completion_token_count,
        "total_tokens": prompt_token_count + completion_token_count,
    }


def _completion_stream_usage_event(
    *,
    completion_id: str,
    created: int,
    model: str,
    prompt_token_count: int,
    completion_token_count: int,
) -> dict[str, Any]:
    return {
        "id": completion_id,
        "object": "text_completion",
        "created": created,
        "model": model,
        "choices": [],
        "usage": _usage_dict(prompt_token_count, completion_token_count),
    }


def _chat_stream_usage_event(
    *,
    completion_id: str,
    created: int,
    model: str,
    prompt_token_count: int,
    completion_token_count: int,
) -> dict[str, Any]:
    return {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [],
        "usage": _usage_dict(prompt_token_count, completion_token_count),
    }


def _finish_reason(completion_token_count: int, requested_max_tokens: int) -> str:
    return "length" if completion_token_count >= requested_max_tokens else "stop"


def _format_ms(seconds: float | None) -> str:
    value = 0.0 if seconds is None else seconds * 1000.0
    return f"{value:.3f}"


def _format_rate(tokens: int, elapsed_s: float | None) -> str | None:
    if elapsed_s is None or elapsed_s <= 0:
        return None
    return f"{tokens / elapsed_s:.3f}"


def _chat_stream_finish_reason(finish_reason: str) -> str | None:
    # openai-python's chat stream helper raises on finish_reason="length" when
    # get_final_completion() reparses the aggregated response. Keep sync
    # responses accurate and suppress the streamed terminal marker so the helper
    # can still return the accumulated text.
    if finish_reason == "length":
        return None
    return finish_reason


def _decode_visible_text(tokenizer, token_ids: list[int]) -> str:
    try:
        return tokenizer.decode(token_ids, utf8_errors="ignore")
    except TypeError:
        return tokenizer.decode(token_ids)


def _token_texts_and_offsets(tokenizer, token_ids: list[int]) -> tuple[list[str], list[int]]:
    pieces: list[str] = []
    offsets: list[int] = []
    prefix_text = ""
    prefix_token_ids: list[int] = []
    for token_id in token_ids:
        offsets.append(len(prefix_text))
        prefix_token_ids.append(token_id)
        decoded = _decode_visible_text(tokenizer, prefix_token_ids)
        if decoded.startswith(prefix_text):
            pieces.append(decoded[len(prefix_text):])
        else:
            pieces.append(_decode_visible_text(tokenizer, [token_id]))
        prefix_text = decoded
    return pieces, offsets


def _format_top_logprobs(tokenizer, top_logprobs: list[tuple[int, float]] | None) -> dict[str, float] | None:
    if top_logprobs is None:
        return None
    formatted: dict[str, float] = {}
    for token_id, logprob in top_logprobs:
        formatted[_decode_visible_text(tokenizer, [token_id])] = logprob
    return formatted


def _completion_response_text(tokenizer, request: BatchedRequest) -> str:
    token_ids = list(request.completion_token_ids)
    if request.echo and request.prompt_token_ids is not None:
        token_ids = list(request.prompt_token_ids) + token_ids
    if not token_ids:
        return ""
    return _decode_visible_text(tokenizer, token_ids)


def _build_completion_logprobs(tokenizer, request: BatchedRequest) -> dict[str, Any] | None:
    if not request.capture_logprobs:
        return None
    token_ids: list[int] = []
    token_logprobs: list[float | None] = []
    top_logprobs: list[dict[str, float] | None] = []
    if request.echo and request.prompt_token_ids:
        token_ids.append(request.prompt_token_ids[0])
        token_logprobs.append(None)
        top_logprobs.append(None)
        for record in request.prompt_logprob_records:
            token_ids.append(record.token_id)
            token_logprobs.append(record.logprob)
            top_logprobs.append(_format_top_logprobs(tokenizer, record.top_logprobs))
    for record in request.completion_logprob_records:
        token_ids.append(record.token_id)
        token_logprobs.append(record.logprob)
        top_logprobs.append(_format_top_logprobs(tokenizer, record.top_logprobs))
    tokens, offsets = _token_texts_and_offsets(tokenizer, token_ids)
    return {
        "tokens": tokens,
        "token_logprobs": token_logprobs,
        "top_logprobs": top_logprobs,
        "text_offset": offsets,
    }


def _tokenize_response_body(tokenizer, text: str) -> dict[str, Any]:
    token_ids = tokenizer.encode(text)
    tokens, offsets = _token_texts_and_offsets(tokenizer, token_ids)
    return {
        "token_ids": token_ids,
        "tokens": tokens,
        "text_offset": offsets,
        "count": len(token_ids),
    }


def _detokenize_response_body(tokenizer, token_ids: list[int]) -> dict[str, Any]:
    normalized = _normalize_prompt_token_ids(token_ids)
    return {
        "text": _decode_visible_text(tokenizer, normalized),
        "count": len(normalized),
    }


def _parse_tokenize_payload(payload: dict[str, Any]) -> tuple[str, str]:
    _reject_unknown_payload_fields(payload, {"model", "text"})
    return _require_string(payload, "model"), _require_string(payload, "text")


def _parse_detokenize_payload(payload: dict[str, Any]) -> tuple[str, list[int]]:
    _reject_unknown_payload_fields(payload, {"model", "token_ids"})
    token_ids = _optional_int_list(payload, "token_ids")
    if token_ids is None:
        raise _invalid_request_body("Missing required field: token_ids.")
    return _require_string(payload, "model"), token_ids


def _handle_tokenize_payload(state: Any, payload: dict[str, Any]) -> dict[str, Any]:
    model, text = _parse_tokenize_payload(payload)
    _validate_model(model, state)
    return _tokenize_response_body(_state_tokenizer(state), text)


def _handle_detokenize_payload(state: Any, payload: dict[str, Any]) -> dict[str, Any]:
    model, token_ids = _parse_detokenize_payload(payload)
    _validate_model(model, state)
    return _detokenize_response_body(_state_tokenizer(state), token_ids)


def _sse_payload(data: dict[str, Any] | str) -> bytes:
    if isinstance(data, str):
        return f"data: {data}\n\n".encode("utf-8")
    return b"data: " + _json_encode(data) + b"\n\n"


def _json_encode(payload: Any) -> bytes:
    return msgspec.json.encode(payload)


def _json_decode(payload: bytes) -> Any:
    return msgspec.json.decode(payload)


def _json_response(
    content: Any,
    *,
    status_code: int = 200,
    headers: dict[str, str] | None = None,
) -> Response:
    return Response(
        content=_json_encode(content),
        status_code=status_code,
        headers=headers,
        media_type="application/json",
    )


def _enable_uvloop() -> None:
    if uvloop is not None:
        asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())


async def _ipc_write_frame(writer: asyncio.StreamWriter, payload: dict[str, Any]) -> None:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    writer.write(len(data).to_bytes(4, "little"))
    writer.write(data)
    await writer.drain()


async def _ipc_read_frame(reader: asyncio.StreamReader) -> dict[str, Any] | None:
    header = await reader.readexactly(4)
    size = int.from_bytes(header, "little")
    if size == 0:
        return None
    payload = await reader.readexactly(size)
    return json.loads(payload.decode("utf-8"))


def _queue_put_frame(response_queue: Any, request_id: str, payload: dict[str, Any]) -> None:
    response_queue.put({"request_id": request_id, **payload})


def _queue_backend_send_completed_request(
    request: BatchedRequest,
    state: ServerState,
    response_queues: list[Any],
) -> None:
    frontend_id = request.frontend_id
    if frontend_id is None:
        raise RuntimeError("Queue backend completed request is missing frontend_id.")
    response_queue = response_queues[frontend_id]
    if request.error is not None:
        if isinstance(request.error, OpenAIAPIError):
            error = request.error
        else:
            error = OpenAIAPIError(
                500,
                f"Backend request failed: {type(request.error).__name__}: {request.error}",
                error_type="server_error",
                code="backend_failure",
            )
        _queue_put_frame(
            response_queue,
            request.request_id,
            {"kind": "error", "error": _serialize_openai_error(error)},
        )
        return
    result = request.result()
    _queue_put_frame(
        response_queue,
        request.request_id,
        {
            "kind": "result",
            "prompt_token_count": len(result.prompt_token_ids),
            "completion_token_count": len(result.completion_token_ids),
            "text": _completion_response_text(state.llm.tokenizer, request)
            if request.endpoint == "completion"
            else result.text,
            "logprobs": _build_completion_logprobs(state.llm.tokenizer, request)
            if request.endpoint == "completion"
            else None,
            "finish_reason": result.finish_reason,
            "queue_wait_s": request.queue_wait_s,
            "ttft_s": result.ttft_s,
            "generation_s": result.generation_s,
            "total_s": 0.0 if request.total_s is None else request.total_s,
        },
    )


_HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
}


def _proxy_response_headers(headers: httpx.Headers, *, streaming: bool) -> dict[str, str]:
    filtered: dict[str, str] = {}
    for key, value in headers.items():
        lower = key.lower()
        if lower in _HOP_BY_HOP_HEADERS:
            continue
        if streaming and lower == "content-length":
            continue
        filtered[key] = value
    return filtered


def _require_api_key(state: ServerState, authorization: str | None):
    if state.api_key is None:
        return
    expected = f"Bearer {state.api_key}"
    if authorization != expected:
        raise OpenAIAPIError(
            401,
            "Invalid or missing API key.",
            error_type="authentication_error",
            code="invalid_api_key",
        )


def _response_headers(
    *,
    request_id: str,
    prompt_token_count: int,
    completion_token_count: int | None = None,
    queue_wait_s: float,
    processing_s: float,
    request_parse_s: float | None = None,
    request_setup_s: float | None = None,
    response_build_s: float | None = None,
    server_app_s: float | None = None,
    ttft_s: float | None = None,
    generation_s: float | None = None,
    total_s: float | None = None,
    streaming: bool,
) -> dict[str, str]:
    headers = {
        "x-request-id": request_id,
        "openai-processing-ms": _format_ms(processing_s),
        "x-nanovllm-streaming": "true" if streaming else "false",
        "x-nanovllm-metrics-scope": "partial" if streaming else "final",
        "x-nanovllm-queue-wait-ms": _format_ms(queue_wait_s),
        "x-nanovllm-prompt-tokens": str(prompt_token_count),
    }
    if request_parse_s is not None:
        headers["x-nanovllm-request-parse-ms"] = _format_ms(request_parse_s)
    if request_setup_s is not None:
        headers["x-nanovllm-request-setup-ms"] = _format_ms(request_setup_s)
    if response_build_s is not None:
        headers["x-nanovllm-response-build-ms"] = _format_ms(response_build_s)
    if server_app_s is not None:
        headers["x-nanovllm-server-app-ms"] = _format_ms(server_app_s)
    if ttft_s is not None:
        headers["x-nanovllm-ttft-ms"] = _format_ms(ttft_s)
    if completion_token_count is not None:
        headers["x-nanovllm-completion-tokens"] = str(completion_token_count)
    if generation_s is not None:
        headers["x-nanovllm-generation-ms"] = _format_ms(generation_s)
        output_tps = _format_rate(completion_token_count or 0, generation_s)
        if output_tps is not None:
            headers["x-nanovllm-output-tokens-per-second"] = output_tps
        if ttft_s is not None and completion_token_count is not None and completion_token_count > 1:
            decode_s = generation_s - ttft_s
            decode_tps = _format_rate(completion_token_count - 1, decode_s)
            if decode_tps is not None:
                headers["x-nanovllm-decode-tokens-per-second"] = decode_tps
    if total_s is not None:
        headers["x-nanovllm-total-ms"] = _format_ms(total_s)
    return headers


def _raise_request_error(request: BatchedRequest):
    if request.error is None:
        return
    if isinstance(request.error, OpenAIAPIError):
        raise request.error
    raise RuntimeError("OpenAI API batch worker failed.") from request.error


def _submit_request(
    state: ServerState,
    *,
    endpoint: Literal["completion", "chat"],
    prepared: PreparedOpenAIRequest,
    request_id: str,
    created: int,
    http_received_at: float,
    handler_started_at: float,
    stream: bool,
    completion_notify=None,
    frontend_id: int | None = None,
    use_async_signals: bool = True,
) -> BatchedRequest:
    if state.batcher is None:
        raise RuntimeError("OpenAI API batch worker is not initialized.")
    loop = asyncio.get_running_loop() if use_async_signals else None
    request = BatchedRequest(
        request_id=request_id,
        endpoint=endpoint,
        prompt_text=prepared.prompt_text,
        sampling_params=prepared.sampling_params,
        requested_max_tokens=prepared.requested_max_tokens,
        prompt_token_ids_input=None if prepared.prompt_token_ids is None else list(prepared.prompt_token_ids),
        capture_logprobs=prepared.capture_logprobs,
        top_logprobs=prepared.top_logprobs,
        echo=prepared.echo,
        stop_token_seqs=prepared.stop_token_seqs,
        created=created,
        http_received_at=http_received_at,
        handler_started_at=handler_started_at,
        http_started_at=time.perf_counter(),
        stream=stream,
        loop=loop,
        ready_event=asyncio.Event() if stream and use_async_signals else None,
        done_event=asyncio.Event() if use_async_signals else None,
        stream_queue=asyncio.Queue() if stream and use_async_signals else None,
        completion_notify=completion_notify,
        frontend_id=frontend_id,
    )
    state.batcher.submit(request)
    return request


def _prepare_completion_request(
    state: Any,
    req: CompletionRequest | ParsedCompletionRequest,
) -> PreparedOpenAIRequest:
    _validate_openai_request_model(req.model, state)
    sampling_params = _sampling_params_from_completion(req)
    prompt_text, prompt_token_ids = _resolve_completion_prompt(req.prompt, req.prompt_token_ids)
    if prompt_token_ids is None and prompt_text is not None:
        prompt_token_ids = _state_encode_text(state, prompt_text)
    requested_max_tokens = DEFAULT_OPENAI_MAX_TOKENS if req.max_tokens is None else int(req.max_tokens)
    return PreparedOpenAIRequest(
        prompt_text=prompt_text,
        sampling_params=sampling_params,
        requested_max_tokens=requested_max_tokens,
        prompt_token_ids=prompt_token_ids,
        capture_logprobs=_completion_logprob_count(req.logprobs) > 0,
        top_logprobs=_completion_logprob_count(req.logprobs),
        echo=bool(req.echo),
    )


def _prepare_chat_request(
    state: Any,
    req: ChatCompletionRequest | ParsedChatCompletionRequest,
) -> PreparedOpenAIRequest:
    _base_model, mode = _validate_openai_request_model(req.model, state)
    sampling_params = _sampling_params_from_chat(req)
    tokenizer = _state_tokenizer(state)
    prompt_text = _render_openai_chat_prompt(tokenizer, req.messages, mode=mode)
    prompt_token_ids = _state_encode_text(state, prompt_text)
    # print(f"```{prompt_text}```")
    return PreparedOpenAIRequest(
        prompt_text=prompt_text,
        sampling_params=sampling_params,
        requested_max_tokens=int(sampling_params.max_tokens),
        prompt_token_ids=prompt_token_ids,
    )


async def _serve_completion_request(
    state: ServerState,
    req: CompletionRequest | ParsedCompletionRequest,
    *,
    authorization: str | None,
    http_received_at: float,
    handler_started_at: float,
) -> Response:
    _require_api_key(state, authorization)
    response_model = req.model
    prepared = _prepare_completion_request(state, req)
    stream_include_usage = _stream_options_include_usage(req.stream_options)
    created = int(time.time())
    completion_id = f"cmpl-{uuid.uuid4().hex}"
    request = _submit_request(
        state,
        endpoint="completion",
        prepared=prepared,
        request_id=completion_id,
        created=created,
        http_received_at=http_received_at,
        handler_started_at=handler_started_at,
        stream=bool(req.stream),
    )
    if req.stream:
        assert request.ready_event is not None
        try:
            await request.ready_event.wait()
        except asyncio.CancelledError:
            state.batcher.cancel(request)
            raise
        _raise_request_error(request)
        assert request.prompt_token_ids is not None
        request.response_built_at = time.perf_counter()
        headers = _response_headers(
            request_id=completion_id,
            prompt_token_count=len(request.prompt_token_ids),
            queue_wait_s=request.queue_wait_s,
            processing_s=0.0 if request.processing_s is None else request.processing_s,
            request_parse_s=request.request_parse_s,
            request_setup_s=request.request_setup_s,
            response_build_s=request.response_build_s,
            server_app_s=request.server_app_s,
            ttft_s=request.ttft_s,
            streaming=True,
        )

        async def event_stream():
            completed = False
            try:
                assert request.stream_queue is not None
                while True:
                    kind, value = await request.stream_queue.get()
                    if kind == "delta":
                        yield _sse_payload(
                            {
                                "id": completion_id,
                                "object": "text_completion",
                                "created": created,
                                "model": response_model,
                                "choices": [
                                    {
                                        "index": 0,
                                        "text": value or "",
                                        "finish_reason": None,
                                        "logprobs": None,
                                    }
                                ],
                            }
                        )
                        continue
                    if kind == "finish":
                        yield _sse_payload(
                            {
                                "id": completion_id,
                                "object": "text_completion",
                                "created": created,
                                "model": response_model,
                                "choices": [
                                    {
                                        "index": 0,
                                        "text": "",
                                        "finish_reason": value,
                                        "logprobs": None,
                                    }
                                ],
                            }
                        )
                        continue
                    if kind == "error":
                        completed = True
                        _raise_request_error(request)
                    if kind == "done":
                        if stream_include_usage:
                            yield _sse_payload(
                                _completion_stream_usage_event(
                                    completion_id=completion_id,
                                    created=created,
                                    model=response_model,
                                    prompt_token_count=len(request.prompt_token_ids),
                                    completion_token_count=len(request.completion_token_ids),
                                )
                            )
                        completed = True
                        yield _sse_payload("[DONE]")
                        return
            finally:
                if not completed:
                    state.batcher.cancel(request)

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers=headers,
        )

    assert request.done_event is not None
    await request.done_event.wait()
    _raise_request_error(request)
    result = request.result()
    response_text = _completion_response_text(state.llm.tokenizer, request)
    response_logprobs = _build_completion_logprobs(state.llm.tokenizer, request)
    total_s = 0.0 if request.total_s is None else request.total_s
    request.response_built_at = time.perf_counter()
    headers = _response_headers(
        request_id=completion_id,
        prompt_token_count=len(result.prompt_token_ids),
        completion_token_count=len(result.completion_token_ids),
        queue_wait_s=request.queue_wait_s,
        processing_s=total_s,
        request_parse_s=request.request_parse_s,
        request_setup_s=request.request_setup_s,
        response_build_s=request.response_build_s,
        server_app_s=request.server_app_s,
        ttft_s=result.ttft_s,
        generation_s=result.generation_s,
        total_s=total_s,
        streaming=False,
    )
    return _json_response(
        {
            "id": completion_id,
            "object": "text_completion",
            "created": created,
            "model": response_model,
            "choices": [
                {
                    "index": 0,
                    "text": response_text,
                    "finish_reason": result.finish_reason,
                    "logprobs": response_logprobs,
                }
            ],
            "usage": _usage_dict(
                len(result.prompt_token_ids),
                len(result.completion_token_ids),
            ),
        },
        headers=headers,
    )


async def _serve_chat_completion_request(
    state: ServerState,
    req: ChatCompletionRequest | ParsedChatCompletionRequest,
    *,
    authorization: str | None,
    http_received_at: float,
    handler_started_at: float,
) -> Response:
    _require_api_key(state, authorization)
    response_model = req.model
    prepared = _prepare_chat_request(state, req)
    stream_include_usage = _stream_options_include_usage(req.stream_options)
    created = int(time.time())
    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    request = _submit_request(
        state,
        endpoint="chat",
        prepared=prepared,
        request_id=completion_id,
        created=created,
        http_received_at=http_received_at,
        handler_started_at=handler_started_at,
        stream=bool(req.stream),
    )
    prompt_mode = _chat_output_mode_from_prompt(request.prompt_text)
    if req.stream:
        assert request.ready_event is not None
        try:
            await request.ready_event.wait()
        except asyncio.CancelledError:
            state.batcher.cancel(request)
            raise
        _raise_request_error(request)
        assert request.prompt_token_ids is not None
        request.response_built_at = time.perf_counter()
        headers = _response_headers(
            request_id=completion_id,
            prompt_token_count=len(request.prompt_token_ids),
            queue_wait_s=request.queue_wait_s,
            processing_s=0.0 if request.processing_s is None else request.processing_s,
            request_parse_s=request.request_parse_s,
            request_setup_s=request.request_setup_s,
            response_build_s=request.response_build_s,
            server_app_s=request.server_app_s,
            ttft_s=request.ttft_s,
            streaming=True,
        )

        async def event_stream():
            completed = False
            filter_state = _make_chat_output_filter(prompt_mode)
            flushed = False
            try:
                yield _sse_payload(
                    {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": response_model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": _chat_stream_delta_payload(role="assistant", content=""),
                                "finish_reason": None,
                            }
                        ],
                    }
                )
                assert request.stream_queue is not None
                while True:
                    kind, value = await request.stream_queue.get()
                    if kind == "delta":
                        for output_kind, output_text in _filter_chat_delta(filter_state, value or ""):
                            yield _sse_payload(
                                {
                                    "id": completion_id,
                                    "object": "chat.completion.chunk",
                                    "created": created,
                                    "model": response_model,
                                    "choices": [
                                        {
                                            "index": 0,
                                            "delta": _chat_stream_delta_payload(
                                                content=output_text if output_kind == "content" else None,
                                                reasoning_content=output_text if output_kind == "reasoning_content" else None,
                                            ),
                                            "finish_reason": None,
                                        }
                                    ],
                                }
                            )
                        continue
                    if kind == "finish":
                        if not flushed:
                            for output_kind, output_text in _flush_chat_delta_filter(filter_state):
                                yield _sse_payload(
                                    {
                                        "id": completion_id,
                                        "object": "chat.completion.chunk",
                                        "created": created,
                                        "model": response_model,
                                        "choices": [
                                            {
                                                "index": 0,
                                                "delta": _chat_stream_delta_payload(
                                                    content=output_text if output_kind == "content" else None,
                                                    reasoning_content=output_text if output_kind == "reasoning_content" else None,
                                                ),
                                                "finish_reason": None,
                                            }
                                        ],
                                    }
                                )
                            flushed = True
                        yield _sse_payload(
                            {
                                "id": completion_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": response_model,
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": {},
                                        "finish_reason": _chat_stream_finish_reason(
                                            value or "stop"
                                        ),
                                    }
                                ],
                            }
                        )
                        continue
                    if kind == "error":
                        completed = True
                        _raise_request_error(request)
                    if kind == "done":
                        if not flushed:
                            for output_kind, output_text in _flush_chat_delta_filter(filter_state):
                                yield _sse_payload(
                                    {
                                        "id": completion_id,
                                        "object": "chat.completion.chunk",
                                        "created": created,
                                        "model": response_model,
                                        "choices": [
                                            {
                                                "index": 0,
                                                "delta": _chat_stream_delta_payload(
                                                    content=output_text if output_kind == "content" else None,
                                                    reasoning_content=output_text if output_kind == "reasoning_content" else None,
                                                ),
                                                "finish_reason": None,
                                            }
                                        ],
                                    }
                                )
                            flushed = True
                        if stream_include_usage:
                            yield _sse_payload(
                                _chat_stream_usage_event(
                                    completion_id=completion_id,
                                    created=created,
                                    model=response_model,
                                    prompt_token_count=len(request.prompt_token_ids),
                                    completion_token_count=len(request.completion_token_ids),
                                )
                            )
                        completed = True
                        yield _sse_payload("[DONE]")
                        return
            finally:
                if not completed:
                    state.batcher.cancel(request)

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers=headers,
        )

    assert request.done_event is not None
    await request.done_event.wait()
    _raise_request_error(request)
    result = request.result()
    content_text, reasoning_text = _filter_chat_text(result.text, mode=prompt_mode)
    total_s = 0.0 if request.total_s is None else request.total_s
    request.response_built_at = time.perf_counter()
    headers = _response_headers(
        request_id=completion_id,
        prompt_token_count=len(result.prompt_token_ids),
        completion_token_count=len(result.completion_token_ids),
        queue_wait_s=request.queue_wait_s,
        processing_s=total_s,
        request_parse_s=request.request_parse_s,
        request_setup_s=request.request_setup_s,
        response_build_s=request.response_build_s,
        server_app_s=request.server_app_s,
        ttft_s=result.ttft_s,
        generation_s=result.generation_s,
        total_s=total_s,
        streaming=False,
    )
    return _json_response(
        {
            "id": completion_id,
            "object": "chat.completion",
            "created": created,
            "model": response_model,
            "choices": [
                {
                    "index": 0,
                    "message": _assistant_message_payload(content_text, reasoning_text),
                    "finish_reason": result.finish_reason,
                }
            ],
            "usage": _usage_dict(
                len(result.prompt_token_ids),
                len(result.completion_token_ids),
            ),
        },
        headers=headers,
    )


def _is_lightning_private_payload(payload: dict[str, Any]) -> bool:
    return "contents" in payload or "prefix" in payload or "suffix" in payload


def _private_request_model(payload: dict[str, Any], state: Any) -> str:
    value = payload.get("model")
    if value is None:
        return getattr(state, "model_id", "rwkv7")
    if not isinstance(value, str):
        raise _invalid_request_body("Field model must be a string.")
    return value


def _require_private_api_key(state: Any, authorization: str | None, payload: dict[str, Any]) -> None:
    api_key = getattr(state, "api_key", None)
    if api_key is None:
        return
    if authorization == f"Bearer {api_key}":
        return
    if payload.get("password") == api_key:
        return
    raise OpenAIAPIError(
        401,
        "Invalid or missing API key.",
        error_type="authentication_error",
        code="invalid_api_key",
    )


def _lightning_string_list(payload: dict[str, Any], field_name: str, default=None) -> list[str]:
    if field_name not in payload:
        return list(default or [])
    value = payload[field_name]
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise _invalid_request_body(f"Field {field_name} must be a list of strings.")
    return list(value)


def _lightning_sampling_params(payload: dict[str, Any]) -> SamplingParams:
    max_tokens = _optional_int(payload, "max_tokens", DEFAULT_OPENAI_MAX_TOKENS)
    temperature = _optional_float(payload, "temperature", 1.0)
    top_k = _optional_int(payload, "top_k", 50)
    top_p = _optional_float(payload, "top_p", 0.6)
    alpha_presence = _optional_float(payload, "alpha_presence", 2.0)
    alpha_frequency = _optional_float(payload, "alpha_frequency", 0.2)
    alpha_decay = _optional_float(payload, "alpha_decay", DEFAULT_OPENAI_PENALTY_DECAY)
    if max_tokens is None or max_tokens <= 0:
        raise OpenAIAPIError(400, "max_tokens must be positive.", param="max_tokens")
    if temperature is None or temperature < 0:
        raise OpenAIAPIError(400, "temperature must be non-negative.", param="temperature")
    if top_k is None:
        top_k = -1
    if top_k != -1 and top_k <= 0:
        raise OpenAIAPIError(400, "top_k must be -1 or a positive integer.", param="top_k")
    if top_p is None or not (0.0 <= top_p <= 1.0):
        raise OpenAIAPIError(400, "top_p must be in [0, 1].", param="top_p")
    if alpha_decay is None or not (0.0 <= alpha_decay <= 1.0):
        raise OpenAIAPIError(400, "alpha_decay must be in [0, 1].", param="alpha_decay")
    return SamplingParams(
        temperature=float(temperature),
        top_k=int(top_k),
        top_p=float(top_p),
        presence_penalty=float(alpha_presence or 0.0),
        repetition_penalty=float(alpha_frequency or 0.0),
        penalty_decay=float(alpha_decay),
        max_tokens=int(max_tokens),
    )


def _lightning_stop_token_seqs(state: Any, payload: dict[str, Any]) -> tuple[tuple[int, ...], ...]:
    stop_tokens = _lightning_string_list(payload, "stop_tokens", ["\nUser:"])
    tokenizer = _state_tokenizer(state)
    encoded = []
    for stop_text in stop_tokens:
        token_ids = tokenizer.encode(stop_text)
        if token_ids:
            encoded.append(tuple(int(token_id) for token_id in token_ids))
    return tuple(encoded)


def _lightning_prepared_requests(
    state: Any,
    payload: dict[str, Any],
    prompts: list[str],
    *,
    pad_zero: bool | None = None,
) -> list[PreparedOpenAIRequest]:
    sampling_params = _lightning_sampling_params(payload)
    stop_token_seqs = _lightning_stop_token_seqs(state, payload)
    if pad_zero is None:
        pad_zero = bool(payload.get("pad_zero", False))
    prepared_requests = []
    for prompt in prompts:
        prompt_text = prompt
        prompt_token_ids = None
        if pad_zero:
            prompt_token_ids = [0] + _state_encode_text(state, prompt)
            prompt_text = None
        prepared_requests.append(
            PreparedOpenAIRequest(
                prompt_text=prompt_text,
                prompt_token_ids=prompt_token_ids,
                sampling_params=sampling_params,
                requested_max_tokens=int(sampling_params.max_tokens),
                stop_token_seqs=stop_token_seqs,
            )
        )
    return prepared_requests


def _lightning_response_choice(
    index: int,
    text: str,
    finish_reason: str,
    reasoning_text: str | None = None,
) -> dict[str, Any]:
    return {
        "index": index,
        "message": _assistant_message_payload(text, reasoning_text),
        "finish_reason": finish_reason,
    }


def _lightning_session_store(state: Any) -> dict[str, str]:
    sessions = getattr(state, "lightning_sessions", None)
    if sessions is None:
        sessions = {}
        setattr(state, "lightning_sessions", sessions)
    return sessions


def _lightning_session_lock(state: Any):
    lock = getattr(state, "lightning_sessions_lock", None)
    if lock is None:
        lock = threading.Lock()
        setattr(state, "lightning_sessions_lock", lock)
    return lock


def _lightning_prompt_with_session(state: Any, session_id: str | None, prompt: str) -> tuple[str, str | None]:
    if not session_id:
        return prompt, None
    sessions = _lightning_session_store(state)
    lock = _lightning_session_lock(state)
    with lock:
        history = sessions.get(session_id, "")
    if history:
        return history + ("\n\n" + prompt if prompt and not prompt.startswith("\n\n") else prompt), history
    return prompt, ""


def _lightning_update_session(state: Any, session_id: str | None, prompt: str, completion: str) -> None:
    if not session_id:
        return
    sessions = _lightning_session_store(state)
    lock = _lightning_session_lock(state)
    with lock:
        sessions[session_id] = prompt + completion


async def _serve_lightning_private_chat(
    state: ServerState,
    payload: dict[str, Any],
    *,
    authorization: str | None,
    http_received_at: float,
    handler_started_at: float,
    response_object: str = "chat.completion",
    response_id: str = "rwkv7-batch",
    state_session_id: str | None = None,
    force_single_prompt: bool = False,
) -> Response:
    _require_private_api_key(state, authorization, payload)
    prompts = _lightning_string_list(payload, "contents")
    if not prompts:
        raise OpenAIAPIError(400, "Empty prompts list.", param="contents")
    if force_single_prompt and len(prompts) != 1:
        raise OpenAIAPIError(500, "Server Error: Request must be single prompt!", error_type="server_error")

    session_prompt = None
    session_id = state_session_id or payload.get("session_id")
    if session_id is not None and not isinstance(session_id, str):
        raise _invalid_request_body("Field session_id must be a string.")
    if state_session_id is not None or payload.get("session_id") is not None:
        prompts[0], _ = _lightning_prompt_with_session(state, session_id, prompts[0])
        session_prompt = prompts[0]

    stream = bool(_optional_bool(payload, "stream", False))
    model_name = _private_request_model(payload, state)
    prepared_requests = _lightning_prepared_requests(state, payload, prompts)
    created = int(time.time())
    request_group_id = f"rwkv-lightning-{uuid.uuid4().hex}"
    requests = [
        _submit_request(
            state,
            endpoint="chat",
            prepared=prepared,
            request_id=f"{request_group_id}-{index}",
            created=created,
            http_received_at=http_received_at,
            handler_started_at=handler_started_at,
            stream=stream,
        )
        for index, prepared in enumerate(prepared_requests)
    ]

    if not stream:
        await asyncio.gather(*(request.done_event.wait() for request in requests if request.done_event is not None))
        choices = []
        completion_texts = []
        for index, request in enumerate(requests):
            _raise_request_error(request)
            result = request.result()
            prompt_mode = _chat_output_mode_from_prompt(request.prompt_text)
            content_text, reasoning_text = _filter_chat_text(result.text, mode=prompt_mode)
            completion_texts.append(content_text)
            choices.append(_lightning_response_choice(index, content_text, result.finish_reason, reasoning_text))
        if session_prompt is not None and completion_texts:
            _lightning_update_session(state, session_id, session_prompt, completion_texts[0])
        return _json_response(
            {
                "id": response_id,
                "object": response_object,
                "model": model_name,
                "choices": choices,
            }
        )

    await asyncio.gather(*(request.ready_event.wait() for request in requests if request.ready_event is not None))
    for request in requests:
        _raise_request_error(request)

    async def event_stream():
        tasks: dict[asyncio.Task, int] = {}
        completion_buffers = [""] * len(requests)
        filter_states = [_make_chat_output_filter(_chat_output_mode_from_prompt(request.prompt_text)) for request in requests]
        flushed: set[int] = set()
        completed = set()
        try:
            for index, request in enumerate(requests):
                assert request.stream_queue is not None
                tasks[asyncio.create_task(request.stream_queue.get())] = index
            while tasks:
                done, _pending = await asyncio.wait(tasks.keys(), return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    index = tasks.pop(task)
                    request = requests[index]
                    kind, value = task.result()
                    if kind == "delta":
                        for output_kind, output_text in _filter_chat_delta(filter_states[index], value or ""):
                            if output_kind == "content":
                                completion_buffers[index] += output_text
                            yield _sse_payload(
                                {
                                    "object": "chat.completion.chunk",
                                    "choices": [
                                        {
                                            "index": index,
                                            "delta": _chat_stream_delta_payload(
                                                content=output_text if output_kind == "content" else None,
                                                reasoning_content=output_text if output_kind == "reasoning_content" else None,
                                            ),
                                        }
                                    ],
                                }
                            )
                        assert request.stream_queue is not None
                        tasks[asyncio.create_task(request.stream_queue.get())] = index
                        continue
                    if kind == "finish":
                        if index not in flushed:
                            for output_kind, output_text in _flush_chat_delta_filter(filter_states[index]):
                                if output_kind == "content":
                                    completion_buffers[index] += output_text
                                yield _sse_payload(
                                    {
                                        "object": "chat.completion.chunk",
                                        "choices": [
                                            {
                                                "index": index,
                                                "delta": _chat_stream_delta_payload(
                                                    content=output_text if output_kind == "content" else None,
                                                    reasoning_content=output_text if output_kind == "reasoning_content" else None,
                                                ),
                                            }
                                        ],
                                    }
                                )
                            flushed.add(index)
                        assert request.stream_queue is not None
                        tasks[asyncio.create_task(request.stream_queue.get())] = index
                        continue
                    if kind == "error":
                        _raise_request_error(request)
                    if kind == "done":
                        if index not in flushed:
                            for output_kind, output_text in _flush_chat_delta_filter(filter_states[index]):
                                if output_kind == "content":
                                    completion_buffers[index] += output_text
                                yield _sse_payload(
                                    {
                                        "object": "chat.completion.chunk",
                                        "choices": [
                                            {
                                                "index": index,
                                                "delta": _chat_stream_delta_payload(
                                                    content=output_text if output_kind == "content" else None,
                                                    reasoning_content=output_text if output_kind == "reasoning_content" else None,
                                                ),
                                            }
                                        ],
                                    }
                                )
                            flushed.add(index)
                        completed.add(index)
            if session_prompt is not None and completion_buffers:
                _lightning_update_session(state, session_id, session_prompt, completion_buffers[0])
            yield _sse_payload("[DONE]")
        finally:
            for task in tasks:
                task.cancel()
            for index, request in enumerate(requests):
                if index not in completed:
                    state.batcher.cancel(request)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


def _lightning_translation_prompt(source_lang: str, target_lang: str, text: str) -> str:
    lang_names = {
        "zh-CN": "Chinese",
        "zh-TW": "Chinese",
        "en": "English",
        "ja": "Japanese",
        "fr": "French",
        "de": "German",
        "es": "Spanish",
        "ru": "Russian",
    }
    source_name = lang_names.get(source_lang, source_lang)
    target_name = lang_names.get(target_lang, target_lang)
    return f"{source_name}: {text}\n\n{target_name}:"


async def _serve_lightning_translate(
    state: ServerState,
    payload: dict[str, Any],
    *,
    authorization: str | None,
    http_received_at: float,
    handler_started_at: float,
) -> Response:
    _require_private_api_key(state, authorization, payload)
    target_lang = _require_string(payload, "target_lang")
    source_lang = payload.get("source_lang", "auto")
    if not isinstance(source_lang, str):
        raise _invalid_request_body("Field source_lang must be a string.")
    text_list = _lightning_string_list(payload, "text_list")
    prompts = [_lightning_translation_prompt(source_lang, target_lang, text) for text in text_list]
    translate_payload = {
        **payload,
        "contents": prompts,
        "max_tokens": payload.get("max_tokens", 2048),
        "temperature": payload.get("temperature", 1.0),
        "top_k": payload.get("top_k", 1),
        "top_p": payload.get("top_p", 0.0),
        "alpha_presence": payload.get("alpha_presence", 0.0),
        "alpha_frequency": payload.get("alpha_frequency", 0.0),
        "stop_tokens": payload.get("stop_tokens", []),
        "stream": False,
    }
    response = await _serve_lightning_private_chat(
        state,
        translate_payload,
        authorization=authorization,
        http_received_at=http_received_at,
        handler_started_at=handler_started_at,
    )
    body = _json_decode(response.body)
    translations = [
        {
            "detected_source_lang": source_lang if source_lang != "auto" else "en",
            "text": choice["message"]["content"].strip(),
        }
        for choice in body.get("choices", [])
    ]
    return _json_response({"translations": translations})


def _openai_compat_prompt_parts(body: dict[str, Any]) -> tuple[str, list[str]]:
    system_parts = []
    transcript_parts = []
    system = body.get("system")
    if isinstance(system, str) and system.strip():
        system_parts.append(system.strip())
    for message in body.get("messages") or []:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role", "user")).lower()
        content = _coerce_text_content(message.get("content")).strip()
        if not content:
            continue
        if role in {"system", "developer"}:
            system_parts.append(content)
        elif role == "assistant":
            transcript_parts.append(f"Assistant: {content}")
        else:
            transcript_parts.append(f"User: {content}")
    contents = body.get("contents") or []
    if isinstance(contents, list) and contents and isinstance(contents[0], str) and contents[0].strip():
        transcript_parts.append(f"User: {contents[0].strip()}")
    return "\n".join(system_parts).strip(), transcript_parts


def _format_lightning_openai_prompt(body: dict[str, Any]) -> str:
    system_text, transcript_parts = _openai_compat_prompt_parts(body)
    prompt_parts = []
    if system_text:
        prompt_parts.append(f"System: {system_text}")
    prompt_parts.extend(transcript_parts)
    prompt_text = "\n\n".join(part for part in prompt_parts if part).strip()
    if not prompt_text:
        raise OpenAIAPIError(400, "OpenAI chat completions require system or user text.", param="messages")
    if bool(body.get("enable_think", False)):
        return f"{prompt_text}\n\nAssistant: <think"
    return f"{prompt_text}\n\nAssistant: <think>\n</think>\n"


async def _serve_lightning_openai_compat_chat(
    state: ServerState,
    payload: dict[str, Any],
    *,
    authorization: str | None,
    http_received_at: float,
    handler_started_at: float,
) -> Response:
    _require_private_api_key(state, authorization, payload)
    prompt = _format_lightning_openai_prompt(payload)
    private_payload = {**payload, "contents": [prompt]}
    private_payload.setdefault("stream", bool(payload.get("stream", False)))
    private_payload.setdefault("max_tokens", payload.get("max_completion_tokens", payload.get("max_tokens", DEFAULT_OPENAI_MAX_TOKENS)))
    private_payload.setdefault("alpha_presence", payload.get("presence_penalty", payload.get("alpha_presence", 1.0)))
    private_payload.setdefault("alpha_frequency", payload.get("frequency_penalty", payload.get("alpha_frequency", 0.1)))
    private_payload.setdefault("alpha_decay", payload.get("penalty_decay", payload.get("alpha_decay", DEFAULT_OPENAI_PENALTY_DECAY)))
    stream = bool(_optional_bool(private_payload, "stream", False))
    prepared = _lightning_prepared_requests(state, private_payload, [prompt], pad_zero=False)[0]
    created = int(time.time())
    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    model_name = _private_request_model(payload, state)
    request = _submit_request(
        state,
        endpoint="chat",
        prepared=prepared,
        request_id=completion_id,
        created=created,
        http_received_at=http_received_at,
        handler_started_at=handler_started_at,
        stream=stream,
    )
    prompt_mode = _chat_output_mode_from_prompt(request.prompt_text)
    if stream:
        assert request.ready_event is not None
        await request.ready_event.wait()
        _raise_request_error(request)

        async def event_stream():
            completed = False
            filter_state = _make_chat_output_filter(prompt_mode)
            flushed = False
            try:
                yield _sse_payload(
                    {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model_name,
                        "choices": [
                            {
                                "index": 0,
                                "delta": _chat_stream_delta_payload(role="assistant"),
                                "finish_reason": None,
                            }
                        ],
                    }
                )
                assert request.stream_queue is not None
                while True:
                    kind, value = await request.stream_queue.get()
                    if kind == "delta":
                        for output_kind, output_text in _filter_chat_delta(filter_state, value or ""):
                            yield _sse_payload(
                                {
                                    "id": completion_id,
                                    "object": "chat.completion.chunk",
                                    "created": created,
                                    "model": model_name,
                                    "choices": [
                                        {
                                            "index": 0,
                                            "delta": _chat_stream_delta_payload(
                                                content=output_text if output_kind == "content" else None,
                                                reasoning_content=output_text if output_kind == "reasoning_content" else None,
                                            ),
                                            "finish_reason": None,
                                        }
                                    ],
                                }
                            )
                        continue
                    if kind == "finish":
                        if not flushed:
                            for output_kind, output_text in _flush_chat_delta_filter(filter_state):
                                yield _sse_payload(
                                    {
                                        "id": completion_id,
                                        "object": "chat.completion.chunk",
                                        "created": created,
                                        "model": model_name,
                                        "choices": [
                                            {
                                                "index": 0,
                                                "delta": _chat_stream_delta_payload(
                                                    content=output_text if output_kind == "content" else None,
                                                    reasoning_content=output_text if output_kind == "reasoning_content" else None,
                                                ),
                                                "finish_reason": None,
                                            }
                                        ],
                                    }
                                )
                            flushed = True
                        yield _sse_payload(
                            {
                                "id": completion_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": model_name,
                                "choices": [{"index": 0, "delta": {}, "finish_reason": _chat_stream_finish_reason(value or "stop")}],
                            }
                        )
                        continue
                    if kind == "error":
                        completed = True
                        _raise_request_error(request)
                    if kind == "done":
                        if not flushed:
                            for output_kind, output_text in _flush_chat_delta_filter(filter_state):
                                yield _sse_payload(
                                    {
                                        "id": completion_id,
                                        "object": "chat.completion.chunk",
                                        "created": created,
                                        "model": model_name,
                                        "choices": [
                                            {
                                                "index": 0,
                                                "delta": _chat_stream_delta_payload(
                                                    content=output_text if output_kind == "content" else None,
                                                    reasoning_content=output_text if output_kind == "reasoning_content" else None,
                                                ),
                                                "finish_reason": None,
                                            }
                                        ],
                                    }
                                )
                            flushed = True
                        completed = True
                        yield _sse_payload("[DONE]")
                        return
            finally:
                if not completed:
                    state.batcher.cancel(request)

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    assert request.done_event is not None
    await request.done_event.wait()
    _raise_request_error(request)
    result = request.result()
    content_text, reasoning_text = _filter_chat_text(result.text, mode=prompt_mode)
    return _json_response(
        {
            "id": completion_id,
            "object": "chat.completion",
            "created": created,
            "model": model_name,
            "choices": [
                {
                    "index": 0,
                    "message": _assistant_message_payload(content_text, reasoning_text),
                    "finish_reason": result.finish_reason,
                }
            ],
            "usage": _usage_dict(len(result.prompt_token_ids), len(result.completion_token_ids)),
        }
    )


def _prepare_request_from_payload(
    state: Any,
    *,
    endpoint: Literal["completion", "chat"],
    payload: dict[str, Any],
) -> PreparedOpenAIRequest:
    if endpoint == "completion":
        req = _parse_completion_request_payload(payload)
        return _prepare_completion_request(state, req)
    req = _parse_chat_request_payload(payload)
    return _prepare_chat_request(state, req)


def _build_prepared_queue_request(
    state: QueueFrontendState,
    *,
    endpoint: Literal["completion", "chat"],
    payload: dict[str, Any],
    request_id: str,
    created: int,
    stream: bool,
) -> dict[str, Any]:
    prepared = _prepare_request_from_payload(
        state,
        endpoint=endpoint,
        payload=payload,
    )
    return {
        "frontend_id": state.frontend_id,
        "kind": "request",
        "endpoint": endpoint,
        "request_id": request_id,
        "created": created,
        "stream": stream,
        "prepared": _serialize_prepared_request(prepared),
    }


def _create_server_state(
    *,
    model: str,
    served_model_name: str | None = None,
    api_key: str | None = None,
    llm_kwargs: dict[str, Any] | None = None,
) -> ServerState:
    llm = LLM(model, **(llm_kwargs or {}))
    return ServerState(
        llm=llm,
        model_id=served_model_name or _default_model_name(model),
        created=int(time.time()),
        api_key=api_key,
        lock=threading.Lock(),
        prompt_token_cache=PromptTokenCache(),
        batcher=RequestBatcher(llm),
    )


def create_app(
    *,
    model: str,
    served_model_name: str | None = None,
    api_key: str | None = None,
    llm_kwargs: dict[str, Any] | None = None,
    server_state: ServerState | None = None,
    manage_resources: bool = True,
    disable_cors: bool = False,
) -> FastAPI:
    state = server_state or _create_server_state(
        model=model,
        served_model_name=served_model_name,
        api_key=api_key,
        llm_kwargs=llm_kwargs,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if manage_resources:
            if app.state.server.batcher is not None:
                app.state.server.batcher.start()
            try:
                yield
            finally:
                if app.state.server.batcher is not None:
                    app.state.server.batcher.stop()
                app.state.server.llm.exit()
            return
        yield

    app = FastAPI(title="nano-vllm OpenAI-compatible API", lifespan=lifespan)
    _apply_disable_cors(app, disable_cors)
    app.state.server = state

    @app.middleware("http")
    async def _capture_request_received_at(request: Request, call_next):
        request.state.nanovllm_received_at = time.perf_counter()
        return await call_next(request)

    @app.exception_handler(OpenAIAPIError)
    async def _handle_openai_error(request: Request, exc: OpenAIAPIError):
        return _openai_error_response(exc)

    @app.exception_handler(RequestValidationError)
    async def _handle_validation_error(request: Request, exc: RequestValidationError):
        return _openai_error_response(
            OpenAIAPIError(400, f"Invalid request body: {exc.errors()}", param=None)
        )

    @app.get("/health")
    async def health():
        return {"status": "ok", "model": app.state.server.model_id}

    @app.get("/v1/models")
    async def list_models(authorization: str | None = Header(default=None)):
        _require_api_key(app.state.server, authorization)
        return _openai_models_response(app.state.server)

    @app.get("/v1/models/{model_id}")
    async def retrieve_model(model_id: str, authorization: str | None = Header(default=None)):
        _require_api_key(app.state.server, authorization)
        return _retrieve_openai_model_response(app.state.server, model_id)

    @app.post("/v1/completions")
    async def completions(
        request_http: Request,
        authorization: str | None = Header(default=None),
    ):
        http_received_at = getattr(
            request_http.state,
            "nanovllm_received_at",
            time.perf_counter(),
        )
        req = _parse_completion_request_payload(await _load_json_body(request_http))
        handler_started_at = time.perf_counter()
        return await _serve_completion_request(
            app.state.server,
            req,
            authorization=authorization,
            http_received_at=http_received_at,
            handler_started_at=handler_started_at,
        )

    @app.post("/v1/chat/completions")
    async def chat_completions(
        request_http: Request,
        authorization: str | None = Header(default=None),
    ):
        http_received_at = getattr(
            request_http.state,
            "nanovllm_received_at",
            time.perf_counter(),
        )
        payload = await _load_json_body(request_http)
        handler_started_at = time.perf_counter()
        if _is_lightning_private_payload(payload):
            return await _serve_lightning_private_chat(
                app.state.server,
                payload,
                authorization=authorization,
                http_received_at=http_received_at,
                handler_started_at=handler_started_at,
            )
        req = _parse_chat_request_payload(payload)
        return await _serve_chat_completion_request(
            app.state.server,
            req,
            authorization=authorization,
            http_received_at=http_received_at,
            handler_started_at=handler_started_at,
        )

    @app.post("/v2/chat/completions")
    async def lightning_v2_chat_completions(
        request_http: Request,
        authorization: str | None = Header(default=None),
    ):
        http_received_at = getattr(request_http.state, "nanovllm_received_at", time.perf_counter())
        payload = await _load_json_body(request_http)
        handler_started_at = time.perf_counter()
        return await _serve_lightning_private_chat(
            app.state.server,
            payload,
            authorization=authorization,
            http_received_at=http_received_at,
            handler_started_at=handler_started_at,
        )

    @app.post("/state/chat/completions")
    async def lightning_state_chat_completions(
        request_http: Request,
        authorization: str | None = Header(default=None),
    ):
        http_received_at = getattr(request_http.state, "nanovllm_received_at", time.perf_counter())
        payload = await _load_json_body(request_http)
        handler_started_at = time.perf_counter()
        session_id = payload.get("session_id")
        if not isinstance(session_id, str) or not session_id:
            raise OpenAIAPIError(400, "Missing session_id parameter.", param="session_id")
        return await _serve_lightning_private_chat(
            app.state.server,
            payload,
            authorization=authorization,
            http_received_at=http_received_at,
            handler_started_at=handler_started_at,
            state_session_id=session_id,
            force_single_prompt=True,
        )

    @app.post("/state/status")
    async def lightning_state_status(
        request_http: Request,
        authorization: str | None = Header(default=None),
    ):
        payload = await _load_json_body(request_http)
        _require_private_api_key(app.state.server, authorization, payload)
        sessions = _lightning_session_store(app.state.server)
        with _lightning_session_lock(app.state.server):
            session_ids = list(sessions.keys())
        detailed_states = [
            {
                "session_id": session_id,
                "cache_level": "In Memory",
                "last_updated": "In Memory",
                "timestamp": time.time(),
            }
            for session_id in session_ids
        ]
        return _json_response(
            {
                "status": "success",
                "total_sessions": len(session_ids),
                "l1_cache_count": len(session_ids),
                "l2_cache_count": 0,
                "database_count": 0,
                "sessions": detailed_states,
            }
        )

    @app.post("/state/delete")
    async def lightning_state_delete(
        request_http: Request,
        authorization: str | None = Header(default=None),
    ):
        payload = await _load_json_body(request_http)
        _require_private_api_key(app.state.server, authorization, payload)
        session_id = payload.get("session_id")
        if not isinstance(session_id, str) or not session_id:
            raise OpenAIAPIError(400, "Missing session_id parameter.", param="session_id")
        delete_prefix = bool(payload.get("delete_prefix", False))
        sessions = _lightning_session_store(app.state.server)
        deleted = False
        with _lightning_session_lock(app.state.server):
            if session_id in sessions:
                del sessions[session_id]
                deleted = True
            if delete_prefix:
                prefix = f"{session_id}:"
                for key in list(sessions.keys()):
                    if key.startswith(prefix):
                        del sessions[key]
                        deleted = True
        if deleted or delete_prefix:
            return _json_response({"status": "success", "message": f"Session {session_id} deleted successfully"})
        return _json_response(
            {"status": "not_found", "message": f"Session {session_id} not found in database"},
            status_code=404,
        )

    @app.post("/translate/v1/batch-translate")
    async def lightning_batch_translate(
        request_http: Request,
        authorization: str | None = Header(default=None),
    ):
        http_received_at = getattr(request_http.state, "nanovllm_received_at", time.perf_counter())
        payload = await _load_json_body(request_http)
        handler_started_at = time.perf_counter()
        return await _serve_lightning_translate(
            app.state.server,
            payload,
            authorization=authorization,
            http_received_at=http_received_at,
            handler_started_at=handler_started_at,
        )

    @app.post("/FIM/v1/batch-FIM")
    async def lightning_fim_completions(
        request_http: Request,
        authorization: str | None = Header(default=None),
    ):
        http_received_at = getattr(request_http.state, "nanovllm_received_at", time.perf_counter())
        payload = await _load_json_body(request_http)
        prefixes = _lightning_string_list(payload, "prefix")
        suffixes = _lightning_string_list(payload, "suffix")
        prompts = [f"✿prefix✿✿suffix✿{suffix}✿middle✿{prefix}" for prefix, suffix in zip(prefixes, suffixes)]
        handler_started_at = time.perf_counter()
        return await _serve_lightning_private_chat(
            app.state.server,
            {**payload, "contents": prompts, "stop_tokens": payload.get("stop_tokens", [])},
            authorization=authorization,
            http_received_at=http_received_at,
            handler_started_at=handler_started_at,
            response_object="FIM.completion",
        )

    @app.post("/openai/v1/chat/completions")
    async def lightning_openai_chat_completions(
        request_http: Request,
        authorization: str | None = Header(default=None),
    ):
        http_received_at = getattr(request_http.state, "nanovllm_received_at", time.perf_counter())
        payload = await _load_json_body(request_http)
        handler_started_at = time.perf_counter()
        return await _serve_lightning_openai_compat_chat(
            app.state.server,
            payload,
            authorization=authorization,
            http_received_at=http_received_at,
            handler_started_at=handler_started_at,
        )

    @app.post("/v1/tokenize")
    async def tokenize(
        request_http: Request,
        authorization: str | None = Header(default=None),
    ):
        _require_api_key(app.state.server, authorization)
        payload = await _load_json_body(request_http)
        return _json_response(_handle_tokenize_payload(app.state.server, payload))

    @app.post("/v1/detokenize")
    async def detokenize(
        request_http: Request,
        authorization: str | None = Header(default=None),
    ):
        _require_api_key(app.state.server, authorization)
        payload = await _load_json_body(request_http)
        return _json_response(_handle_detokenize_payload(app.state.server, payload))

    return app


def create_proxy_app(
    *,
    backend_uds: str,
    backend_max_connections: int,
    backend_keepalive_connections: int,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        transport = httpx.AsyncHTTPTransport(uds=backend_uds)
        limits = httpx.Limits(
            max_connections=backend_max_connections,
            max_keepalive_connections=backend_keepalive_connections,
        )
        timeout = httpx.Timeout(600.0, connect=10.0)
        app.state.backend_client = httpx.AsyncClient(
            base_url="http://nanovllm-backend",
            transport=transport,
            limits=limits,
            timeout=timeout,
        )
        try:
            yield
        finally:
            await app.state.backend_client.aclose()

    app = FastAPI(title="nano-vllm OpenAI proxy", lifespan=lifespan)

    @app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"])
    async def proxy(path: str, request: Request):
        client: httpx.AsyncClient = app.state.backend_client
        body = await request.body()
        headers = dict(request.headers)
        headers.pop("host", None)
        url = httpx.URL(path="/" + path, query=request.url.query.encode("utf-8"))
        backend_request = client.build_request(
            request.method,
            url,
            headers=headers,
            content=body,
        )
        try:
            backend_response = await client.send(backend_request, stream=True)
        except httpx.HTTPError as exc:
            return _json_response(
                {
                    "error": {
                        "message": f"Backend unavailable: {exc}",
                        "type": "server_error",
                        "param": None,
                        "code": "backend_unavailable",
                    }
                },
                status_code=503,
            )
        media_type = backend_response.headers.get("content-type")
        is_streaming = media_type is not None and media_type.startswith("text/event-stream")
        response_headers = _proxy_response_headers(backend_response.headers, streaming=is_streaming)
        if is_streaming:
            async def stream_backend():
                try:
                    async for chunk in backend_response.aiter_raw():
                        if chunk:
                            yield chunk
                finally:
                    await backend_response.aclose()

            return StreamingResponse(
                stream_backend(),
                status_code=backend_response.status_code,
                headers=response_headers,
                media_type=media_type,
            )

        content = await backend_response.aread()
        await backend_response.aclose()
        return Response(
            content=content,
            status_code=backend_response.status_code,
            headers=response_headers,
            media_type=media_type,
        )

    return app


async def _open_backend_connection(backend_uds: str) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
    last_exc: BaseException | None = None
    for attempt in range(8):
        try:
            return await asyncio.open_unix_connection(backend_uds)
        except OSError as exc:
            last_exc = exc
            if exc.errno not in (2, 11, 111):
                break
            await asyncio.sleep(min(0.001 * (2**attempt), 0.05))
        except Exception as exc:
            last_exc = exc
            break
    raise OpenAIAPIError(
        503,
        f"Backend unavailable: {last_exc}",
        error_type="server_error",
        code="backend_unavailable",
    ) from last_exc


async def _ipc_backend_handle_connection(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    state: ServerState,
):
    try:
        while True:
            try:
                frame = await _ipc_read_frame(reader)
            except asyncio.IncompleteReadError:
                break
            try:
                if frame is None:
                    break
                if frame.get("kind") == "ping":
                    await _ipc_write_frame(writer, {"kind": "pong"})
                    continue
                if frame.get("kind") != "request":
                    raise OpenAIAPIError(
                        400,
                        "Invalid IPC request frame.",
                        error_type="server_error",
                        code="bad_ipc_frame",
                    )

                endpoint = str(frame["endpoint"])
                stream = bool(frame.get("stream"))
                payload = frame.get("payload")
                if endpoint == "tokenize":
                    if not isinstance(payload, dict):
                        raise OpenAIAPIError(400, "Missing tokenize payload.", error_type="server_error", code="bad_ipc_frame")
                    await _ipc_write_frame(writer, {"kind": "tokenize_result", "body": _handle_tokenize_payload(state, payload)})
                    continue
                if endpoint == "detokenize":
                    if not isinstance(payload, dict):
                        raise OpenAIAPIError(400, "Missing detokenize payload.", error_type="server_error", code="bad_ipc_frame")
                    await _ipc_write_frame(writer, {"kind": "detokenize_result", "body": _handle_detokenize_payload(state, payload)})
                    continue
                if isinstance(payload, dict):
                    prepared = _prepare_request_from_payload(
                        state,
                        endpoint=endpoint,
                        payload=payload,
                    )
                else:
                    sampling_params = _sampling_params_from_payload(frame["sampling"])
                    if endpoint == "chat":
                        _base_model, mode = _validate_openai_request_model(
                            str(frame.get("model", state.model_id)),
                            state,
                        )
                        prompt_text = _render_openai_chat_prompt(
                            state.llm.tokenizer,
                            frame["messages"],
                            mode=mode,
                        )
                        prepared = PreparedOpenAIRequest(
                            prompt_text=prompt_text,
                            sampling_params=sampling_params,
                            requested_max_tokens=int(sampling_params.max_tokens),
                        )
                    else:
                        prepared = PreparedOpenAIRequest(
                            prompt_text=str(frame["prompt_text"]),
                            sampling_params=sampling_params,
                            requested_max_tokens=int(frame.get("requested_max_tokens", sampling_params.max_tokens)),
                            prompt_token_ids=frame.get("prompt_token_ids"),
                            capture_logprobs=bool(frame.get("capture_logprobs", False)),
                            top_logprobs=int(frame.get("top_logprobs", 0)),
                            echo=bool(frame.get("echo", False)),
                        )
                request_id = str(frame["request_id"])
                created = int(frame["created"])
                now = time.perf_counter()
                request = _submit_request(
                    state,
                    endpoint=endpoint,
                    prepared=prepared,
                    request_id=request_id,
                    created=created,
                    http_received_at=now,
                    handler_started_at=now,
                    stream=stream,
                )
                if stream:
                    assert request.ready_event is not None
                    await request.ready_event.wait()
                    _raise_request_error(request)
                    assert request.prompt_token_ids is not None
                    await _ipc_write_frame(
                        writer,
                        {
                            "kind": "start",
                            "prompt_token_count": len(request.prompt_token_ids),
                            "queue_wait_s": request.queue_wait_s,
                            "ttft_s": request.ttft_s,
                        },
                    )
                    finish_reason = "stop"
                    assert request.stream_queue is not None
                    while True:
                        kind, value = await request.stream_queue.get()
                        if kind == "delta":
                            await _ipc_write_frame(writer, {"kind": "delta", "text": value or ""})
                            continue
                        if kind == "finish":
                            finish_reason = value or "stop"
                            continue
                        if kind == "error":
                            _raise_request_error(request)
                        if kind == "done":
                            await _ipc_write_frame(
                                writer,
                                {
                                    "kind": "done",
                                    "finish_reason": finish_reason,
                                    "completion_token_count": len(request.completion_token_ids),
                                    "generation_s": 0.0 if request.generation_s is None else request.generation_s,
                                    "total_s": 0.0 if request.total_s is None else request.total_s,
                                },
                            )
                            break
                    continue

                assert request.done_event is not None
                await request.done_event.wait()
                _raise_request_error(request)
                result = request.result()
                await _ipc_write_frame(
                    writer,
                    {
                        "kind": "result",
                        "prompt_token_count": len(result.prompt_token_ids),
                        "completion_token_count": len(result.completion_token_ids),
                        "text": _completion_response_text(state.llm.tokenizer, request)
                        if endpoint == "completion"
                        else result.text,
                        "logprobs": _build_completion_logprobs(state.llm.tokenizer, request)
                        if endpoint == "completion"
                        else None,
                        "finish_reason": result.finish_reason,
                        "queue_wait_s": request.queue_wait_s,
                        "ttft_s": result.ttft_s,
                        "generation_s": result.generation_s,
                        "total_s": 0.0 if request.total_s is None else request.total_s,
                    },
                )
            except OpenAIAPIError as exc:
                await _ipc_write_frame(writer, {"kind": "error", "error": _serialize_openai_error(exc)})
            except BaseException as exc:
                await _ipc_write_frame(
                    writer,
                    {
                        "kind": "error",
                        "error": _serialize_openai_error(
                            OpenAIAPIError(
                                500,
                                f"Backend request failed: {type(exc).__name__}: {exc}",
                                error_type="server_error",
                                code="backend_failure",
                            )
                        ),
                    },
                )
    finally:
        writer.close()
        await writer.wait_closed()


async def _run_ipc_backend_async(
    *,
    model: str,
    served_model_name: str | None,
    api_key: str | None,
    llm_kwargs: dict[str, Any],
    backend_uds: str,
    backlog: int,
):
    if os.path.exists(backend_uds):
        os.remove(backend_uds)
    state = _create_server_state(
        model=model,
        served_model_name=served_model_name,
        api_key=api_key,
        llm_kwargs=llm_kwargs,
    )
    if state.batcher is not None:
        state.batcher.start()
    server = await asyncio.start_unix_server(
        lambda reader, writer: _ipc_backend_handle_connection(reader, writer, state),
        path=backend_uds,
        backlog=backlog,
    )
    try:
        async with server:
            await server.serve_forever()
    finally:
        server.close()
        await server.wait_closed()
        if state.batcher is not None:
            state.batcher.stop()
        state.llm.exit()
        if os.path.exists(backend_uds):
            os.remove(backend_uds)


def _run_ipc_backend_process(
    *,
    model: str,
    served_model_name: str | None,
    api_key: str | None,
    llm_kwargs: dict[str, Any],
    backend_uds: str,
    backlog: int,
):
    _enable_uvloop()
    asyncio.run(
        _run_ipc_backend_async(
            model=model,
            served_model_name=served_model_name,
            api_key=api_key,
            llm_kwargs=llm_kwargs,
            backend_uds=backend_uds,
            backlog=backlog,
        )
    )


def _run_ipc_frontend_process(
    *,
    model_id: str,
    created: int,
    api_key: str | None,
    backend_uds: str,
    backend_channel_count: int,
    public_socket: socket.socket,
    log_level: str,
    access_log: bool,
    backlog: int,
    timeout_keep_alive: int,
    disable_cors: bool,
    frontend_ready_event: Any | None = None,
):
    app = create_ipc_frontend_app(
        model_id=model_id,
        created=created,
        api_key=api_key,
        backend_uds=backend_uds,
        backend_channel_count=backend_channel_count,
        disable_cors=disable_cors,
    )
    _run_uvicorn_server(
        app,
        host=None,
        port=None,
        uds=None,
        log_level=log_level,
        access_log=access_log,
        backlog=backlog,
        timeout_keep_alive=timeout_keep_alive,
        sockets=[public_socket],
        ready_callback=(lambda: _set_ready_event(frontend_ready_event)) if frontend_ready_event is not None else None,
    )


def create_ipc_frontend_app(
    *,
    model_id: str,
    created: int,
    api_key: str | None,
    backend_uds: str,
    backend_channel_count: int,
    disable_cors: bool = False,
) -> FastAPI:
    state = IPCFrontendState(
        model_id=model_id,
        created=created,
        api_key=api_key,
        backend_uds=backend_uds,
        backend_channel_count=max(1, backend_channel_count),
        prompt_token_cache=PromptTokenCache(),
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        channels: list[tuple[asyncio.StreamReader, asyncio.StreamWriter]] = []
        backend_pool: asyncio.Queue[tuple[asyncio.StreamReader, asyncio.StreamWriter]] = asyncio.Queue()
        try:
            for _ in range(app.state.server.backend_channel_count):
                channel = await _open_backend_connection(app.state.server.backend_uds)
                channels.append(channel)
                backend_pool.put_nowait(channel)
            app.state.backend_pool = backend_pool
            app.state.backend_channels = channels
            yield
        finally:
            while not backend_pool.empty():
                try:
                    backend_pool.get_nowait()
                except asyncio.QueueEmpty:
                    break
            for _reader, writer in channels:
                writer.close()
            for _reader, writer in channels:
                try:
                    await writer.wait_closed()
                except Exception:
                    pass

    app = FastAPI(title="nano-vllm OpenAI IPC frontend", lifespan=lifespan)
    _apply_disable_cors(app, disable_cors)
    app.state.server = state

    async def _checkout_backend_channel() -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
        return await app.state.backend_pool.get()

    def _return_backend_channel(channel: tuple[asyncio.StreamReader, asyncio.StreamWriter]) -> None:
        app.state.backend_pool.put_nowait(channel)

    async def _release_backend_channel(channel: tuple[asyncio.StreamReader, asyncio.StreamWriter]) -> None:
        reader, writer = channel
        if writer.is_closing() or reader.at_eof():
            if not writer.is_closing():
                writer.close()
                try:
                    await writer.wait_closed()
                except Exception:
                    pass
            return
        _return_backend_channel(channel)

    @app.middleware("http")
    async def _capture_request_received_at(request: Request, call_next):
        request.state.nanovllm_received_at = time.perf_counter()
        return await call_next(request)

    @app.exception_handler(OpenAIAPIError)
    async def _handle_openai_error(request: Request, exc: OpenAIAPIError):
        return _openai_error_response(exc)

    @app.get("/health")
    async def health():
        return {"status": "ok", "model": app.state.server.model_id}

    @app.get("/v1/models")
    async def list_models(authorization: str | None = Header(default=None)):
        _require_api_key(app.state.server, authorization)
        return _openai_models_response(app.state.server)

    @app.get("/v1/models/{model_id}")
    async def retrieve_model(model_id: str, authorization: str | None = Header(default=None)):
        _require_api_key(app.state.server, authorization)
        return _retrieve_openai_model_response(app.state.server, model_id)

    @app.post("/v1/completions")
    async def completions(request_http: Request, authorization: str | None = Header(default=None)):
        http_received_at = getattr(request_http.state, "nanovllm_received_at", time.perf_counter())
        payload = await _load_json_body(request_http)
        handler_started_at = time.perf_counter()
        _require_api_key(app.state.server, authorization)
        stream = bool(_optional_bool(payload, "stream", False))
        stream_include_usage = _stream_options_include_usage(payload.get("stream_options"))
        created_at = int(time.time())
        completion_id = f"cmpl-{uuid.uuid4().hex}"
        submit_started_at = time.perf_counter()
        channel = await _checkout_backend_channel()
        reader, writer = channel
        try:
            await _ipc_write_frame(
                writer,
                {
                    "kind": "request",
                    "endpoint": "completion",
                    "request_id": completion_id,
                    "created": created_at,
                    "stream": stream,
                    "payload": payload,
                },
            )
            if stream:
                start_frame = await _ipc_read_frame(reader)
                if start_frame is None:
                    raise OpenAIAPIError(503, "Backend closed stream before start.", error_type="server_error")
                if start_frame.get("kind") == "error":
                    raise _deserialize_openai_error(start_frame["error"])
                first_ready_at = time.perf_counter()
                headers = _response_headers(
                    request_id=completion_id,
                    prompt_token_count=int(start_frame["prompt_token_count"]),
                    queue_wait_s=float(start_frame["queue_wait_s"]),
                    processing_s=first_ready_at - submit_started_at,
                    request_parse_s=handler_started_at - http_received_at,
                    request_setup_s=submit_started_at - handler_started_at,
                    response_build_s=0.0,
                    server_app_s=first_ready_at - http_received_at,
                    ttft_s=start_frame.get("ttft_s"),
                    streaming=True,
                )

                async def event_stream():
                    try:
                        while True:
                            frame = await _ipc_read_frame(reader)
                            if frame is None:
                                return
                            kind = frame.get("kind")
                            if kind == "delta":
                                yield _sse_payload(
                                    {
                                        "id": completion_id,
                                        "object": "text_completion",
                                        "created": created_at,
                                        "model": app.state.server.model_id,
                                        "choices": [
                                            {
                                                "index": 0,
                                                "text": frame.get("text") or "",
                                                "finish_reason": None,
                                                "logprobs": None,
                                            }
                                        ],
                                    }
                                )
                                continue
                            if kind == "done":
                                yield _sse_payload(
                                    {
                                        "id": completion_id,
                                        "object": "text_completion",
                                        "created": created_at,
                                        "model": app.state.server.model_id,
                                        "choices": [
                                            {
                                                "index": 0,
                                                "text": "",
                                                "finish_reason": frame.get("finish_reason"),
                                                "logprobs": None,
                                            }
                                        ],
                                    }
                                )
                                if stream_include_usage:
                                    yield _sse_payload(
                                        _completion_stream_usage_event(
                                            completion_id=completion_id,
                                            created=created_at,
                                            model=app.state.server.model_id,
                                            prompt_token_count=int(start_frame["prompt_token_count"]),
                                            completion_token_count=int(frame["completion_token_count"]),
                                        )
                                    )
                                yield _sse_payload("[DONE]")
                                return
                            if kind == "error":
                                raise _deserialize_openai_error(frame["error"])
                    finally:
                        await _release_backend_channel(channel)

                return StreamingResponse(event_stream(), media_type="text/event-stream", headers=headers)

            frame = await _ipc_read_frame(reader)
        except Exception:
            await _release_backend_channel(channel)
            raise
        await _release_backend_channel(channel)
        if frame is None:
            raise OpenAIAPIError(503, "Backend closed connection without a result.", error_type="server_error")
        if frame.get("kind") == "error":
            raise _deserialize_openai_error(frame["error"])
        backend_finished_at = time.perf_counter()
        response_built_at = time.perf_counter()
        processing_s = backend_finished_at - submit_started_at
        headers = _response_headers(
            request_id=completion_id,
            prompt_token_count=int(frame["prompt_token_count"]),
            completion_token_count=int(frame["completion_token_count"]),
            queue_wait_s=float(frame["queue_wait_s"]),
            processing_s=processing_s,
            request_parse_s=handler_started_at - http_received_at,
            request_setup_s=submit_started_at - handler_started_at,
            response_build_s=response_built_at - backend_finished_at,
            server_app_s=response_built_at - http_received_at,
            ttft_s=frame.get("ttft_s"),
            generation_s=float(frame["generation_s"]),
            total_s=processing_s,
            streaming=False,
        )
        return _json_response(
            {
                "id": completion_id,
                "object": "text_completion",
                "created": created_at,
                "model": app.state.server.model_id,
                "choices": [
                    {
                        "index": 0,
                        "text": frame["text"],
                        "finish_reason": frame["finish_reason"],
                        "logprobs": frame.get("logprobs"),
                    }
                ],
                "usage": _usage_dict(int(frame["prompt_token_count"]), int(frame["completion_token_count"])),
            },
            headers=headers,
        )

    @app.post("/v1/chat/completions")
    async def chat_completions(request_http: Request, authorization: str | None = Header(default=None)):
        http_received_at = getattr(request_http.state, "nanovllm_received_at", time.perf_counter())
        payload = await _load_json_body(request_http)
        handler_started_at = time.perf_counter()
        _require_api_key(app.state.server, authorization)
        stream = bool(_optional_bool(payload, "stream", False))
        stream_include_usage = _stream_options_include_usage(payload.get("stream_options"))
        created_at = int(time.time())
        completion_id = f"chatcmpl-{uuid.uuid4().hex}"
        submit_started_at = time.perf_counter()
        channel = await _checkout_backend_channel()
        reader, writer = channel
        try:
            await _ipc_write_frame(
                writer,
                {
                    "kind": "request",
                    "endpoint": "chat",
                    "request_id": completion_id,
                    "created": created_at,
                    "stream": stream,
                    "payload": payload,
                },
            )
            if stream:
                start_frame = await _ipc_read_frame(reader)
                if start_frame is None:
                    raise OpenAIAPIError(503, "Backend closed stream before start.", error_type="server_error")
                if start_frame.get("kind") == "error":
                    raise _deserialize_openai_error(start_frame["error"])
                first_ready_at = time.perf_counter()
                headers = _response_headers(
                    request_id=completion_id,
                    prompt_token_count=int(start_frame["prompt_token_count"]),
                    queue_wait_s=float(start_frame["queue_wait_s"]),
                    processing_s=first_ready_at - submit_started_at,
                    request_parse_s=handler_started_at - http_received_at,
                    request_setup_s=submit_started_at - handler_started_at,
                    response_build_s=0.0,
                    server_app_s=first_ready_at - http_received_at,
                    ttft_s=start_frame.get("ttft_s"),
                    streaming=True,
                )

                async def event_stream():
                    filter_state = _ChatOutputFilter()
                    flushed = False
                    try:
                        yield _sse_payload(
                            {
                                "id": completion_id,
                                "object": "chat.completion.chunk",
                                "created": created_at,
                                "model": app.state.server.model_id,
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": _chat_stream_delta_payload(role="assistant", content=""),
                                        "finish_reason": None,
                                    }
                                ],
                            }
                        )
                        while True:
                            frame = await _ipc_read_frame(reader)
                            if frame is None:
                                return
                            kind = frame.get("kind")
                            if kind == "delta":
                                for output_kind, output_text in _filter_chat_delta(filter_state, frame.get("text") or ""):
                                    yield _sse_payload(
                                        {
                                            "id": completion_id,
                                            "object": "chat.completion.chunk",
                                            "created": created_at,
                                            "model": app.state.server.model_id,
                                            "choices": [
                                                {
                                                    "index": 0,
                                                    "delta": _chat_stream_delta_payload(
                                                        content=output_text if output_kind == "content" else None,
                                                        reasoning_content=output_text if output_kind == "reasoning_content" else None,
                                                    ),
                                                    "finish_reason": None,
                                                }
                                            ],
                                        }
                                    )
                                continue
                            if kind == "done":
                                if not flushed:
                                    for output_kind, output_text in _flush_chat_delta_filter(filter_state):
                                        yield _sse_payload(
                                            {
                                                "id": completion_id,
                                                "object": "chat.completion.chunk",
                                                "created": created_at,
                                                "model": app.state.server.model_id,
                                                "choices": [
                                                    {
                                                        "index": 0,
                                                        "delta": _chat_stream_delta_payload(
                                                            content=output_text if output_kind == "content" else None,
                                                            reasoning_content=output_text if output_kind == "reasoning_content" else None,
                                                        ),
                                                        "finish_reason": None,
                                                    }
                                                ],
                                            }
                                        )
                                    flushed = True
                                yield _sse_payload(
                                    {
                                        "id": completion_id,
                                        "object": "chat.completion.chunk",
                                        "created": created_at,
                                        "model": app.state.server.model_id,
                                        "choices": [
                                            {
                                                "index": 0,
                                                "delta": {},
                                                "finish_reason": _chat_stream_finish_reason(frame.get("finish_reason") or "stop"),
                                            }
                                        ],
                                    }
                                )
                                if stream_include_usage:
                                    yield _sse_payload(
                                        _chat_stream_usage_event(
                                            completion_id=completion_id,
                                            created=created_at,
                                            model=app.state.server.model_id,
                                            prompt_token_count=int(start_frame["prompt_token_count"]),
                                            completion_token_count=int(frame["completion_token_count"]),
                                        )
                                    )
                                yield _sse_payload("[DONE]")
                                return
                            if kind == "error":
                                raise _deserialize_openai_error(frame["error"])
                    finally:
                        await _release_backend_channel(channel)

                return StreamingResponse(event_stream(), media_type="text/event-stream", headers=headers)

            frame = await _ipc_read_frame(reader)
        except Exception:
            await _release_backend_channel(channel)
            raise
        await _release_backend_channel(channel)
        if frame is None:
            raise OpenAIAPIError(503, "Backend closed connection without a result.", error_type="server_error")
        if frame.get("kind") == "error":
            raise _deserialize_openai_error(frame["error"])
        content_text, reasoning_text = _filter_chat_text(frame["text"])
        backend_finished_at = time.perf_counter()
        response_built_at = time.perf_counter()
        processing_s = backend_finished_at - submit_started_at
        headers = _response_headers(
            request_id=completion_id,
            prompt_token_count=int(frame["prompt_token_count"]),
            completion_token_count=int(frame["completion_token_count"]),
            queue_wait_s=float(frame["queue_wait_s"]),
            processing_s=processing_s,
            request_parse_s=handler_started_at - http_received_at,
            request_setup_s=submit_started_at - handler_started_at,
            response_build_s=response_built_at - backend_finished_at,
            server_app_s=response_built_at - http_received_at,
            ttft_s=frame.get("ttft_s"),
            generation_s=float(frame["generation_s"]),
            total_s=processing_s,
            streaming=False,
        )
        return _json_response(
            {
                "id": completion_id,
                "object": "chat.completion",
                "created": created_at,
                "model": app.state.server.model_id,
                "choices": [
                    {
                        "index": 0,
                        "message": _assistant_message_payload(content_text, reasoning_text),
                        "finish_reason": frame["finish_reason"],
                    }
                ],
                "usage": _usage_dict(int(frame["prompt_token_count"]), int(frame["completion_token_count"])),
            },
            headers=headers,
        )

    @app.post("/v1/tokenize")
    async def tokenize(request_http: Request, authorization: str | None = Header(default=None)):
        _require_api_key(app.state.server, authorization)
        payload = await _load_json_body(request_http)
        request_id = f"tok-{uuid.uuid4().hex}"
        channel = await _checkout_backend_channel()
        reader, writer = channel
        try:
            await _ipc_write_frame(
                writer,
                {
                    "kind": "request",
                    "endpoint": "tokenize",
                    "request_id": request_id,
                    "created": int(time.time()),
                    "stream": False,
                    "payload": payload,
                },
            )
            frame = await _ipc_read_frame(reader)
        except Exception:
            await _release_backend_channel(channel)
            raise
        await _release_backend_channel(channel)
        if frame is None:
            raise OpenAIAPIError(503, "Backend closed connection without a result.", error_type="server_error")
        if frame.get("kind") == "error":
            raise _deserialize_openai_error(frame["error"])
        return _json_response(frame["body"])

    @app.post("/v1/detokenize")
    async def detokenize(request_http: Request, authorization: str | None = Header(default=None)):
        _require_api_key(app.state.server, authorization)
        payload = await _load_json_body(request_http)
        request_id = f"detok-{uuid.uuid4().hex}"
        channel = await _checkout_backend_channel()
        reader, writer = channel
        try:
            await _ipc_write_frame(
                writer,
                {
                    "kind": "request",
                    "endpoint": "detokenize",
                    "request_id": request_id,
                    "created": int(time.time()),
                    "stream": False,
                    "payload": payload,
                },
            )
            frame = await _ipc_read_frame(reader)
        except Exception:
            await _release_backend_channel(channel)
            raise
        await _release_backend_channel(channel)
        if frame is None:
            raise OpenAIAPIError(503, "Backend closed connection without a result.", error_type="server_error")
        if frame.get("kind") == "error":
            raise _deserialize_openai_error(frame["error"])
        return _json_response(frame["body"])

    return app


async def _queue_backend_handle_message(
    message: dict[str, Any],
    state: ServerState,
    response_queues: list[Any],
):
    frontend_id = int(message["frontend_id"])
    response_queue = response_queues[frontend_id]
    request_id = str(message["request_id"])

    def send_frame(payload: dict[str, Any]) -> None:
        _queue_put_frame(response_queue, request_id, payload)

    try:
        endpoint = str(message["endpoint"])
        stream = bool(message.get("stream"))
        if endpoint == "tokenize":
            send_frame({"kind": "tokenize_result", "body": _handle_tokenize_payload(state, message["payload"])})
            return
        if endpoint == "detokenize":
            send_frame({"kind": "detokenize_result", "body": _handle_detokenize_payload(state, message["payload"])})
            return
        if "prepared" in message:
            prepared = _deserialize_prepared_request(message["prepared"])
        else:
            prepared = _prepare_request_from_payload(
                state,
                endpoint=endpoint,
                payload=message["payload"],
            )
        created = int(message["created"])
        now = time.perf_counter()
        request = _submit_request(
            state,
            endpoint=endpoint,
            prepared=prepared,
            request_id=request_id,
            created=created,
            http_received_at=now,
            handler_started_at=now,
            stream=stream,
        )
        if stream:
            assert request.ready_event is not None
            await request.ready_event.wait()
            _raise_request_error(request)
            assert request.prompt_token_ids is not None
            send_frame(
                {
                    "kind": "start",
                    "prompt_token_count": len(request.prompt_token_ids),
                    "queue_wait_s": request.queue_wait_s,
                    "ttft_s": request.ttft_s,
                }
            )
            finish_reason = "stop"
            assert request.stream_queue is not None
            while True:
                kind, value = await request.stream_queue.get()
                if kind == "delta":
                    send_frame({"kind": "delta", "text": value or ""})
                    continue
                if kind == "finish":
                    finish_reason = value or "stop"
                    continue
                if kind == "error":
                    _raise_request_error(request)
                if kind == "done":
                    send_frame(
                        {
                            "kind": "done",
                            "finish_reason": finish_reason,
                            "completion_token_count": len(request.completion_token_ids),
                            "generation_s": 0.0 if request.generation_s is None else request.generation_s,
                            "total_s": 0.0 if request.total_s is None else request.total_s,
                        }
                    )
                    break
            return

        assert request.done_event is not None
        await request.done_event.wait()
        _raise_request_error(request)
        result = request.result()
        send_frame(
            {
                "kind": "result",
                "prompt_token_count": len(result.prompt_token_ids),
                "completion_token_count": len(result.completion_token_ids),
                "text": _completion_response_text(state.llm.tokenizer, request)
                if endpoint == "completion"
                else result.text,
                "logprobs": _build_completion_logprobs(state.llm.tokenizer, request)
                if endpoint == "completion"
                else None,
                "finish_reason": result.finish_reason,
                "queue_wait_s": request.queue_wait_s,
                "ttft_s": result.ttft_s,
                "generation_s": result.generation_s,
                "total_s": 0.0 if request.total_s is None else request.total_s,
            }
        )
    except OpenAIAPIError as exc:
        send_frame({"kind": "error", "error": _serialize_openai_error(exc)})
    except BaseException as exc:
        send_frame(
            {
                "kind": "error",
                "error": _serialize_openai_error(
                    OpenAIAPIError(
                        500,
                        f"Backend request failed: {type(exc).__name__}: {exc}",
                        error_type="server_error",
                        code="backend_failure",
                    )
                ),
            }
        )


async def _run_queue_backend_async(
    *,
    model: str,
    served_model_name: str | None,
    api_key: str | None,
    llm_kwargs: dict[str, Any],
    request_queues: list[Any],
    response_queues: list[Any],
    ready_event: Any,
):
    state = _create_server_state(
        model=model,
        served_model_name=served_model_name,
        api_key=api_key,
        llm_kwargs=llm_kwargs,
    )
    if state.batcher is not None:
        state.batcher.start()
    loop = asyncio.get_running_loop()
    inbound: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
    active_tasks: set[asyncio.Task[Any]] = set()

    def inbound_bridge(request_queue: Any):
        while True:
            try:
                message = request_queue.get()
            except (EOFError, OSError):
                loop.call_soon_threadsafe(inbound.put_nowait, None)
                return
            loop.call_soon_threadsafe(inbound.put_nowait, message)
            if message is None:
                return

    bridge_threads: list[threading.Thread] = []
    for frontend_id, request_queue in enumerate(request_queues):
        thread = threading.Thread(
            target=inbound_bridge,
            args=(request_queue,),
            name=f"nanovllm-openai-backend-request-bridge-{frontend_id}",
            daemon=True,
        )
        thread.start()
        bridge_threads.append(thread)

    ready_event.set()
    try:
        while True:
            message = await inbound.get()
            if message is None:
                break
            task = asyncio.create_task(_queue_backend_handle_message(message, state, response_queues))
            active_tasks.add(task)
            task.add_done_callback(active_tasks.discard)
    finally:
        if active_tasks:
            await asyncio.gather(*active_tasks, return_exceptions=True)
        if state.batcher is not None:
            state.batcher.stop()
        state.llm.exit()


def _run_queue_backend_process(
    *,
    model: str,
    served_model_name: str | None,
    api_key: str | None,
    llm_kwargs: dict[str, Any],
    request_queues: list[Any],
    response_queues: list[Any],
    ready_event: Any,
):
    _enable_uvloop()
    asyncio.run(
        _run_queue_backend_async(
            model=model,
            served_model_name=served_model_name,
            api_key=api_key,
            llm_kwargs=llm_kwargs,
            request_queues=request_queues,
            response_queues=response_queues,
            ready_event=ready_event,
        )
    )


def create_queue_frontend_app(
    *,
    model_id: str,
    created: int,
    api_key: str | None,
    frontend_id: int,
    request_queue: Any,
    response_queue: Any,
    disable_cors: bool = False,
) -> FastAPI:
    state = QueueFrontendState(
        model_id=model_id,
        created=created,
        api_key=api_key,
        frontend_id=frontend_id,
        request_queue=request_queue,
        response_queue=response_queue,
        tokenizer=get_rwkv_tokenizer(),
        prompt_token_cache=PromptTokenCache(),
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        bridge = FrontendResponseBridge(app.state.server.response_queue)
        bridge.start()
        app.state.response_bridge = bridge
        try:
            yield
        finally:
            bridge.stop()

    app = FastAPI(title="nano-vllm OpenAI queue frontend", lifespan=lifespan)
    _apply_disable_cors(app, disable_cors)
    app.state.server = state

    @app.middleware("http")
    async def _capture_request_received_at(request: Request, call_next):
        request.state.nanovllm_received_at = time.perf_counter()
        return await call_next(request)

    @app.exception_handler(OpenAIAPIError)
    async def _handle_openai_error(request: Request, exc: OpenAIAPIError):
        return _openai_error_response(exc)

    @app.get("/health")
    async def health():
        return {"status": "ok", "model": app.state.server.model_id}

    @app.get("/v1/models")
    async def list_models(authorization: str | None = Header(default=None)):
        _require_api_key(app.state.server, authorization)
        return _openai_models_response(app.state.server)

    @app.get("/v1/models/{model_id}")
    async def retrieve_model(model_id: str, authorization: str | None = Header(default=None)):
        _require_api_key(app.state.server, authorization)
        return _retrieve_openai_model_response(app.state.server, model_id)

    @app.post("/v1/completions")
    async def completions(request_http: Request, authorization: str | None = Header(default=None)):
        http_received_at = getattr(request_http.state, "nanovllm_received_at", time.perf_counter())
        payload = await _load_json_body(request_http)
        handler_started_at = time.perf_counter()
        _require_api_key(app.state.server, authorization)
        stream = bool(_optional_bool(payload, "stream", False))
        stream_include_usage = _stream_options_include_usage(payload.get("stream_options"))
        created_at = int(time.time())
        completion_id = f"cmpl-{uuid.uuid4().hex}"
        submit_started_at = time.perf_counter()
        frame_queue = app.state.response_bridge.register(completion_id)
        try:
            app.state.server.request_queue.put(
                _build_prepared_queue_request(
                    app.state.server,
                    endpoint="completion",
                    payload=payload,
                    request_id=completion_id,
                    created=created_at,
                    stream=stream,
                )
            )
            if stream:
                start_frame = await frame_queue.get()
                if start_frame.get("kind") == "error":
                    raise _deserialize_openai_error(start_frame["error"])
                first_ready_at = time.perf_counter()
                headers = _response_headers(
                    request_id=completion_id,
                    prompt_token_count=int(start_frame["prompt_token_count"]),
                    queue_wait_s=float(start_frame["queue_wait_s"]),
                    processing_s=first_ready_at - submit_started_at,
                    request_parse_s=handler_started_at - http_received_at,
                    request_setup_s=submit_started_at - handler_started_at,
                    response_build_s=0.0,
                    server_app_s=first_ready_at - http_received_at,
                    ttft_s=start_frame.get("ttft_s"),
                    streaming=True,
                )

                async def event_stream():
                    try:
                        while True:
                            frame = await frame_queue.get()
                            kind = frame.get("kind")
                            if kind == "delta":
                                yield _sse_payload(
                                    {
                                        "id": completion_id,
                                        "object": "text_completion",
                                        "created": created_at,
                                        "model": app.state.server.model_id,
                                        "choices": [
                                            {
                                                "index": 0,
                                                "text": frame.get("text") or "",
                                                "finish_reason": None,
                                                "logprobs": None,
                                            }
                                        ],
                                    }
                                )
                                continue
                            if kind == "done":
                                yield _sse_payload(
                                    {
                                        "id": completion_id,
                                        "object": "text_completion",
                                        "created": created_at,
                                        "model": app.state.server.model_id,
                                        "choices": [
                                            {
                                                "index": 0,
                                                "text": "",
                                                "finish_reason": frame.get("finish_reason"),
                                                "logprobs": None,
                                            }
                                        ],
                                    }
                                )
                                if stream_include_usage:
                                    yield _sse_payload(
                                        _completion_stream_usage_event(
                                            completion_id=completion_id,
                                            created=created_at,
                                            model=app.state.server.model_id,
                                            prompt_token_count=int(start_frame["prompt_token_count"]),
                                            completion_token_count=int(frame["completion_token_count"]),
                                        )
                                    )
                                yield _sse_payload("[DONE]")
                                return
                            if kind == "error":
                                raise _deserialize_openai_error(frame["error"])
                    finally:
                        app.state.response_bridge.unregister(completion_id)

                return StreamingResponse(event_stream(), media_type="text/event-stream", headers=headers)

            frame = await frame_queue.get()
        finally:
            if not stream:
                app.state.response_bridge.unregister(completion_id)
        if frame.get("kind") == "error":
            raise _deserialize_openai_error(frame["error"])
        backend_finished_at = time.perf_counter()
        response_built_at = time.perf_counter()
        processing_s = backend_finished_at - submit_started_at
        headers = _response_headers(
            request_id=completion_id,
            prompt_token_count=int(frame["prompt_token_count"]),
            completion_token_count=int(frame["completion_token_count"]),
            queue_wait_s=float(frame["queue_wait_s"]),
            processing_s=processing_s,
            request_parse_s=handler_started_at - http_received_at,
            request_setup_s=submit_started_at - handler_started_at,
            response_build_s=response_built_at - backend_finished_at,
            server_app_s=response_built_at - http_received_at,
            ttft_s=frame.get("ttft_s"),
            generation_s=float(frame["generation_s"]),
            total_s=processing_s,
            streaming=False,
        )
        return _json_response(
            {
                "id": completion_id,
                "object": "text_completion",
                "created": created_at,
                "model": app.state.server.model_id,
                "choices": [
                    {
                        "index": 0,
                        "text": frame["text"],
                        "finish_reason": frame["finish_reason"],
                        "logprobs": frame.get("logprobs"),
                    }
                ],
                "usage": _usage_dict(int(frame["prompt_token_count"]), int(frame["completion_token_count"])),
            },
            headers=headers,
        )

    @app.post("/v1/chat/completions")
    async def chat_completions(request_http: Request, authorization: str | None = Header(default=None)):
        http_received_at = getattr(request_http.state, "nanovllm_received_at", time.perf_counter())
        payload = await _load_json_body(request_http)
        handler_started_at = time.perf_counter()
        _require_api_key(app.state.server, authorization)
        stream = bool(_optional_bool(payload, "stream", False))
        stream_include_usage = _stream_options_include_usage(payload.get("stream_options"))
        created_at = int(time.time())
        completion_id = f"chatcmpl-{uuid.uuid4().hex}"
        submit_started_at = time.perf_counter()
        frame_queue = app.state.response_bridge.register(completion_id)
        try:
            app.state.server.request_queue.put(
                _build_prepared_queue_request(
                    app.state.server,
                    endpoint="chat",
                    payload=payload,
                    request_id=completion_id,
                    created=created_at,
                    stream=stream,
                )
            )
            if stream:
                start_frame = await frame_queue.get()
                if start_frame.get("kind") == "error":
                    raise _deserialize_openai_error(start_frame["error"])
                first_ready_at = time.perf_counter()
                headers = _response_headers(
                    request_id=completion_id,
                    prompt_token_count=int(start_frame["prompt_token_count"]),
                    queue_wait_s=float(start_frame["queue_wait_s"]),
                    processing_s=first_ready_at - submit_started_at,
                    request_parse_s=handler_started_at - http_received_at,
                    request_setup_s=submit_started_at - handler_started_at,
                    response_build_s=0.0,
                    server_app_s=first_ready_at - http_received_at,
                    ttft_s=start_frame.get("ttft_s"),
                    streaming=True,
                )

                async def event_stream():
                    filter_state = _ChatOutputFilter()
                    flushed = False
                    try:
                        yield _sse_payload(
                            {
                                "id": completion_id,
                                "object": "chat.completion.chunk",
                                "created": created_at,
                                "model": app.state.server.model_id,
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": _chat_stream_delta_payload(role="assistant", content=""),
                                        "finish_reason": None,
                                    }
                                ],
                            }
                        )
                        while True:
                            frame = await frame_queue.get()
                            kind = frame.get("kind")
                            if kind == "delta":
                                for output_kind, output_text in _filter_chat_delta(filter_state, frame.get("text") or ""):
                                    yield _sse_payload(
                                        {
                                            "id": completion_id,
                                            "object": "chat.completion.chunk",
                                            "created": created_at,
                                            "model": app.state.server.model_id,
                                            "choices": [
                                                {
                                                    "index": 0,
                                                    "delta": _chat_stream_delta_payload(
                                                        content=output_text if output_kind == "content" else None,
                                                        reasoning_content=output_text if output_kind == "reasoning_content" else None,
                                                    ),
                                                    "finish_reason": None,
                                                }
                                            ],
                                        }
                                    )
                                continue
                            if kind == "done":
                                if not flushed:
                                    for output_kind, output_text in _flush_chat_delta_filter(filter_state):
                                        yield _sse_payload(
                                            {
                                                "id": completion_id,
                                                "object": "chat.completion.chunk",
                                                "created": created_at,
                                                "model": app.state.server.model_id,
                                                "choices": [
                                                    {
                                                        "index": 0,
                                                        "delta": _chat_stream_delta_payload(
                                                            content=output_text if output_kind == "content" else None,
                                                            reasoning_content=output_text if output_kind == "reasoning_content" else None,
                                                        ),
                                                        "finish_reason": None,
                                                    }
                                                ],
                                            }
                                        )
                                    flushed = True
                                yield _sse_payload(
                                    {
                                        "id": completion_id,
                                        "object": "chat.completion.chunk",
                                        "created": created_at,
                                        "model": app.state.server.model_id,
                                        "choices": [
                                            {
                                                "index": 0,
                                                "delta": {},
                                                "finish_reason": _chat_stream_finish_reason(frame.get("finish_reason") or "stop"),
                                            }
                                        ],
                                    }
                                )
                                if stream_include_usage:
                                    yield _sse_payload(
                                        _chat_stream_usage_event(
                                            completion_id=completion_id,
                                            created=created_at,
                                            model=app.state.server.model_id,
                                            prompt_token_count=int(start_frame["prompt_token_count"]),
                                            completion_token_count=int(frame["completion_token_count"]),
                                        )
                                    )
                                yield _sse_payload("[DONE]")
                                return
                            if kind == "error":
                                raise _deserialize_openai_error(frame["error"])
                    finally:
                        app.state.response_bridge.unregister(completion_id)

                return StreamingResponse(event_stream(), media_type="text/event-stream", headers=headers)

            frame = await frame_queue.get()
        finally:
            if not stream:
                app.state.response_bridge.unregister(completion_id)
        if frame.get("kind") == "error":
            raise _deserialize_openai_error(frame["error"])
        content_text, reasoning_text = _filter_chat_text(frame["text"])
        backend_finished_at = time.perf_counter()
        response_built_at = time.perf_counter()
        processing_s = backend_finished_at - submit_started_at
        headers = _response_headers(
            request_id=completion_id,
            prompt_token_count=int(frame["prompt_token_count"]),
            completion_token_count=int(frame["completion_token_count"]),
            queue_wait_s=float(frame["queue_wait_s"]),
            processing_s=processing_s,
            request_parse_s=handler_started_at - http_received_at,
            request_setup_s=submit_started_at - handler_started_at,
            response_build_s=response_built_at - backend_finished_at,
            server_app_s=response_built_at - http_received_at,
            ttft_s=frame.get("ttft_s"),
            generation_s=float(frame["generation_s"]),
            total_s=processing_s,
            streaming=False,
        )
        return _json_response(
            {
                "id": completion_id,
                "object": "chat.completion",
                "created": created_at,
                "model": app.state.server.model_id,
                "choices": [
                    {
                        "index": 0,
                        "message": _assistant_message_payload(content_text, reasoning_text),
                        "finish_reason": frame["finish_reason"],
                    }
                ],
                "usage": _usage_dict(int(frame["prompt_token_count"]), int(frame["completion_token_count"])),
            },
            headers=headers,
        )

    @app.post("/v1/tokenize")
    async def tokenize(request_http: Request, authorization: str | None = Header(default=None)):
        _require_api_key(app.state.server, authorization)
        payload = await _load_json_body(request_http)
        return _json_response(_handle_tokenize_payload(app.state.server, payload))

    @app.post("/v1/detokenize")
    async def detokenize(request_http: Request, authorization: str | None = Header(default=None)):
        _require_api_key(app.state.server, authorization)
        payload = await _load_json_body(request_http)
        return _json_response(_handle_detokenize_payload(app.state.server, payload))

    return app


def _aiohttp_json_response(
    content: Any,
    *,
    status: int = 200,
    headers: dict[str, str] | None = None,
) -> web.Response:
    return web.Response(
        status=status,
        headers=headers,
        body=_json_encode(content),
        content_type="application/json",
    )


def create_queue_frontend_aiohttp_app(
    *,
    model_id: str,
    created: int,
    api_key: str | None,
    frontend_id: int,
    request_queue: Any,
    response_queue: Any,
) -> web.Application:
    state = QueueFrontendState(
        model_id=model_id,
        created=created,
        api_key=api_key,
        frontend_id=frontend_id,
        request_queue=request_queue,
        response_queue=response_queue,
        tokenizer=get_rwkv_tokenizer(),
        prompt_token_cache=PromptTokenCache(),
    )

    @web.middleware
    async def capture_received_at(request: web.Request, handler):
        request["nanovllm_received_at"] = time.perf_counter()
        try:
            return await handler(request)
        except OpenAIAPIError as exc:
            return _aiohttp_json_response(
                {
                    "error": {
                        "message": exc.message,
                        "type": exc.error_type,
                        "param": exc.param,
                        "code": exc.code,
                    }
                },
                status=exc.status_code,
            )

    app = web.Application(middlewares=[capture_received_at])
    app["server"] = state

    async def on_startup(_app: web.Application):
        bridge = FrontendResponseBridge(state.response_queue)
        bridge.start()
        _app["response_bridge"] = bridge

    async def on_cleanup(_app: web.Application):
        bridge = _app.get("response_bridge")
        if bridge is not None:
            bridge.stop()

    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)

    async def health(_request: web.Request):
        return _aiohttp_json_response({"status": "ok", "model": state.model_id})

    async def list_models(request: web.Request):
        _require_api_key(state, request.headers.get("Authorization"))
        return _aiohttp_json_response(_openai_models_response(state))

    async def retrieve_model(request: web.Request):
        _require_api_key(state, request.headers.get("Authorization"))
        return _aiohttp_json_response(_retrieve_openai_model_response(state, request.match_info["model_id"]))

    async def completions(request: web.Request):
        http_received_at = request["nanovllm_received_at"]
        payload = _decode_json_body(await request.read())
        handler_started_at = time.perf_counter()
        _require_api_key(state, request.headers.get("Authorization"))
        response_model = str(payload.get("model", state.model_id))
        stream = bool(_optional_bool(payload, "stream", False))
        stream_include_usage = _stream_options_include_usage(payload.get("stream_options"))
        created_at = int(time.time())
        completion_id = f"cmpl-{uuid.uuid4().hex}"
        submit_started_at = time.perf_counter()
        response_bridge: FrontendResponseBridge = request.app["response_bridge"]
        frame_queue = response_bridge.register(completion_id)
        try:
            state.request_queue.put(
                _build_prepared_queue_request(
                    state,
                    endpoint="completion",
                    payload=payload,
                    request_id=completion_id,
                    created=created_at,
                    stream=stream,
                )
            )
            if stream:
                start_frame = await frame_queue.get()
                if start_frame.get("kind") == "error":
                    raise _deserialize_openai_error(start_frame["error"])
                first_ready_at = time.perf_counter()
                headers = _response_headers(
                    request_id=completion_id,
                    prompt_token_count=int(start_frame["prompt_token_count"]),
                    queue_wait_s=float(start_frame["queue_wait_s"]),
                    processing_s=first_ready_at - submit_started_at,
                    request_parse_s=handler_started_at - http_received_at,
                    request_setup_s=submit_started_at - handler_started_at,
                    response_build_s=0.0,
                    server_app_s=first_ready_at - http_received_at,
                    ttft_s=start_frame.get("ttft_s"),
                    streaming=True,
                )
                response = web.StreamResponse(
                    status=200,
                    headers=headers,
                )
                response.content_type = "text/event-stream"
                await response.prepare(request)
                try:
                    while True:
                        frame = await frame_queue.get()
                        kind = frame.get("kind")
                        if kind == "delta":
                            await response.write(
                                _sse_payload(
                                    {
                                        "id": completion_id,
                                        "object": "text_completion",
                                        "created": created_at,
                                        "model": response_model,
                                        "choices": [
                                            {
                                                "index": 0,
                                                "text": frame.get("text") or "",
                                                "finish_reason": None,
                                                "logprobs": None,
                                            }
                                        ],
                                    }
                                )
                            )
                            continue
                        if kind == "done":
                            await response.write(
                                _sse_payload(
                                    {
                                        "id": completion_id,
                                        "object": "text_completion",
                                        "created": created_at,
                                        "model": response_model,
                                        "choices": [
                                            {
                                                "index": 0,
                                                "text": "",
                                                "finish_reason": frame.get("finish_reason"),
                                                "logprobs": None,
                                            }
                                        ],
                                    }
                                )
                            )
                            if stream_include_usage:
                                await response.write(
                                    _sse_payload(
                                        _completion_stream_usage_event(
                                            completion_id=completion_id,
                                            created=created_at,
                                            model=response_model,
                                            prompt_token_count=int(start_frame["prompt_token_count"]),
                                            completion_token_count=int(frame["completion_token_count"]),
                                        )
                                    )
                                )
                            await response.write(_sse_payload("[DONE]"))
                            await response.write_eof()
                            return response
                        if kind == "error":
                            raise _deserialize_openai_error(frame["error"])
                finally:
                    response_bridge.unregister(completion_id)

            frame = await frame_queue.get()
        finally:
            if not stream:
                response_bridge.unregister(completion_id)
        if frame.get("kind") == "error":
            raise _deserialize_openai_error(frame["error"])
        backend_finished_at = time.perf_counter()
        response_built_at = time.perf_counter()
        processing_s = backend_finished_at - submit_started_at
        headers = _response_headers(
            request_id=completion_id,
            prompt_token_count=int(frame["prompt_token_count"]),
            completion_token_count=int(frame["completion_token_count"]),
            queue_wait_s=float(frame["queue_wait_s"]),
            processing_s=processing_s,
            request_parse_s=handler_started_at - http_received_at,
            request_setup_s=submit_started_at - handler_started_at,
            response_build_s=response_built_at - backend_finished_at,
            server_app_s=response_built_at - http_received_at,
            ttft_s=frame.get("ttft_s"),
            generation_s=float(frame["generation_s"]),
            total_s=processing_s,
            streaming=False,
        )
        return _aiohttp_json_response(
            {
                "id": completion_id,
                "object": "text_completion",
                "created": created_at,
                "model": response_model,
                "choices": [
                    {
                        "index": 0,
                        "text": frame["text"],
                        "finish_reason": frame["finish_reason"],
                        "logprobs": frame.get("logprobs"),
                    }
                ],
                "usage": _usage_dict(int(frame["prompt_token_count"]), int(frame["completion_token_count"])),
            },
            headers=headers,
        )

    async def chat_completions(request: web.Request):
        http_received_at = request["nanovllm_received_at"]
        payload = _decode_json_body(await request.read())
        handler_started_at = time.perf_counter()
        _require_api_key(state, request.headers.get("Authorization"))
        response_model = str(payload.get("model", state.model_id))
        _base_model, prompt_mode = _parse_openai_model_mode(response_model)
        stream = bool(_optional_bool(payload, "stream", False))
        stream_include_usage = _stream_options_include_usage(payload.get("stream_options"))
        created_at = int(time.time())
        completion_id = f"chatcmpl-{uuid.uuid4().hex}"
        submit_started_at = time.perf_counter()
        response_bridge: FrontendResponseBridge = request.app["response_bridge"]
        frame_queue = response_bridge.register(completion_id)
        try:
            state.request_queue.put(
                _build_prepared_queue_request(
                    state,
                    endpoint="chat",
                    payload=payload,
                    request_id=completion_id,
                    created=created_at,
                    stream=stream,
                )
            )
            if stream:
                start_frame = await frame_queue.get()
                if start_frame.get("kind") == "error":
                    raise _deserialize_openai_error(start_frame["error"])
                first_ready_at = time.perf_counter()
                headers = _response_headers(
                    request_id=completion_id,
                    prompt_token_count=int(start_frame["prompt_token_count"]),
                    queue_wait_s=float(start_frame["queue_wait_s"]),
                    processing_s=first_ready_at - submit_started_at,
                    request_parse_s=handler_started_at - http_received_at,
                    request_setup_s=submit_started_at - handler_started_at,
                    response_build_s=0.0,
                    server_app_s=first_ready_at - http_received_at,
                    ttft_s=start_frame.get("ttft_s"),
                    streaming=True,
                )
                response = web.StreamResponse(status=200, headers=headers)
                response.content_type = "text/event-stream"
                await response.prepare(request)
                try:
                    filter_state = _make_chat_output_filter(prompt_mode)
                    flushed = False
                    await response.write(
                        _sse_payload(
                            {
                                "id": completion_id,
                                "object": "chat.completion.chunk",
                                "created": created_at,
                                "model": response_model,
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": _chat_stream_delta_payload(role="assistant", content=""),
                                        "finish_reason": None,
                                    }
                                ],
                            }
                        )
                    )
                    while True:
                        frame = await frame_queue.get()
                        kind = frame.get("kind")
                        if kind == "delta":
                            for output_kind, output_text in _filter_chat_delta(filter_state, frame.get("text") or ""):
                                await response.write(
                                    _sse_payload(
                                        {
                                            "id": completion_id,
                                            "object": "chat.completion.chunk",
                                            "created": created_at,
                                            "model": response_model,
                                            "choices": [
                                                {
                                                    "index": 0,
                                                    "delta": _chat_stream_delta_payload(
                                                        content=output_text if output_kind == "content" else None,
                                                        reasoning_content=output_text if output_kind == "reasoning_content" else None,
                                                    ),
                                                    "finish_reason": None,
                                                }
                                            ],
                                        }
                                    )
                                )
                            continue
                        if kind == "done":
                            if not flushed:
                                for output_kind, output_text in _flush_chat_delta_filter(filter_state):
                                    await response.write(
                                        _sse_payload(
                                            {
                                                "id": completion_id,
                                                "object": "chat.completion.chunk",
                                                "created": created_at,
                                                "model": response_model,
                                                "choices": [
                                                    {
                                                        "index": 0,
                                                        "delta": _chat_stream_delta_payload(
                                                            content=output_text if output_kind == "content" else None,
                                                            reasoning_content=output_text if output_kind == "reasoning_content" else None,
                                                        ),
                                                        "finish_reason": None,
                                                    }
                                                ],
                                            }
                                        )
                                    )
                                flushed = True
                            await response.write(
                                _sse_payload(
                                    {
                                        "id": completion_id,
                                        "object": "chat.completion.chunk",
                                        "created": created_at,
                                        "model": response_model,
                                        "choices": [
                                            {
                                                "index": 0,
                                                "delta": {},
                                                "finish_reason": _chat_stream_finish_reason(frame.get("finish_reason") or "stop"),
                                            }
                                        ],
                                    }
                                )
                            )
                            if stream_include_usage:
                                await response.write(
                                    _sse_payload(
                                        _chat_stream_usage_event(
                                            completion_id=completion_id,
                                            created=created_at,
                                            model=response_model,
                                            prompt_token_count=int(start_frame["prompt_token_count"]),
                                            completion_token_count=int(frame["completion_token_count"]),
                                        )
                                    )
                                )
                            await response.write(_sse_payload("[DONE]"))
                            await response.write_eof()
                            return response
                        if kind == "error":
                            raise _deserialize_openai_error(frame["error"])
                finally:
                    response_bridge.unregister(completion_id)

            frame = await frame_queue.get()
        finally:
            if not stream:
                response_bridge.unregister(completion_id)
        if frame.get("kind") == "error":
            raise _deserialize_openai_error(frame["error"])
        content_text, reasoning_text = _filter_chat_text(frame["text"], mode=prompt_mode)
        backend_finished_at = time.perf_counter()
        response_built_at = time.perf_counter()
        processing_s = backend_finished_at - submit_started_at
        headers = _response_headers(
            request_id=completion_id,
            prompt_token_count=int(frame["prompt_token_count"]),
            completion_token_count=int(frame["completion_token_count"]),
            queue_wait_s=float(frame["queue_wait_s"]),
            processing_s=processing_s,
            request_parse_s=handler_started_at - http_received_at,
            request_setup_s=submit_started_at - handler_started_at,
            response_build_s=response_built_at - backend_finished_at,
            server_app_s=response_built_at - http_received_at,
            ttft_s=frame.get("ttft_s"),
            generation_s=float(frame["generation_s"]),
            total_s=processing_s,
            streaming=False,
        )
        return _aiohttp_json_response(
            {
                "id": completion_id,
                "object": "chat.completion",
                "created": created_at,
                "model": response_model,
                "choices": [
                    {
                        "index": 0,
                        "message": _assistant_message_payload(content_text, reasoning_text),
                        "finish_reason": frame["finish_reason"],
                    }
                ],
                "usage": _usage_dict(int(frame["prompt_token_count"]), int(frame["completion_token_count"])),
            },
            headers=headers,
        )

    async def tokenize(request: web.Request):
        _require_api_key(state, request.headers.get("Authorization"))
        payload = _decode_json_body(await request.read())
        return _aiohttp_json_response(_handle_tokenize_payload(state, payload))

    async def detokenize(request: web.Request):
        _require_api_key(state, request.headers.get("Authorization"))
        payload = _decode_json_body(await request.read())
        return _aiohttp_json_response(_handle_detokenize_payload(state, payload))

    app.router.add_get("/health", health)
    app.router.add_get("/v1/models", list_models)
    app.router.add_get("/v1/models/{model_id}", retrieve_model)
    app.router.add_post("/v1/completions", completions)
    app.router.add_post("/v1/chat/completions", chat_completions)
    app.router.add_post("/v1/tokenize", tokenize)
    app.router.add_post("/v1/detokenize", detokenize)
    return app


async def _run_queue_frontend_aiohttp_async(
    *,
    model_id: str,
    created: int,
    api_key: str | None,
    frontend_id: int,
    request_queue: Any,
    response_queue: Any,
    public_socket: socket.socket,
    frontend_ready_event: Any | None = None,
):
    app = create_queue_frontend_aiohttp_app(
        model_id=model_id,
        created=created,
        api_key=api_key,
        frontend_id=frontend_id,
        request_queue=request_queue,
        response_queue=response_queue,
    )
    runner = web.AppRunner(app, access_log=None)
    await runner.setup()
    site = web.SockSite(runner, public_socket)
    await site.start()
    _set_ready_event(frontend_ready_event)
    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    for signame in ("SIGINT", "SIGTERM"):
        sig = getattr(signal, signame, None)
        if sig is None:
            continue
        try:
            loop.add_signal_handler(sig, stop_event.set)
        except (NotImplementedError, RuntimeError):
            pass
    try:
        await stop_event.wait()
    finally:
        await runner.cleanup()


def _run_queue_frontend_process(
    *,
    model_id: str,
    created: int,
    api_key: str | None,
    frontend_id: int,
    request_queue: Any,
    response_queue: Any,
    public_socket: socket.socket,
    frontend_http_stack: str,
    log_level: str,
    access_log: bool,
    backlog: int,
    timeout_keep_alive: int,
    disable_cors: bool,
    frontend_ready_event: Any | None = None,
):
    if frontend_http_stack == "aiohttp":
        _enable_uvloop()
        asyncio.run(
            _run_queue_frontend_aiohttp_async(
                model_id=model_id,
                created=created,
                api_key=api_key,
                frontend_id=frontend_id,
                request_queue=request_queue,
                response_queue=response_queue,
                public_socket=public_socket,
                frontend_ready_event=frontend_ready_event,
            )
        )
        return
    app = create_queue_frontend_app(
        model_id=model_id,
        created=created,
        api_key=api_key,
        frontend_id=frontend_id,
        request_queue=request_queue,
        response_queue=response_queue,
        disable_cors=disable_cors,
    )
    _run_uvicorn_server(
        app,
        host=None,
        port=None,
        uds=None,
        log_level=log_level,
        access_log=access_log,
        backlog=backlog,
        timeout_keep_alive=timeout_keep_alive,
        sockets=[public_socket],
        ready_callback=(lambda: _set_ready_event(frontend_ready_event)) if frontend_ready_event is not None else None,
    )


def _run_queue_cluster(args, llm_kwargs: dict[str, Any]):
    if os.name != "posix":
        raise RuntimeError("frontend workers require a POSIX platform.")
    ctx = mp.get_context("spawn")
    model_id = args.served_model_name or _default_model_name(args.model)
    created = int(time.time())
    request_queues = [ctx.SimpleQueue() for _ in range(args.frontend_workers)]
    response_queues = [ctx.SimpleQueue() for _ in range(args.frontend_workers)]
    backend_ready = ctx.Event()
    frontend_ready = ctx.Event()
    public_sockets = _create_reuse_port_sockets(
        args.host,
        args.port,
        args.backlog,
        args.frontend_workers,
    )
    children: list[mp.Process] = []
    try:
        backend = ctx.Process(
            target=_run_queue_backend_process,
            kwargs={
                "model": args.model,
                "served_model_name": args.served_model_name,
                "api_key": args.api_key,
                "llm_kwargs": llm_kwargs,
                "request_queues": request_queues,
                "response_queues": response_queues,
                "ready_event": backend_ready,
            },
            daemon=True,
        )
        backend.start()
        children.append(backend)
        if not backend_ready.wait(timeout=120.0):
            raise RuntimeError("Timed out waiting for queue backend to become ready.")
        for frontend_id, (sock, request_queue, response_queue) in enumerate(
            zip(public_sockets, request_queues, response_queues, strict=True)
        ):
            proc = ctx.Process(
                target=_run_queue_frontend_process,
                kwargs={
                    "model_id": model_id,
                    "created": created,
                    "api_key": args.api_key,
                    "frontend_id": frontend_id,
                    "request_queue": request_queue,
                    "response_queue": response_queue,
                    "public_socket": sock,
                    "frontend_http_stack": args.frontend_http_stack,
                    "log_level": args.log_level,
                    "access_log": args.access_log,
                    "backlog": args.backlog,
                    "timeout_keep_alive": args.timeout_keep_alive,
                    "disable_cors": args.disable_cors,
                    "frontend_ready_event": frontend_ready,
                },
                daemon=True,
            )
            proc.start()
            children.append(proc)
        for sock in public_sockets:
            sock.close()
        _wait_for_frontend_ready(frontend_ready, children)
        _print_public_ready_banner(args.host, args.port)
        while True:
            for proc in children:
                proc.join(timeout=0.2)
                if not proc.is_alive():
                    raise SystemExit(proc.exitcode or 0)
    except KeyboardInterrupt:
        pass
    finally:
        for proc in children:
            if proc.is_alive():
                proc.terminate()
        for proc in children:
            proc.join(timeout=5.0)
        for queue in request_queues:
            try:
                queue.put(None)
            except Exception:
                pass
        for queue in response_queues:
            try:
                queue.put({"kind": "__frontend_stop__"})
            except Exception:
                pass
        for sock in public_sockets:
            try:
                sock.close()
            except OSError:
                pass


def _wait_for_ipc_backend(backend_uds: str, timeout_s: float = 120.0):
    deadline = time.time() + timeout_s
    ping_payload = json.dumps({"kind": "ping"}, ensure_ascii=False).encode("utf-8")
    request = len(ping_payload).to_bytes(4, "little") + ping_payload
    while time.time() < deadline:
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
                sock.settimeout(1.0)
                sock.connect(backend_uds)
                sock.sendall(request)
                header = sock.recv(4)
                if len(header) != 4:
                    raise RuntimeError("short IPC response header")
                size = int.from_bytes(header, "little")
                payload = b""
                while len(payload) < size:
                    chunk = sock.recv(size - len(payload))
                    if not chunk:
                        raise RuntimeError("short IPC response body")
                    payload += chunk
                response = json.loads(payload.decode("utf-8"))
                if response.get("kind") == "pong":
                    return
        except Exception:
            time.sleep(0.1)
    raise RuntimeError(f"Timed out waiting for IPC backend to become ready on uds={backend_uds}")


def _run_ipc_cluster(args, llm_kwargs: dict[str, Any]):
    if os.name != "posix":
        raise RuntimeError("frontend workers require a POSIX platform.")
    ctx = mp.get_context("spawn")
    backend_uds = f"/tmp/nanovllm-ipc-{os.getpid()}.sock"
    if os.path.exists(backend_uds):
        os.remove(backend_uds)
    model_id = args.served_model_name or _default_model_name(args.model)
    created = int(time.time())
    frontend_ready = ctx.Event()
    public_sockets = _create_reuse_port_sockets(
        args.host,
        args.port,
        args.backlog,
        args.frontend_workers,
    )
    children: list[mp.Process] = []
    try:
        backend = ctx.Process(
            target=_run_ipc_backend_process,
            kwargs={
                "model": args.model,
                "served_model_name": args.served_model_name,
                "api_key": args.api_key,
                "llm_kwargs": llm_kwargs,
                "backend_uds": backend_uds,
                "backlog": args.backlog,
            },
            daemon=True,
        )
        backend.start()
        children.append(backend)
        _wait_for_ipc_backend(backend_uds)
        for sock in public_sockets:
            proc = ctx.Process(
                target=_run_ipc_frontend_process,
                kwargs={
                    "model_id": model_id,
                    "created": created,
                    "api_key": args.api_key,
                    "backend_uds": backend_uds,
                    "backend_channel_count": args.proxy_backend_keepalive_connections,
                    "public_socket": sock,
                    "log_level": args.log_level,
                    "access_log": args.access_log,
                    "backlog": args.backlog,
                    "timeout_keep_alive": args.timeout_keep_alive,
                    "disable_cors": args.disable_cors,
                    "frontend_ready_event": frontend_ready,
                },
                daemon=True,
            )
            proc.start()
            children.append(proc)
        for sock in public_sockets:
            sock.close()
        _wait_for_frontend_ready(frontend_ready, children)
        _print_public_ready_banner(args.host, args.port)
        while True:
            for proc in children:
                proc.join(timeout=0.2)
                if not proc.is_alive():
                    raise SystemExit(proc.exitcode or 0)
    except KeyboardInterrupt:
        pass
    finally:
        for proc in children:
            if proc.is_alive():
                proc.terminate()
        for proc in children:
            proc.join(timeout=5.0)
        for sock in public_sockets:
            try:
                sock.close()
            except OSError:
                pass
        if os.path.exists(backend_uds):
            os.remove(backend_uds)


def build_arg_parser():
    parser = argparse.ArgumentParser()
    default_frontend_workers = 2 if os.name == "posix" else 1
    parser.add_argument("--model", required=True)
    parser.add_argument("--served-model-name", default=None)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--log-level", default="warning")
    parser.add_argument("--access-log", action="store_true")
    parser.add_argument("--backlog", type=int, default=4096)
    parser.add_argument("--timeout-keep-alive", type=int, default=30)
    parser.add_argument("--listener-threads", type=int, default=1)
    parser.add_argument(
        "--frontend-mode",
        choices=["auto", "queue", "shared"],
        default="auto",
        help=(
            "HTTP frontend topology. "
            "'auto' prefers shared-state listener threads for multi-listener "
            "single-process serving, and falls back to the queue-backed "
            "multi-process path when needed."
        ),
    )
    parser.add_argument(
        "--frontend-workers",
        type=int,
        default=default_frontend_workers,
        help=(
            "Number of frontend listeners/workers. In auto mode on POSIX, values "
            "> 1 prefer the shared-state listener-thread path for tp=1 serving."
        ),
    )
    parser.add_argument(
        "--frontend-http-stack",
        choices=["fastapi", "aiohttp"],
        default="aiohttp",
        help="HTTP stack for multi-frontend mode. aiohttp is the current throughput default.",
    )
    parser.add_argument("--proxy-backend-max-connections", type=int, default=256)
    parser.add_argument("--proxy-backend-keepalive-connections", type=int, default=256)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--disable-cors", action="store_true")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-num-batched-tokens", type=int, default=16384)
    parser.add_argument("--max-num-seqs", type=int, default=512)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-state-slots", type=int, default=-1)
    parser.add_argument("--sampling-bucket-temperature-resolution", type=float, default=0.0)
    parser.add_argument("--sampling-bucket-top-p-resolution", type=float, default=0.0)
    parser.add_argument("--rwkv-state-cache-safety-reserve-slots", type=int, default=0)
    parser.add_argument("--rwkv-prefill-token-budget", type=int, default=2048)
    parser.add_argument("--rwkv-prefill-max-batch-size", type=int, default=128)
    parser.add_argument("--rwkv-prefill-chunk-size", type=int, default=256)
    parser.add_argument("--rwkv-state-cache-enable", action="store_true")
    add_rwkv_int8_cli_args(parser)
    parser.add_argument("--enforce-eager", action="store_true")
    return parser


def _uvicorn_config(
    app: FastAPI,
    *,
    host: str | None,
    port: int | None,
    uds: str | None,
    log_level: str,
    access_log: bool,
    backlog: int,
    timeout_keep_alive: int,
) -> uvicorn.Config:
    loop_name = "uvloop" if uvloop is not None else "asyncio"
    return uvicorn.Config(
        app,
        host=host or "127.0.0.1",
        port=port or 8000,
        uds=uds,
        loop=loop_name,
        http="httptools",
        log_level=log_level,
        access_log=access_log,
        backlog=backlog,
        timeout_keep_alive=timeout_keep_alive,
    )


def _create_uvicorn_server(
    app: FastAPI,
    *,
    host: str | None,
    port: int | None,
    uds: str | None,
    log_level: str,
    access_log: bool,
    backlog: int,
    timeout_keep_alive: int,
    ready_callback: Callable[[], None] | None = None,
) -> uvicorn.Server:
    return _ReadyAwareUvicornServer(
        _uvicorn_config(
            app,
            host=host,
            port=port,
            uds=uds,
            log_level=log_level,
            access_log=access_log,
            backlog=backlog,
            timeout_keep_alive=timeout_keep_alive,
        ),
        ready_callback=ready_callback,
    )


def _run_uvicorn_server(
    app: FastAPI,
    *,
    host: str | None,
    port: int | None,
    uds: str | None,
    log_level: str,
    access_log: bool,
    backlog: int,
    timeout_keep_alive: int,
    sockets: list[socket.socket] | None = None,
    ready_callback: Callable[[], None] | None = None,
):
    server = _create_uvicorn_server(
        app,
        host=host,
        port=port,
        uds=uds,
        log_level=log_level,
        access_log=access_log,
        backlog=backlog,
        timeout_keep_alive=timeout_keep_alive,
        ready_callback=ready_callback,
    )
    server.run(sockets=sockets)


def _create_listening_socket(host: str, port: int, backlog: int) -> socket.socket:
    addrinfos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM, flags=socket.AI_PASSIVE)
    family, socktype, proto, _, sockaddr = addrinfos[0]
    sock = socket.socket(family, socktype, proto)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(sockaddr)
    sock.listen(backlog)
    sock.set_inheritable(True)
    return sock


def _create_reuse_port_sockets(host: str, port: int, backlog: int, count: int) -> list[socket.socket]:
    addrinfos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM, flags=socket.AI_PASSIVE)
    family, socktype, proto, _, sockaddr = addrinfos[0]
    sockets: list[socket.socket] = []
    try:
        for _ in range(count):
            sock = socket.socket(family, socktype, proto)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
            sock.bind(sockaddr)
            sock.listen(backlog)
            sock.set_inheritable(True)
            sockets.append(sock)
    except Exception:
        for sock in sockets:
            sock.close()
        raise
    return sockets


def _run_backend_process(
    *,
    model: str,
    served_model_name: str | None,
    api_key: str | None,
    llm_kwargs: dict[str, Any],
    backend_uds: str,
    log_level: str,
    backlog: int,
    timeout_keep_alive: int,
):
    app = create_app(
        model=model,
        served_model_name=served_model_name,
        api_key=api_key,
        llm_kwargs=llm_kwargs,
    )
    _run_uvicorn_server(
        app,
        host=None,
        port=None,
        uds=backend_uds,
        log_level=log_level,
        access_log=False,
        backlog=backlog,
        timeout_keep_alive=timeout_keep_alive,
    )


def _run_frontend_process(
    *,
    backend_uds: str,
    public_socket: socket.socket,
    log_level: str,
    access_log: bool,
    backlog: int,
    timeout_keep_alive: int,
    backend_max_connections: int,
    backend_keepalive_connections: int,
    frontend_ready_event: Any | None = None,
):
    app = create_proxy_app(
        backend_uds=backend_uds,
        backend_max_connections=backend_max_connections,
        backend_keepalive_connections=backend_keepalive_connections,
    )
    _run_uvicorn_server(
        app,
        host=None,
        port=None,
        uds=None,
        log_level=log_level,
        access_log=access_log,
        backlog=backlog,
        timeout_keep_alive=timeout_keep_alive,
        sockets=[public_socket],
        ready_callback=(lambda: _set_ready_event(frontend_ready_event)) if frontend_ready_event is not None else None,
    )


def _wait_for_backend(backend_uds: str, timeout_s: float = 120.0):
    deadline = time.time() + timeout_s
    transport = httpx.HTTPTransport(uds=backend_uds)
    with httpx.Client(base_url="http://nanovllm-backend", transport=transport, timeout=1.0) as client:
        while time.time() < deadline:
            try:
                response = client.get("/health")
                if response.status_code == 200:
                    return
            except Exception:
                pass
            time.sleep(0.1)
    raise RuntimeError(f"Timed out waiting for backend to become ready on uds={backend_uds}")


def _wait_for_frontend_ready(frontend_ready_event: Any, children: list[mp.Process], timeout_s: float = 120.0):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if frontend_ready_event.wait(timeout=0.1):
            return
        for proc in children:
            if not proc.is_alive():
                raise RuntimeError(
                    f"Frontend process exited before becoming ready (pid={proc.pid}, exitcode={proc.exitcode})."
                )
    raise RuntimeError("Timed out waiting for public frontend to become ready.")


def _run_proxy_cluster(args, llm_kwargs: dict[str, Any]):
    if os.name != "posix":
        raise RuntimeError("frontend proxy workers require a POSIX platform.")
    ctx = mp.get_context("spawn")
    backend_uds = f"/tmp/nanovllm-api-{os.getpid()}.sock"
    if os.path.exists(backend_uds):
        os.remove(backend_uds)
    public_socket = _create_listening_socket(args.host, args.port, args.backlog)
    frontend_ready = ctx.Event()
    children: list[mp.Process] = []
    try:
        backend = ctx.Process(
            target=_run_backend_process,
            kwargs={
                "model": args.model,
                "served_model_name": args.served_model_name,
                "api_key": args.api_key,
                "llm_kwargs": llm_kwargs,
                "backend_uds": backend_uds,
                "log_level": args.log_level,
                "backlog": args.backlog,
                "timeout_keep_alive": args.timeout_keep_alive,
            },
            daemon=True,
        )
        backend.start()
        children.append(backend)
        _wait_for_backend(backend_uds)
        for _ in range(args.frontend_workers):
            proc = ctx.Process(
                target=_run_frontend_process,
                kwargs={
                    "backend_uds": backend_uds,
                    "public_socket": public_socket,
                    "log_level": args.log_level,
                    "access_log": args.access_log,
                    "backlog": args.backlog,
                    "timeout_keep_alive": args.timeout_keep_alive,
                    "backend_max_connections": args.proxy_backend_max_connections,
                    "backend_keepalive_connections": args.proxy_backend_keepalive_connections,
                    "frontend_ready_event": frontend_ready,
                },
                daemon=True,
            )
            proc.start()
            children.append(proc)
        _wait_for_frontend_ready(frontend_ready, children)
        _print_public_ready_banner(args.host, args.port)
        while True:
            for proc in children:
                proc.join(timeout=0.2)
                if not proc.is_alive():
                    raise SystemExit(proc.exitcode or 0)
    except KeyboardInterrupt:
        pass
    finally:
        for proc in children:
            if proc.is_alive():
                proc.terminate()
        for proc in children:
            proc.join(timeout=5.0)
        public_socket.close()
        if os.path.exists(backend_uds):
            os.remove(backend_uds)


def _run_listener_threads(args, llm_kwargs: dict[str, Any]):
    if args.listener_threads <= 1:
        raise ValueError("listener_threads must be > 1 for multi-listener mode.")
    shared_state = _create_server_state(
        model=args.model,
        served_model_name=args.served_model_name,
        api_key=args.api_key,
        llm_kwargs=llm_kwargs,
    )
    if shared_state.batcher is not None:
        shared_state.batcher.start()
    sockets = _create_reuse_port_sockets(args.host, args.port, args.backlog, args.listener_threads)
    announce_ready = _make_once_callback(lambda: _print_public_ready_banner(args.host, args.port))
    servers: list[uvicorn.Server] = []
    threads: list[threading.Thread] = []
    try:
        for sock in sockets:
            app = create_app(
                model=args.model,
                served_model_name=args.served_model_name,
                api_key=args.api_key,
                llm_kwargs=None,
                server_state=shared_state,
                manage_resources=False,
                disable_cors=args.disable_cors,
            )
            server = _create_uvicorn_server(
                app,
                host=None,
                port=None,
                uds=None,
                log_level=args.log_level,
                access_log=args.access_log,
                backlog=args.backlog,
                timeout_keep_alive=args.timeout_keep_alive,
                ready_callback=announce_ready,
            )
            thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
            thread.start()
            servers.append(server)
            threads.append(thread)
        while True:
            any_alive = False
            for thread in threads:
                thread.join(timeout=0.2)
                any_alive = any_alive or thread.is_alive()
            if not any_alive:
                return
    except KeyboardInterrupt:
        pass
    finally:
        for server in servers:
            server.should_exit = True
        for sock in sockets:
            sock.close()
        for thread in threads:
            thread.join(timeout=5.0)
        if shared_state.batcher is not None:
            shared_state.batcher.stop()
        shared_state.llm.exit()


def _resolve_frontend_runtime(args) -> tuple[str, int]:
    listener_threads = args.listener_threads
    frontend_mode = args.frontend_mode
    if frontend_mode == "auto":
        if listener_threads > 1:
            frontend_mode = "shared"
        elif (
            args.frontend_workers > 1
            and os.name == "posix"
            and args.tensor_parallel_size == 1
        ):
            frontend_mode = "shared"
            listener_threads = args.frontend_workers
        elif args.frontend_workers > 1:
            frontend_mode = "queue"
        else:
            frontend_mode = "single"
    elif frontend_mode == "shared":
        if listener_threads <= 1 and args.frontend_workers > 1:
            listener_threads = args.frontend_workers
        if listener_threads <= 1:
            frontend_mode = "single"
    elif args.frontend_workers <= 1:
        frontend_mode = "single"
    return frontend_mode, listener_threads


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    llm_kwargs = {
        "tensor_parallel_size": args.tensor_parallel_size,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "max_num_seqs": args.max_num_seqs,
        "max_model_len": args.max_model_len,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_state_slots": args.max_state_slots,
        "sampling_bucket_temperature_resolution": args.sampling_bucket_temperature_resolution,
        "sampling_bucket_top_p_resolution": args.sampling_bucket_top_p_resolution,
        "rwkv_state_cache_safety_reserve_slots": args.rwkv_state_cache_safety_reserve_slots,
        "rwkv_prefill_token_budget": args.rwkv_prefill_token_budget,
        "rwkv_prefill_max_batch_size": args.rwkv_prefill_max_batch_size,
        "rwkv_prefill_chunk_size": args.rwkv_prefill_chunk_size,
        "rwkv_state_cache_enable": args.rwkv_state_cache_enable,
        "rwkv_quant_int8": args.rwkv_quant_int8,
        "rwkv_int8_fp16_lm_head": args.rwkv_int8_fp16_lm_head,
        "enforce_eager": args.enforce_eager,
    }
    frontend_mode, listener_threads = _resolve_frontend_runtime(args)

    if frontend_mode == "queue":
        _run_queue_cluster(args, llm_kwargs)
        return
    if frontend_mode == "shared":
        shared_args = argparse.Namespace(**vars(args))
        shared_args.listener_threads = listener_threads
        _run_listener_threads(shared_args, llm_kwargs)
        return
    app = create_app(
        model=args.model,
        served_model_name=args.served_model_name,
        api_key=args.api_key,
        llm_kwargs=llm_kwargs,
        disable_cors=args.disable_cors,
    )
    _run_uvicorn_server(
        app,
        host=args.host,
        port=args.port,
        uds=None,
        log_level=args.log_level,
        access_log=args.access_log,
        backlog=args.backlog,
        timeout_keep_alive=args.timeout_keep_alive,
        ready_callback=lambda: _print_public_ready_banner(args.host, args.port),
    )


if __name__ == "__main__":
    main()
