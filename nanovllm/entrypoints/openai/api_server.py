from __future__ import annotations

import argparse
import json
import os
import threading
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Literal

import uvicorn
from fastapi import FastAPI, Header, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict

from nanovllm import LLM, SamplingParams
from nanovllm.utils.rwkv_int8 import (
    add_rwkv_int8_cli_args,
)


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


class CompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str
    prompt: str | list[str]
    max_tokens: int | None = None
    temperature: float | None = None
    stream: bool | None = False
    n: int | None = 1
    top_p: float | None = None
    stop: str | list[str] | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    logprobs: int | bool | None = None
    echo: bool | None = None
    seed: int | None = None
    user: str | None = None


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
    logprobs: bool | None = None
    top_logprobs: int | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None
    parallel_tool_calls: bool | None = None
    response_format: dict[str, Any] | None = None
    seed: int | None = None
    user: str | None = None


@dataclass
class ServerState:
    llm: LLM
    model_id: str
    created: int
    api_key: str | None
    lock: threading.Lock


@dataclass
class SingleRequestResult:
    prompt_token_ids: list[int]
    completion_token_ids: list[int]
    text: str
    finish_reason: str
    ttft_s: float | None
    generation_s: float


def _openai_error_response(exc: OpenAIAPIError):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "message": exc.message,
                "type": exc.error_type,
                "param": exc.param,
                "code": exc.code,
            }
        },
    )


def _coerce_text_content(content: str | list[TextPart] | None) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    chunks: list[str] = []
    for part in content:
        if part.type != "text":
            raise OpenAIAPIError(
                400,
                f"Only text content parts are supported in this server. Got content part type={part.type!r}.",
                param="messages",
            )
        chunks.append(part.text or "")
    return "".join(chunks)


def _render_chat_prompt(tokenizer, messages: list[ChatMessage]) -> str:
    normalized_messages = []
    for msg in messages:
        role = "system" if msg.role == "developer" else msg.role
        normalized_messages.append(
            {
                "role": role,
                "content": _coerce_text_content(msg.content),
            }
        )

    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                normalized_messages,
                tokenize=False,
                add_generation_prompt=True,
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
        lines.append(f"{role_names.get(msg['role'], msg['role'].title())}: {msg['content']}")
    if not normalized_messages or normalized_messages[-1]["role"] != "assistant":
        lines.append("Assistant:")
    return "\n".join(lines)


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
    if not req.model_extra:
        return
    unsupported = [key for key, value in req.model_extra.items() if value is not None]
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
    if presence_penalty not in (None, 0, 0.0):
        raise OpenAIAPIError(400, "presence_penalty is not supported.", param="presence_penalty")
    if frequency_penalty not in (None, 0, 0.0):
        raise OpenAIAPIError(400, "frequency_penalty is not supported.", param="frequency_penalty")
    if seed is not None:
        raise OpenAIAPIError(400, "seed is not supported yet.", param="seed")


def _validated_sampling_params(
    *,
    temperature: float | None,
    top_p: float | None,
    max_tokens: int | None,
) -> SamplingParams:
    use_temperature = 1.0 if temperature is None else temperature
    use_top_p = 1.0 if top_p is None else top_p
    use_max_tokens = 16 if max_tokens is None else max_tokens
    if use_temperature < 0:
        raise OpenAIAPIError(400, "temperature must be non-negative.", param="temperature")
    if not (0.0 <= use_top_p <= 1.0):
        raise OpenAIAPIError(400, "top_p must be in [0, 1].", param="top_p")
    if use_max_tokens <= 0:
        raise OpenAIAPIError(400, "max_tokens must be positive.", param="max_tokens")
    return SamplingParams(temperature=use_temperature, top_p=use_top_p, max_tokens=use_max_tokens)


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
    if req.logprobs not in (None, False, 0):
        raise OpenAIAPIError(400, "logprobs is not supported.", param="logprobs")
    if req.echo not in (None, False):
        raise OpenAIAPIError(400, "echo is not supported.", param="echo")
    return _validated_sampling_params(
        temperature=req.temperature,
        top_p=req.top_p,
        max_tokens=req.max_tokens,
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
    )


def _normalize_prompt(prompt: str | list[str]) -> str:
    if isinstance(prompt, str):
        return prompt
    if len(prompt) != 1:
        raise OpenAIAPIError(400, "Only a single prompt is supported.", param="prompt")
    return prompt[0]


def _usage_dict(prompt_token_count: int, completion_token_count: int):
    return {
        "prompt_tokens": prompt_token_count,
        "completion_tokens": completion_token_count,
        "total_tokens": prompt_token_count + completion_token_count,
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


def _sse_payload(data: dict[str, Any] | str) -> bytes:
    if isinstance(data, str):
        return f"data: {data}\n\n".encode("utf-8")
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n".encode("utf-8")


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


class _SingleRequestRunner:
    def __init__(self, state: ServerState, prompt_text: str, sampling_params: SamplingParams):
        self.state = state
        self.prompt_token_ids = state.llm.tokenizer.encode(prompt_text)
        self.sampling_params = sampling_params
        state.llm.add_request(prompt_text, sampling_params)
        self.completion_token_ids: list[int] = []
        self.visible_text = ""
        self.started_at = time.perf_counter()
        self.first_token_at: float | None = None
        self.finished_at: float | None = None

    @property
    def is_finished(self) -> bool:
        return self.finished_at is not None

    def step(self) -> str:
        if self.is_finished:
            return ""
        seqs, is_prefill = self.state.llm.scheduler.schedule()
        token_ids = self.state.llm.model_runner.call("run", seqs, is_prefill)
        self.state.llm.scheduler.postprocess(seqs, token_ids)
        now = time.perf_counter()
        self.completion_token_ids.append(token_ids[0])
        if self.first_token_at is None:
            self.first_token_at = now
        new_visible_text = _decode_visible_text(self.state.llm.tokenizer, self.completion_token_ids)
        delta = new_visible_text[len(self.visible_text):] if new_visible_text.startswith(self.visible_text) else new_visible_text
        self.visible_text = new_visible_text
        if self.state.llm.is_finished():
            self.finished_at = now
        return delta

    def run_to_completion(self) -> SingleRequestResult:
        while not self.is_finished:
            self.step()
        assert self.finished_at is not None
        finish_reason = _finish_reason(len(self.completion_token_ids), self.sampling_params.max_tokens)
        ttft_s = None if self.first_token_at is None else self.first_token_at - self.started_at
        return SingleRequestResult(
            prompt_token_ids=self.prompt_token_ids,
            completion_token_ids=self.completion_token_ids,
            text=self.visible_text,
            finish_reason=finish_reason,
            ttft_s=ttft_s,
            generation_s=self.finished_at - self.started_at,
        )


def create_app(
    *,
    model: str,
    served_model_name: str | None = None,
    api_key: str | None = None,
    llm_kwargs: dict[str, Any] | None = None,
) -> FastAPI:
    llm = LLM(model, **(llm_kwargs or {}))
    state = ServerState(
        llm=llm,
        model_id=served_model_name or _default_model_name(model),
        created=int(time.time()),
        api_key=api_key,
        lock=threading.Lock(),
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            yield
        finally:
            app.state.server.llm.exit()

    app = FastAPI(title="nano-vllm OpenAI-compatible API", lifespan=lifespan)
    app.state.server = state

    @app.exception_handler(OpenAIAPIError)
    async def _handle_openai_error(request: Request, exc: OpenAIAPIError):
        return _openai_error_response(exc)

    @app.exception_handler(RequestValidationError)
    async def _handle_validation_error(request: Request, exc: RequestValidationError):
        return _openai_error_response(
            OpenAIAPIError(400, f"Invalid request body: {exc.errors()}", param=None)
        )

    @app.get("/health")
    def health():
        return {"status": "ok", "model": app.state.server.model_id}

    @app.get("/v1/models")
    def list_models(authorization: str | None = Header(default=None)):
        _require_api_key(app.state.server, authorization)
        model_info = {
            "id": app.state.server.model_id,
            "object": "model",
            "created": app.state.server.created,
            "owned_by": "nano-vllm",
        }
        return {"object": "list", "data": [model_info]}

    @app.get("/v1/models/{model_id}")
    def retrieve_model(model_id: str, authorization: str | None = Header(default=None)):
        _require_api_key(app.state.server, authorization)
        _validate_model(model_id, app.state.server)
        return {
            "id": app.state.server.model_id,
            "object": "model",
            "created": app.state.server.created,
            "owned_by": "nano-vllm",
        }

    @app.post("/v1/completions")
    def completions(req: CompletionRequest, authorization: str | None = Header(default=None)):
        _require_api_key(app.state.server, authorization)
        _validate_model(req.model, app.state.server)
        sampling_params = _sampling_params_from_completion(req)
        prompt_text = _normalize_prompt(req.prompt)
        created = int(time.time())
        completion_id = f"cmpl-{uuid.uuid4().hex}"
        if req.stream:
            request_started_at = time.perf_counter()
            app.state.server.lock.acquire()
            queue_wait_s = time.perf_counter() - request_started_at
            try:
                runner = _SingleRequestRunner(app.state.server, prompt_text, sampling_params)
                first_delta = ""
                while runner.first_token_at is None and not runner.is_finished:
                    delta = runner.step()
                    if delta:
                        first_delta = delta
                        break
                processing_s = time.perf_counter() - request_started_at
                headers = _response_headers(
                    request_id=completion_id,
                    prompt_token_count=len(runner.prompt_token_ids),
                    queue_wait_s=queue_wait_s,
                    processing_s=processing_s,
                    ttft_s=None if runner.first_token_at is None else runner.first_token_at - runner.started_at,
                    streaming=True,
                )
            except Exception:
                app.state.server.lock.release()
                raise
            def event_stream():
                try:
                    if first_delta:
                        yield _sse_payload(
                            {
                                "id": completion_id,
                                "object": "text_completion",
                                "created": created,
                                "model": app.state.server.model_id,
                                "choices": [
                                    {
                                        "index": 0,
                                        "text": first_delta,
                                        "finish_reason": None,
                                        "logprobs": None,
                                    }
                                ],
                            }
                        )
                    while not runner.is_finished:
                        delta = runner.step()
                        if delta:
                            yield _sse_payload(
                                {
                                    "id": completion_id,
                                    "object": "text_completion",
                                    "created": created,
                                    "model": app.state.server.model_id,
                                    "choices": [
                                        {
                                            "index": 0,
                                            "text": delta,
                                            "finish_reason": None,
                                            "logprobs": None,
                                        }
                                    ],
                                }
                            )
                    result = runner.run_to_completion()
                    yield _sse_payload(
                        {
                            "id": completion_id,
                            "object": "text_completion",
                            "created": created,
                            "model": app.state.server.model_id,
                            "choices": [
                                {
                                    "index": 0,
                                    "text": "",
                                    "finish_reason": result.finish_reason,
                                    "logprobs": None,
                                }
                            ],
                        }
                    )
                    yield _sse_payload("[DONE]")
                finally:
                    app.state.server.lock.release()

            return StreamingResponse(event_stream(), media_type="text/event-stream", headers=headers)
        request_started_at = time.perf_counter()
        app.state.server.lock.acquire()
        queue_wait_s = time.perf_counter() - request_started_at
        try:
            runner = _SingleRequestRunner(app.state.server, prompt_text, sampling_params)
            result = runner.run_to_completion()
        finally:
            app.state.server.lock.release()
        total_s = time.perf_counter() - request_started_at
        headers = _response_headers(
            request_id=completion_id,
            prompt_token_count=len(result.prompt_token_ids),
            completion_token_count=len(result.completion_token_ids),
            queue_wait_s=queue_wait_s,
            processing_s=total_s,
            ttft_s=result.ttft_s,
            generation_s=result.generation_s,
            total_s=total_s,
            streaming=False,
        )
        return JSONResponse(
            content={
                "id": completion_id,
                "object": "text_completion",
                "created": created,
                "model": app.state.server.model_id,
                "choices": [
                    {
                        "index": 0,
                        "text": result.text,
                        "finish_reason": result.finish_reason,
                        "logprobs": None,
                    }
                ],
                "usage": _usage_dict(len(result.prompt_token_ids), len(result.completion_token_ids)),
            },
            headers=headers,
        )

    @app.post("/v1/chat/completions")
    def chat_completions(req: ChatCompletionRequest, authorization: str | None = Header(default=None)):
        _require_api_key(app.state.server, authorization)
        _validate_model(req.model, app.state.server)
        sampling_params = _sampling_params_from_chat(req)
        prompt_text = _render_chat_prompt(app.state.server.llm.tokenizer, req.messages)
        created = int(time.time())
        completion_id = f"chatcmpl-{uuid.uuid4().hex}"
        if req.stream:
            request_started_at = time.perf_counter()
            app.state.server.lock.acquire()
            queue_wait_s = time.perf_counter() - request_started_at
            try:
                runner = _SingleRequestRunner(app.state.server, prompt_text, sampling_params)
                first_delta = ""
                while runner.first_token_at is None and not runner.is_finished:
                    delta = runner.step()
                    if delta:
                        first_delta = delta
                        break
                processing_s = time.perf_counter() - request_started_at
                headers = _response_headers(
                    request_id=completion_id,
                    prompt_token_count=len(runner.prompt_token_ids),
                    queue_wait_s=queue_wait_s,
                    processing_s=processing_s,
                    ttft_s=None if runner.first_token_at is None else runner.first_token_at - runner.started_at,
                    streaming=True,
                )
            except Exception:
                app.state.server.lock.release()
                raise
            def event_stream():
                try:
                    yield _sse_payload(
                        {
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": app.state.server.model_id,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"role": "assistant", "content": ""},
                                    "finish_reason": None,
                                }
                            ],
                        }
                    )
                    if first_delta:
                        yield _sse_payload(
                            {
                                "id": completion_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": app.state.server.model_id,
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": {"content": first_delta},
                                        "finish_reason": None,
                                    }
                                ],
                            }
                        )
                    while not runner.is_finished:
                        delta = runner.step()
                        if delta:
                            yield _sse_payload(
                                {
                                    "id": completion_id,
                                    "object": "chat.completion.chunk",
                                    "created": created,
                                    "model": app.state.server.model_id,
                                    "choices": [
                                        {
                                            "index": 0,
                                            "delta": {"content": delta},
                                            "finish_reason": None,
                                        }
                                    ],
                                }
                            )
                    result = runner.run_to_completion()
                    yield _sse_payload(
                        {
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": app.state.server.model_id,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {},
                                    "finish_reason": _chat_stream_finish_reason(result.finish_reason),
                                }
                            ],
                        }
                    )
                    yield _sse_payload("[DONE]")
                finally:
                    app.state.server.lock.release()

            return StreamingResponse(event_stream(), media_type="text/event-stream", headers=headers)
        request_started_at = time.perf_counter()
        app.state.server.lock.acquire()
        queue_wait_s = time.perf_counter() - request_started_at
        try:
            runner = _SingleRequestRunner(app.state.server, prompt_text, sampling_params)
            result = runner.run_to_completion()
        finally:
            app.state.server.lock.release()
        total_s = time.perf_counter() - request_started_at
        headers = _response_headers(
            request_id=completion_id,
            prompt_token_count=len(result.prompt_token_ids),
            completion_token_count=len(result.completion_token_ids),
            queue_wait_s=queue_wait_s,
            processing_s=total_s,
            ttft_s=result.ttft_s,
            generation_s=result.generation_s,
            total_s=total_s,
            streaming=False,
        )
        return JSONResponse(
            content={
                "id": completion_id,
                "object": "chat.completion",
                "created": created,
                "model": app.state.server.model_id,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": result.text,
                        },
                        "finish_reason": result.finish_reason,
                    }
                ],
                "usage": _usage_dict(len(result.prompt_token_ids), len(result.completion_token_ids)),
            },
            headers=headers,
        )

    return app


def build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--served-model-name", default=None)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--api-key", default=None)
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
    parser.add_argument("--rwkv-state-cache-enable", action="store_true")
    add_rwkv_int8_cli_args(parser)
    parser.add_argument("--enforce-eager", action="store_true")
    return parser


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
        "rwkv_state_cache_enable": args.rwkv_state_cache_enable,
        "rwkv_quant_int8": args.rwkv_quant_int8,
        "rwkv_int8_fp16_lm_head": args.rwkv_int8_fp16_lm_head,
        "enforce_eager": args.enforce_eager,
    }
    app = create_app(
        model=args.model,
        served_model_name=args.served_model_name,
        api_key=args.api_key,
        llm_kwargs=llm_kwargs,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
