from __future__ import annotations

import time
import threading
from collections import deque
from contextlib import contextmanager
from unittest import mock

import torch
from fastapi.testclient import TestClient

from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.entrypoints.openai import api_server


class FakeTokenizer:
    def encode(self, text: str) -> list[int]:
        return [ord(ch) for ch in text]

    def decode(self, token_ids: list[int], utf8_errors: str = "ignore") -> str:
        del utf8_errors
        return "".join(chr(token_id) for token_id in token_ids)


class FakeTemplateTokenizer(FakeTokenizer):
    def __init__(self, template_text: str = "<CHAT> templated"):
        self.template_text = template_text
        self.calls: list[tuple[list[dict[str, str]], bool, bool]] = []

    def apply_chat_template(self, messages, tokenize: bool, add_generation_prompt: bool) -> str:
        self.calls.append((messages, tokenize, add_generation_prompt))
        return self.template_text


class FakeScheduler:
    def __init__(self, llm: "FakeLLM"):
        self.llm = llm
        self.max_num_seqs = int(llm.llm_kwargs.get("max_num_seqs", 512))
        self.waiting = deque()
        self.running = deque()

    def _prefill_step_tokens(self, seq):
        return seq.prefill_step_tokens(-1)

    def schedule(self):
        scheduled = self.schedule_prefill_only()
        if scheduled:
            return scheduled, True
        scheduled = self.schedule_decode_only()
        if not scheduled:
            raise AssertionError("schedule() called without active sequences.")
        return scheduled, False

    def schedule_prefill_only(self):
        if not self.waiting:
            return []
        scheduled = []
        while self.waiting and len(scheduled) < self.max_num_seqs:
            seq = self.waiting.popleft()
            seq.status = SequenceStatus.RUNNING
            self.running.append(seq)
            scheduled.append(seq)
        return scheduled

    def schedule_decode_only(self):
        if not self.running:
            return []
        return list(self.running)[: self.max_num_seqs]

    def postprocess(self, seqs, token_ids):
        for seq, token_id in zip(seqs, token_ids):
            if token_id is None:
                continue
            pending = self.llm._pending_token_ids_by_seq.get(seq.seq_id)
            seq.append_token(token_id)
            if pending and pending[0] == token_id:
                pending.pop(0)
                if not pending:
                    self.llm._pending_token_ids_by_seq.pop(seq.seq_id, None)
            if seq.num_completion_tokens >= seq.max_tokens or not pending:
                seq.status = SequenceStatus.FINISHED
                self.running.remove(seq)

    def abort(self, seq_id: int) -> bool:
        for queue in (self.waiting, self.running):
            for seq in list(queue):
                if seq.seq_id != seq_id:
                    continue
                queue.remove(seq)
                seq.status = SequenceStatus.FINISHED
                self.llm._pending_token_ids_by_seq.pop(seq.seq_id, None)
                return True
        return False


class FakeModelRunner:
    def __init__(self, llm: "FakeLLM"):
        self.llm = llm
        self.sampler = lambda logits, _seqs: logits.argmax(dim=-1)
        self.calls: list[tuple[str, bool, list[int]]] = []
        self._calls_cv = threading.Condition()

    def wait_for_call_count(self, target: int, timeout: float = 1.0) -> bool:
        deadline = time.time() + timeout
        with self._calls_cv:
            while len(self.calls) < target:
                remaining = deadline - time.time()
                if remaining <= 0:
                    return False
                self._calls_cv.wait(timeout=remaining)
            return True

    def call(self, name: str, seqs, is_prefill: bool):
        if name in {"run", "run_logits"}:
            with self._calls_cv:
                self.calls.append((name, is_prefill, [seq.seq_id for seq in seqs]))
                self._calls_cv.notify_all()
        if name == "run":
            if self.llm.per_token_delay_s > 0:
                time.sleep(self.llm.per_token_delay_s)
            return [self.llm.next_token(seq) for seq in seqs]
        if name == "run_logits":
            rows = []
            for seq in seqs:
                next_token = self.llm.peek_next_token(seq)
                logits = torch.full((256,), -20.0, dtype=torch.float32)
                logits[next_token] = 5.0
                logits[(next_token + 1) % 256] = 4.0
                logits[(next_token + 2) % 256] = 3.0
                rows.append(logits)
            return torch.stack(rows, dim=0)
        if name == "prepare_postprocess":
            return None
        raise AssertionError(f"Unexpected model runner call: {name}")


class FakeLLM:
    def __init__(
        self,
        model: str,
        *,
        tokenizer: FakeTokenizer | None = None,
        completion_text: str = "OK",
        chat_text: str = "CHAT",
        per_token_delay_s: float = 0.0,
        output_resolver=None,
        llm_kwargs: dict | None = None,
    ):
        self.model = model
        self.llm_kwargs = dict(llm_kwargs or {})
        self.tokenizer = tokenizer or FakeTokenizer()
        self.completion_text = completion_text
        self.chat_text = chat_text
        self.per_token_delay_s = per_token_delay_s
        self.output_resolver = output_resolver
        self.scheduler = FakeScheduler(self)
        self.model_runner = FakeModelRunner(self)
        self.received_requests: list[dict[str, object]] = []
        self.exit_called = False
        self._pending_token_ids_by_seq: dict[int, list[int]] = {}

    def add_request(self, prompt_text, sampling_params):
        prompt_token_ids = prompt_text if isinstance(prompt_text, list) else self.tokenizer.encode(prompt_text)
        resolved_prompt_text = (
            self.tokenizer.decode(prompt_token_ids)
            if isinstance(prompt_text, list)
            else prompt_text
        )
        self.received_requests.append(
            {
                "prompt_text": resolved_prompt_text,
                "sampling_params": sampling_params,
            }
        )
        seq = Sequence(prompt_token_ids, sampling_params)
        output_text = self._resolve_output_text(resolved_prompt_text)
        self._pending_token_ids_by_seq[seq.seq_id] = self.tokenizer.encode(output_text)[: sampling_params.max_tokens]
        self.scheduler.waiting.append(seq)
        return seq

    def _resolve_output_text(self, prompt_text: str) -> str:
        if self.output_resolver is not None:
            return self.output_resolver(prompt_text)
        if prompt_text.startswith("<CHAT>") or "Assistant:" in prompt_text:
            return self.chat_text
        return self.completion_text

    def next_token(self, seq: Sequence) -> int:
        pending = self._pending_token_ids_by_seq.get(seq.seq_id)
        if pending is None or not pending:
            raise AssertionError("next_token() called with no pending tokens.")
        token_id = pending.pop(0)
        if not pending:
            self._pending_token_ids_by_seq.pop(seq.seq_id, None)
        return token_id

    def peek_next_token(self, seq: Sequence) -> int:
        pending = self._pending_token_ids_by_seq.get(seq.seq_id)
        if pending is None or not pending:
            return ord("!")
        return pending[0]

    def is_finished(self) -> bool:
        return not self.scheduler.waiting and not self.scheduler.running

    def abort(self, seq_id: int) -> bool:
        return self.scheduler.abort(seq_id)

    def exit(self):
        self.exit_called = True


class FakeLLMFactory:
    def __init__(self, **fake_llm_kwargs):
        self.fake_llm_kwargs = dict(fake_llm_kwargs)
        self.calls: list[tuple[str, dict]] = []
        self.instances: list[FakeLLM] = []

    def __call__(self, model: str, **llm_kwargs):
        self.calls.append((model, dict(llm_kwargs)))
        llm = FakeLLM(model, llm_kwargs=llm_kwargs, **self.fake_llm_kwargs)
        self.instances.append(llm)
        return llm


@contextmanager
def patched_app(
    *,
    model: str = "/models/rwkv-test.pth",
    served_model_name: str = "rwkv-test",
    api_key: str | None = None,
    llm_kwargs: dict | None = None,
    **fake_llm_kwargs,
):
    factory = FakeLLMFactory(**fake_llm_kwargs)
    with mock.patch.object(api_server, "LLM", factory):
        app = api_server.create_app(
            model=model,
            served_model_name=served_model_name,
            api_key=api_key,
            llm_kwargs=llm_kwargs,
        )
    try:
        yield app, factory.instances[0], factory
    finally:
        if app.state.server.batcher is not None:
            app.state.server.batcher.stop()
        llm = factory.instances[0]
        if not llm.exit_called:
            llm.exit()


@contextmanager
def patched_test_client(**kwargs):
    with patched_app(**kwargs) as (app, llm, factory):
        with TestClient(app) as client:
            yield client, llm, factory
