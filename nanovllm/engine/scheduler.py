from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.state_cache import StatePrefixIndex, StateSlotManager


class Scheduler:

    def __init__(self, config: Config):
        self.config = config
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.rwkv_prefill_max_batch_size = config.rwkv_prefill_max_batch_size
        self.rwkv_prefill_token_budget = config.rwkv_prefill_token_budget
        self.eos = config.eos
        self.rwkv_state_cache_enable = config.rwkv_state_cache_enable
        self.block_manager = None if self.rwkv_state_cache_enable else BlockManager(config.num_state_blocks)
        self.slot_manager = StateSlotManager(config.num_state_blocks) if self.rwkv_state_cache_enable else None
        self.prefix_index = StatePrefixIndex() if self.rwkv_state_cache_enable else None
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        if self.rwkv_state_cache_enable:
            return self._schedule_rwkv_state_cache()
        return self._schedule_legacy()

    def _schedule_legacy(self) -> tuple[list[Sequence], bool]:
        # prefill
        scheduled_seqs = []
        num_seqs = 0
        num_batched_tokens = 0
        prefill_max_num_seqs = min(self.max_num_seqs, self.rwkv_prefill_max_batch_size)
        prefill_max_batched_tokens = min(self.max_num_batched_tokens, self.rwkv_prefill_token_budget)
        while self.waiting and num_seqs < self.max_num_seqs:
            seq = self.waiting[0]
            num_new_tokens = len(seq) - seq.num_cached_tokens
            if scheduled_seqs:
                if num_seqs >= prefill_max_num_seqs or num_batched_tokens + num_new_tokens > prefill_max_batched_tokens:
                    break
            elif not scheduled_seqs:
                if num_new_tokens <= prefill_max_batched_tokens and num_seqs >= prefill_max_num_seqs:
                    break
            if num_batched_tokens + num_new_tokens > self.max_num_batched_tokens or not self.block_manager.can_allocate(seq):
                break
            num_seqs += 1
            self.block_manager.allocate(seq)
            num_batched_tokens += num_new_tokens
            seq.status = SequenceStatus.RUNNING
            self.waiting.popleft()
            self.running.append(seq)
            scheduled_seqs.append(seq)
        if scheduled_seqs:
            return scheduled_seqs, True

        # decode
        while self.running and num_seqs < self.max_num_seqs:
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                num_seqs += 1
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
        assert scheduled_seqs
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False

    def _schedule_rwkv_state_cache(self) -> tuple[list[Sequence], bool]:
        scheduled_seqs = []
        num_seqs = 0
        num_batched_tokens = 0
        prefill_max_num_seqs = min(self.max_num_seqs, self.rwkv_prefill_max_batch_size)
        prefill_max_batched_tokens = min(self.max_num_batched_tokens, self.rwkv_prefill_token_budget)
        while self.waiting and num_seqs < self.max_num_seqs:
            seq = self.waiting[0]
            hit = self.prefix_index.lookup(seq.prompt_token_ids)
            cached_prefix_len = hit.prefix_len if hit is not None else 0
            exact_hit = hit.exact if hit is not None else False
            num_new_tokens = len(seq.prompt_token_ids) - cached_prefix_len
            required_slots = 1 if exact_hit else 2
            pinned_slot = None
            if hit is not None:
                self.slot_manager.pin_cached(hit.slot_id)
                pinned_slot = hit.slot_id
            if not self.slot_manager.can_allocate_n(required_slots):
                if pinned_slot is not None:
                    self.slot_manager.unpin_cached(pinned_slot)
                break
            if scheduled_seqs:
                if num_seqs >= prefill_max_num_seqs or num_batched_tokens + num_new_tokens > prefill_max_batched_tokens:
                    if pinned_slot is not None:
                        self.slot_manager.unpin_cached(pinned_slot)
                    break
            elif not scheduled_seqs:
                if num_new_tokens <= prefill_max_batched_tokens and num_seqs >= prefill_max_num_seqs:
                    if pinned_slot is not None:
                        self.slot_manager.unpin_cached(pinned_slot)
                    break
            if num_batched_tokens + num_new_tokens > self.max_num_batched_tokens:
                if pinned_slot is not None:
                    self.slot_manager.unpin_cached(pinned_slot)
                break
            prompt_slot = hit.slot_id if exact_hit else self.slot_manager.allocate_writable_slot(requires_zero_init=hit is None)
            if prompt_slot is None:
                if pinned_slot is not None:
                    self.slot_manager.unpin_cached(pinned_slot)
                break
            if not exact_hit and prompt_slot.evicted_slot_id is not None:
                self.prefix_index.remove_slot(prompt_slot.evicted_slot_id)
            state_slot = self.slot_manager.allocate_writable_slot(requires_zero_init=False)
            if state_slot is None:
                if not exact_hit:
                    self.slot_manager.release_live(prompt_slot.slot_id)
                if pinned_slot is not None:
                    self.slot_manager.unpin_cached(pinned_slot)
                break
            if state_slot.evicted_slot_id is not None:
                self.prefix_index.remove_slot(state_slot.evicted_slot_id)
            num_seqs += 1
            num_batched_tokens += num_new_tokens
            seq.status = SequenceStatus.RUNNING
            seq.cache_hit_slot = hit.slot_id if hit is not None else None
            seq.cached_prefix_len = cached_prefix_len
            seq.num_cached_tokens = cached_prefix_len
            seq.exact_cache_hit = exact_hit
            seq.prompt_cache_slot = hit.slot_id if exact_hit else prompt_slot.slot_id
            seq.state_slot = state_slot.slot_id
            seq.final_cache_published = False
            seq.state_slot_materialized = False
            self.waiting.popleft()
            self.running.append(seq)
            scheduled_seqs.append(seq)
        if scheduled_seqs:
            return scheduled_seqs, True

        while self.running and num_seqs < self.max_num_seqs:
            seq = self.running.popleft()
            num_seqs += 1
            scheduled_seqs.append(seq)
        assert scheduled_seqs
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False

    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        if self.rwkv_state_cache_enable:
            if seq.state_slot is not None:
                self.slot_manager.release_live(seq.state_slot)
            if seq.prompt_cache_slot is not None and not seq.exact_cache_hit:
                self.slot_manager.release_live(seq.prompt_cache_slot)
            if seq.cache_hit_slot is not None:
                self.slot_manager.unpin_cached(seq.cache_hit_slot)
            seq.state_slot = None
            seq.prompt_cache_slot = None
            seq.cache_hit_slot = None
            seq.cached_prefix_len = 0
            seq.num_cached_tokens = 0
            seq.exact_cache_hit = False
            seq.state_slot_materialized = False
            seq.active_state_slot = None
            self.waiting.appendleft(seq)
            return
        self.block_manager.deallocate(seq)
        seq.active_state_slot = None
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int]) -> list[bool]:
        for seq, token_id in zip(seqs, token_ids):
            seq.append_token(token_id)
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                if self.rwkv_state_cache_enable:
                    if not seq.final_cache_published and seq.state_slot is not None:
                        self.slot_manager.release_live(seq.state_slot)
                    seq.state_slot = None
                    seq.prompt_cache_slot = None
                    seq.cache_hit_slot = None
                    seq.cached_prefix_len = 0
                    seq.num_cached_tokens = 0
                    seq.exact_cache_hit = False
                    seq.final_cache_published = False
                    seq.state_slot_materialized = False
                    seq.active_state_slot = None
                else:
                    self.block_manager.deallocate(seq)
                    seq.active_state_slot = None
                self.running.remove(seq)
