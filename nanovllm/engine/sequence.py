from copy import copy
from enum import Enum, auto
from itertools import count

from nanovllm.sampling_params import SamplingParams


class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


class Sequence:
    counter = count()

    def __init__(self, token_ids: list[int], sampling_params: SamplingParams | None = None):
        if sampling_params is None:
            sampling_params = SamplingParams()
        self.seq_id = next(Sequence.counter)
        self.status = SequenceStatus.WAITING
        self.token_ids = copy(token_ids)
        self.last_token = token_ids[-1]
        self.num_tokens = len(self.token_ids)
        self.num_prompt_tokens = len(token_ids)
        self.num_cached_tokens = 0
        self.block_table = []
        self.state_slot: int | None = None
        self.prompt_cache_slot: int | None = None
        self.cache_hit_slot: int | None = None
        self.cached_prefix_len = 0
        self.exact_cache_hit = False
        self.final_cache_published = False
        self.state_slot_materialized = False
        self.active_state_slot: int | None = None
        self.temperature = sampling_params.temperature
        self.top_k = sampling_params.top_k
        self.top_p = sampling_params.top_p
        self.presence_penalty = sampling_params.presence_penalty
        self.repetition_penalty = sampling_params.repetition_penalty
        self.penalty_decay = sampling_params.penalty_decay
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos
        self.penalty_state: dict[int, float] = {}
        self.allow_sparse_penalty_state = False

    def __len__(self):
        return self.num_tokens

    def __getitem__(self, key):
        return self.token_ids[key]

    @property
    def is_finished(self):
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self):
        return self.num_tokens - self.num_prompt_tokens

    @property
    def prompt_token_ids(self):
        return self.token_ids[:self.num_prompt_tokens]

    @property
    def completion_token_ids(self):
        return self.token_ids[self.num_prompt_tokens:]

    def append_token(self, token_id: int):
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1

    def __getstate__(self):
        return (
                self.num_tokens,
                self.num_prompt_tokens,
                self.num_cached_tokens,
                self.block_table,
                self.state_slot,
                self.prompt_cache_slot,
                self.cache_hit_slot,
                self.cached_prefix_len,
                self.exact_cache_hit,
                self.final_cache_published,
                self.state_slot_materialized,
                self.active_state_slot,
                self.temperature,
                self.top_k,
                self.top_p,
                self.presence_penalty,
                self.repetition_penalty,
                self.penalty_decay,
                self.max_tokens,
                self.ignore_eos,
                self.penalty_state,
                self.allow_sparse_penalty_state,
                self.token_ids if self.num_completion_tokens == 0 else self.last_token)

    def __setstate__(self, state):
        (
            self.num_tokens,
            self.num_prompt_tokens,
            self.num_cached_tokens,
            self.block_table,
            self.state_slot,
            self.prompt_cache_slot,
            self.cache_hit_slot,
            self.cached_prefix_len,
            self.exact_cache_hit,
            self.final_cache_published,
            self.state_slot_materialized,
            self.active_state_slot,
            self.temperature,
            self.top_k,
            self.top_p,
            self.presence_penalty,
            self.repetition_penalty,
            self.penalty_decay,
            self.max_tokens,
            self.ignore_eos,
            self.penalty_state,
            self.allow_sparse_penalty_state,
        ) = state[:-1]
        if self.num_completion_tokens == 0:
            self.token_ids = state[-1]
            self.last_token = self.token_ids[-1]
        else:
            self.last_token = state[-1]
