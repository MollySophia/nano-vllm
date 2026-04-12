from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence as SequenceABC

import torch
from torch import nn

import nanovllm.ops.rapid_sampling as rapid_sampling

GREEDY_TEMPERATURE_EPS = 1e-4
RAPID_SAMPLING_MIN_TEMPERATURE = 1e-3
DEFAULT_RAPID_SAMPLING_SEED = 42


class Sampler(nn.Module):

    def __init__(
        self,
        temperature_bucket_resolution: float = 0.0,
        top_p_bucket_resolution: float = 0.0,
        rapid_sampling_seed: int = DEFAULT_RAPID_SAMPLING_SEED,
    ):
        super().__init__()
        self.temperature_bucket_resolution = float(temperature_bucket_resolution)
        self.top_p_bucket_resolution = float(top_p_bucket_resolution)
        self.rapid_sampling_seed = int(rapid_sampling_seed)
        self._rand_states: dict[tuple[str, int], torch.Tensor] = {}

    def forward(
        self,
        logits: torch.Tensor,
        seqs_or_temperatures,
        slot_penalties: torch.Tensor | None = None,
        slot_ids: SequenceABC[int] | torch.Tensor | None = None,
    ):
        if isinstance(seqs_or_temperatures, torch.Tensor) or seqs_or_temperatures is None:
            return self._forward_temperatures(logits, seqs_or_temperatures)
        return self._forward_sequences(logits, list(seqs_or_temperatures), slot_penalties=slot_penalties, slot_ids=slot_ids)

    def _forward_temperatures(self, logits: torch.Tensor, temperatures: torch.Tensor | None) -> torch.Tensor:
        if temperatures is None:
            return logits.argmax(dim=-1)
        if torch.all(temperatures <= GREEDY_TEMPERATURE_EPS):
            return logits.argmax(dim=-1)

        logits = logits.float().div_(temperatures.unsqueeze(dim=1))
        probs = torch.softmax(logits, dim=-1)
        sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
        return sample_tokens

    def _forward_sequences(
        self,
        logits: torch.Tensor,
        seqs: list,
        slot_penalties: torch.Tensor | None = None,
        slot_ids: SequenceABC[int] | torch.Tensor | None = None,
    ) -> torch.Tensor:
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)
        if logits.size(0) != len(seqs):
            raise ValueError(f"logits batch size {logits.size(0)} does not match number of sequences {len(seqs)}")
        if logits.size(0) == 0:
            return torch.empty((0,), dtype=torch.int64, device=logits.device)

        outputs = torch.empty(logits.size(0), dtype=torch.int64, device=logits.device)
        for sampling_config, indices in self._group_indices_by_sampling(seqs):
            index_tensor = torch.tensor(indices, dtype=torch.int64, device=logits.device)
            group_logits = logits.index_select(0, index_tensor)
            mode, temperature, top_k, top_p, presence_penalty, repetition_penalty, penalty_decay = sampling_config
            if mode == "greedy":
                sampled = group_logits.argmax(dim=-1)
            elif presence_penalty != 0.0 or repetition_penalty != 0.0:
                sampled = self._sample_with_penalties(
                    group_logits,
                    [seqs[idx] for idx in indices],
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    presence_penalty=presence_penalty,
                    repetition_penalty=repetition_penalty,
                    penalty_decay=penalty_decay,
                    slot_penalties=slot_penalties,
                    slot_ids=slot_ids,
                    indices=indices,
                )
            else:
                sampled = self._sample_without_penalties(
                    group_logits,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                )
            outputs.index_copy_(0, index_tensor, sampled.to(dtype=torch.int64))
        return outputs

    def _sample_without_penalties(
        self,
        logits: torch.Tensor,
        *,
        temperature: float,
        top_k: int,
        top_p: float,
    ) -> torch.Tensor:
        if self._can_use_rapid_sampling(logits, ("sample", temperature, top_k, top_p, 0.0, 0.0, 1.0)):
            rapid_logits = logits.float().contiguous()
            rand_states = self._ensure_rand_states(rapid_logits.size(0), rapid_logits.device)
            return rapid_sampling.batch_sampling_temperature_topk_topp(
                rapid_logits,
                rand_states,
                temperature,
                top_k,
                top_p,
            )
        return self._sample_top_k_top_p(logits, temperature=temperature, top_k=top_k, top_p=top_p)

    def _sample_with_penalties(
        self,
        logits: torch.Tensor,
        seqs: list,
        *,
        temperature: float,
        top_k: int,
        top_p: float,
        presence_penalty: float,
        repetition_penalty: float,
        penalty_decay: float,
        slot_penalties: torch.Tensor | None,
        slot_ids: SequenceABC[int] | torch.Tensor | None,
        indices: list[int],
    ) -> torch.Tensor:
        if slot_penalties is not None:
            if slot_ids is None:
                raise ValueError("slot_ids are required when slot_penalties are provided.")
            return self._sample_with_dense_slot_penalties(
                logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                presence_penalty=presence_penalty,
                repetition_penalty=repetition_penalty,
                penalty_decay=penalty_decay,
                slot_penalties=slot_penalties,
                slot_ids=slot_ids,
                indices=indices,
            )
        if not all(getattr(seq, "allow_sparse_penalty_state", False) for seq in seqs):
            raise ValueError("penalty sampling requires slot state or explicit sparse penalty-state opt-in.")
        return self._sample_with_sequence_penalties(
            logits,
            seqs,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            presence_penalty=presence_penalty,
            repetition_penalty=repetition_penalty,
            penalty_decay=penalty_decay,
        )

    def _sample_with_dense_slot_penalties(
        self,
        logits: torch.Tensor,
        *,
        temperature: float,
        top_k: int,
        top_p: float,
        presence_penalty: float,
        repetition_penalty: float,
        penalty_decay: float,
        slot_penalties: torch.Tensor,
        slot_ids: SequenceABC[int] | torch.Tensor,
        indices: list[int],
    ) -> torch.Tensor:
        slot_index = self._resolve_slot_index_tensor(slot_ids, indices, logits.device)
        local_penalties = slot_penalties.index_select(0, slot_index.to(slot_penalties.device))
        if self._can_use_rapid_sampling(
            logits,
            ("sample", temperature, top_k, top_p, presence_penalty, repetition_penalty, penalty_decay),
        ):
            rapid_logits = logits.float().contiguous()
            local_penalties = local_penalties.to(device=rapid_logits.device, dtype=torch.float32)
            rand_states = self._ensure_rand_states(rapid_logits.size(0), rapid_logits.device)
            sampled = rapid_sampling.batch_sampling_repetition_temperature_topk_topp(
                rapid_logits,
                local_penalties,
                rand_states,
                presence_penalty,
                repetition_penalty,
                penalty_decay,
                temperature,
                top_k,
                top_p,
            )
        else:
            penalized_logits = logits.float() - local_penalties.to(device=logits.device, dtype=torch.float32)
            sampled = self._sample_top_k_top_p(
                penalized_logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
            self._update_dense_penalties(
                local_penalties,
                sampled,
                presence_penalty=presence_penalty,
                repetition_penalty=repetition_penalty,
                penalty_decay=penalty_decay,
            )
        slot_penalties.index_copy_(0, slot_index.to(slot_penalties.device), local_penalties.to(slot_penalties.device))
        return sampled

    def _sample_with_sequence_penalties(
        self,
        logits: torch.Tensor,
        seqs: list,
        *,
        temperature: float,
        top_k: int,
        top_p: float,
        presence_penalty: float,
        repetition_penalty: float,
        penalty_decay: float,
    ) -> torch.Tensor:
        outputs = torch.empty(logits.size(0), dtype=torch.int64, device=logits.device)
        for row, seq in enumerate(seqs):
            adjusted_logits = logits[row].float().clone()
            for token_id, penalty in seq.penalty_state.items():
                adjusted_logits[token_id] -= penalty
            token = self._sample_top_k_top_p(
                adjusted_logits.unsqueeze(0),
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )[0]
            outputs[row] = token
            self._update_sparse_penalties(
                seq.penalty_state,
                int(token.item()),
                presence_penalty=presence_penalty,
                repetition_penalty=repetition_penalty,
                penalty_decay=penalty_decay,
            )
        return outputs

    def _resolve_slot_index_tensor(
        self,
        slot_ids: SequenceABC[int] | torch.Tensor,
        indices: list[int],
        device: torch.device,
    ) -> torch.Tensor:
        if isinstance(slot_ids, torch.Tensor):
            slot_ids_tensor = slot_ids.to(device=device, dtype=torch.int64)
        else:
            slot_ids_tensor = torch.tensor(list(slot_ids), dtype=torch.int64, device=device)
        return slot_ids_tensor.index_select(0, torch.tensor(indices, dtype=torch.int64, device=device))

    def _update_dense_penalties(
        self,
        penalties: torch.Tensor,
        token_ids: torch.Tensor,
        *,
        presence_penalty: float,
        repetition_penalty: float,
        penalty_decay: float,
    ) -> None:
        token_ids = token_ids.to(device=penalties.device, dtype=torch.int64)
        existing = penalties.gather(1, token_ids.unsqueeze(1))
        penalties.mul_(penalty_decay)
        additions = torch.where(
            existing == 0,
            torch.full_like(existing, presence_penalty),
            torch.full_like(existing, repetition_penalty),
        )
        penalties.scatter_add_(1, token_ids.unsqueeze(1), additions)

    def _update_sparse_penalties(
        self,
        penalty_state: dict[int, float],
        token_id: int,
        *,
        presence_penalty: float,
        repetition_penalty: float,
        penalty_decay: float,
    ) -> None:
        for key in list(penalty_state.keys()):
            penalty_state[key] *= penalty_decay
            if abs(penalty_state[key]) < 1e-8:
                penalty_state.pop(key)
        existing = penalty_state.get(token_id, 0.0)
        penalty_state[token_id] = existing + (presence_penalty if existing == 0.0 else repetition_penalty)

    def _sample_top_k_top_p(
        self,
        logits: torch.Tensor,
        *,
        temperature: float,
        top_k: int,
        top_p: float,
    ) -> torch.Tensor:
        scaled_logits = logits.float()
        if temperature <= GREEDY_TEMPERATURE_EPS:
            return scaled_logits.argmax(dim=-1)
        if temperature > 0.0 and temperature != 1.0:
            scaled_logits = scaled_logits.div(temperature)
        vocab_size = scaled_logits.size(-1)
        top_k, top_p = self._normalize_top_k_top_p(vocab_size, top_k, top_p)
        if top_k == 1:
            return scaled_logits.argmax(dim=-1)
        if top_k < vocab_size:
            threshold = scaled_logits.topk(top_k, dim=-1).values[..., -1:].clone()
            scaled_logits = scaled_logits.masked_fill(scaled_logits < threshold, float("-inf"))
        probs = torch.softmax(scaled_logits, dim=-1)
        if top_p < 1.0:
            sorted_probs, sorted_indices = probs.sort(dim=-1, descending=True)
            cumulative = sorted_probs.cumsum(dim=-1)
            remove_mask = cumulative > top_p
            remove_mask[..., 0] = False
            sorted_probs = sorted_probs.masked_fill(remove_mask, 0.0)
            probs = torch.zeros_like(probs).scatter(-1, sorted_indices, sorted_probs)
            denom = probs.sum(dim=-1, keepdim=True)
            probs = torch.where(denom > 0, probs / denom, torch.zeros_like(probs))
        noise = torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)
        return probs.div_(noise).argmax(dim=-1)

    def _group_indices_by_sampling(self, seqs: list) -> list[tuple[tuple, list[int]]]:
        groups: OrderedDict[tuple, list[int]] = OrderedDict()
        for idx, seq in enumerate(seqs):
            sampling_config = self._sampling_config_for_seq(seq)
            groups.setdefault(sampling_config, []).append(idx)
        return list(groups.items())

    def _sampling_config_for_seq(self, seq) -> tuple:
        uses_penalties = seq.presence_penalty != 0.0 or seq.repetition_penalty != 0.0
        if seq.temperature <= GREEDY_TEMPERATURE_EPS and not uses_penalties:
            return ("greedy", 0.0, 1, 0.0, 0.0, 0.0, 1.0)
        temperature = self._bucket_value(seq.temperature, self.temperature_bucket_resolution)
        top_p = self._bucket_value(seq.top_p, self.top_p_bucket_resolution)
        return (
            "sample",
            temperature,
            int(seq.top_k),
            top_p,
            float(seq.presence_penalty),
            float(seq.repetition_penalty),
            float(seq.penalty_decay),
        )

    def _bucket_value(self, value: float, resolution: float) -> float:
        value = float(value)
        if resolution <= 0:
            return value
        return round(round(value / resolution) * resolution, 10)

    def _normalize_top_k_top_p(self, vocab_size: int, top_k: int, top_p: float) -> tuple[int, float]:
        if top_k <= 0 or top_k > vocab_size:
            top_k = vocab_size
        if top_p < 0.0 or top_p > 1.0:
            top_p = 1.0
        if top_p == 0.0:
            return 1, 1.0
        return top_k, top_p

    def _can_use_rapid_sampling(self, group_logits: torch.Tensor, sampling_config: tuple) -> bool:
        _, temperature, _, _, _, _, _ = sampling_config
        if not group_logits.is_cuda:
            return False
        if group_logits.dim() != 2:
            return False
        if group_logits.size(-1) % 4 != 0:
            return False
        if temperature < RAPID_SAMPLING_MIN_TEMPERATURE:
            return False
        return True

    def _ensure_rand_states(self, batch_size: int, device: torch.device) -> torch.Tensor:
        if device.type != "cuda":
            raise ValueError("rapid sampling random states require a CUDA device")
        key = (str(device), int(batch_size))
        state = self._rand_states.get(key)
        if state is None or state.device != device:
            state = rapid_sampling.setup_rand(self.rapid_sampling_seed, batch_size)
            if state.device != device:
                state = state.to(device=device)
            self._rand_states[key] = state
        return state
