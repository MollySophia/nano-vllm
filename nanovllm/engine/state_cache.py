from __future__ import annotations

from collections import OrderedDict, deque
from dataclasses import dataclass
from enum import Enum, auto


class SlotState(Enum):
    FREE = auto()
    LIVE = auto()
    CACHED_EVICTABLE = auto()
    CACHED_PINNED = auto()


@dataclass
class SlotMeta:
    state: SlotState = SlotState.FREE
    cache_key: tuple[int, ...] | None = None
    prefix_len: int = 0
    pin_count: int = 0


@dataclass
class WritableSlotResult:
    slot_id: int
    evicted_slot_id: int | None
    requires_zero_init: bool


@dataclass
class PrefixCacheHit:
    slot_id: int
    prefix_len: int
    cache_key: tuple[int, ...]
    exact: bool


@dataclass
class _TrieNode:
    children: dict[int, "_TrieNode"]
    slot_id: int | None = None


class StatePrefixIndex:

    def __init__(self, cache_key_token_rewriter=None):
        self.root = _TrieNode(children={})
        self.slot_to_key: dict[int, tuple[int, ...]] = {}
        self.cache_key_token_rewriter = cache_key_token_rewriter

    def _rewrite_cache_key(self, token_ids: list[int] | tuple[int, ...]) -> list[int]:
        if self.cache_key_token_rewriter is None:
            return [int(token_id) for token_id in token_ids]
        return [int(token_id) for token_id in self.cache_key_token_rewriter(token_ids)]

    def lookup(self, token_ids: list[int]) -> PrefixCacheHit | None:
        rewritten_token_ids = self._rewrite_cache_key(token_ids)
        node = self.root
        best_slot_id = None
        best_prefix_len = 0
        for prefix_len, token_id in enumerate(rewritten_token_ids, start=1):
            node = node.children.get(token_id)
            if node is None:
                break
            if node.slot_id is not None:
                best_slot_id = node.slot_id
                best_prefix_len = prefix_len
        if best_slot_id is None:
            return None
        cache_key = self.slot_to_key[best_slot_id]
        return PrefixCacheHit(
            slot_id=best_slot_id,
            prefix_len=best_prefix_len,
            cache_key=cache_key,
            exact=best_prefix_len == len(rewritten_token_ids),
        )

    def insert(self, token_ids: list[int], prefix_len: int, slot_id: int) -> tuple[int, ...]:
        cache_key = tuple(self._rewrite_cache_key(token_ids[:prefix_len]))
        old_key = self.slot_to_key.get(slot_id)
        if old_key is not None and old_key != cache_key:
            self.remove_slot(slot_id)
        node = self.root
        for token_id in cache_key:
            child = node.children.get(token_id)
            if child is None:
                child = _TrieNode(children={})
                node.children[token_id] = child
            node = child
        old_slot = node.slot_id
        if old_slot is not None and old_slot != slot_id:
            self.slot_to_key.pop(old_slot, None)
        node.slot_id = slot_id
        self.slot_to_key[slot_id] = cache_key
        return cache_key

    def remove_slot(self, slot_id: int) -> None:
        cache_key = self.slot_to_key.pop(slot_id, None)
        if cache_key is None:
            return
        node = self.root
        stack: list[tuple[_TrieNode, int]] = []
        for token_id in cache_key:
            child = node.children.get(token_id)
            if child is None:
                return
            stack.append((node, token_id))
            node = child
        if node.slot_id == slot_id:
            node.slot_id = None
        for parent, token_id in reversed(stack):
            child = parent.children[token_id]
            if child.slot_id is None and not child.children:
                del parent.children[token_id]
            else:
                break


class StateSlotManager:

    def __init__(self, num_slots: int):
        self.num_slots = num_slots
        self.free_slots: deque[int] = deque(range(num_slots))
        self.lru_cached_slots: OrderedDict[int, None] = OrderedDict()
        self.slot_meta = [SlotMeta() for _ in range(num_slots)]

    @property
    def num_free_slots(self) -> int:
        return len(self.free_slots)

    @property
    def num_cached_evictable_slots(self) -> int:
        return len(self.lru_cached_slots)

    @property
    def num_writable_slots(self) -> int:
        return self.num_free_slots + self.num_cached_evictable_slots

    def can_allocate_n(self, n: int) -> bool:
        return self.num_writable_slots >= n

    def allocate_writable_slot(self, requires_zero_init: bool) -> WritableSlotResult | None:
        if self.free_slots:
            slot_id = self.free_slots.popleft()
            self._clear_slot(slot_id)
            self.slot_meta[slot_id].state = SlotState.LIVE
            return WritableSlotResult(
                slot_id=slot_id,
                evicted_slot_id=None,
                requires_zero_init=requires_zero_init,
            )
        if self.lru_cached_slots:
            slot_id, _ = self.lru_cached_slots.popitem(last=False)
            self._clear_slot(slot_id)
            self.slot_meta[slot_id].state = SlotState.LIVE
            return WritableSlotResult(
                slot_id=slot_id,
                evicted_slot_id=slot_id,
                requires_zero_init=requires_zero_init,
            )
        return None

    def mark_cached(self, slot_id: int, cache_key: tuple[int, ...], prefix_len: int) -> None:
        meta = self.slot_meta[slot_id]
        meta.cache_key = cache_key
        meta.prefix_len = prefix_len
        if meta.pin_count > 0:
            meta.state = SlotState.CACHED_PINNED
            self.lru_cached_slots.pop(slot_id, None)
            return
        meta.state = SlotState.CACHED_EVICTABLE
        self.lru_cached_slots.pop(slot_id, None)
        self.lru_cached_slots[slot_id] = None

    def pin_cached(self, slot_id: int) -> None:
        meta = self.slot_meta[slot_id]
        if meta.state not in (SlotState.CACHED_EVICTABLE, SlotState.CACHED_PINNED):
            return
        meta.pin_count += 1
        meta.state = SlotState.CACHED_PINNED
        self.lru_cached_slots.pop(slot_id, None)

    def unpin_cached(self, slot_id: int) -> None:
        meta = self.slot_meta[slot_id]
        if meta.state not in (SlotState.CACHED_EVICTABLE, SlotState.CACHED_PINNED):
            return
        if meta.pin_count > 0:
            meta.pin_count -= 1
        if meta.pin_count > 0:
            meta.state = SlotState.CACHED_PINNED
            return
        meta.state = SlotState.CACHED_EVICTABLE
        self.lru_cached_slots.pop(slot_id, None)
        self.lru_cached_slots[slot_id] = None

    def release_live(self, slot_id: int) -> None:
        meta = self.slot_meta[slot_id]
        if meta.state != SlotState.LIVE:
            return
        self._clear_slot(slot_id)
        self.free_slots.append(slot_id)

    def release_cached(self, slot_id: int) -> None:
        meta = self.slot_meta[slot_id]
        if meta.state not in (SlotState.CACHED_EVICTABLE, SlotState.CACHED_PINNED):
            return
        self.lru_cached_slots.pop(slot_id, None)
        self._clear_slot(slot_id)
        self.free_slots.append(slot_id)

    def _clear_slot(self, slot_id: int) -> None:
        self.lru_cached_slots.pop(slot_id, None)
        meta = self.slot_meta[slot_id]
        meta.state = SlotState.FREE
        meta.cache_key = None
        meta.prefix_len = 0
        meta.pin_count = 0
