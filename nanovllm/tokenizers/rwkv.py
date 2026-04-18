import ast
from functools import lru_cache
from importlib.resources import files

import torch


class TRIE:
    __slots__ = ("ch", "to", "values", "front")

    def __init__(self, front=None, ch=None):
        self.ch = ch
        self.to = [None for _ in range(256)]
        self.values = set()
        self.front = front

    def add(self, key: bytes, idx: int = 0, val=None):
        if idx == len(key):
            if val is None:
                val = key
            self.values.add(val)
            return self
        ch = key[idx]
        if self.to[ch] is None:
            self.to[ch] = TRIE(front=self, ch=ch)
        return self.to[ch].add(key, idx=idx + 1, val=val)

    def find_longest(self, key: bytes, idx: int = 0):
        node = self
        ch = key[idx]
        match = None
        while node.to[ch] is not None:
            node = node.to[ch]
            idx += 1
            if node.values:
                match = idx, node.values
            if idx == len(key):
                break
            ch = key[idx]
        if match is None:
            raise ValueError("Failed to match bytes in RWKV tokenizer trie.")
        return match


class RWKVTokenizer:
    eos_token_id = 0
    eod_token = b"<|rwkv_end_of_text|>"
    eod_token_aliases = (b"<|rwkv_end_of_text|>", b"<|endoftext|>")
    state_cache_id_replacements = (
        ((10080, 261), (28329, 11)),
        ((9830, 261), (28324, 11)),
        ((19137, 261), (28331, 11)),
    )
    stop_token_seqs_by_eos_text = {
        "\n\n": (
            (261,),
            (28329, 11),
            (28324, 11),
            (28331, 11),
            (5585,),
        ),
        "\n": (
            (11,),
            (28329,),
            (28324,),
            (28331,),
            (261,),
            (5585,),
        ),
    }

    def __init__(
        self,
        vocab_file=None,
        *,
        user_role: str = "User",
        assistant_role: str = "Assistant",
        system_role: str = "System",
        bos_token: str = "",
        eos_token: str = "\n\n",
        prompt_prefix: str = "",
        space_after_roles: bool = True,
    ):
        if vocab_file is None:
            vocab_file = files("nanovllm.tokenizers").joinpath("rwkv_vocab_v20230424.txt")
        self.idx2token = {0: self.eod_token}
        with vocab_file.open("r", encoding="utf-8") as f:
            for line in f:
                idx = int(line[:line.index(" ")])
                token = ast.literal_eval(line[line.index(" "):line.rindex(" ")])
                token = token.encode("utf-8") if isinstance(token, str) else token
                assert isinstance(token, bytes)
                assert len(token) == int(line[line.rindex(" "):])
                self.idx2token[idx] = token

        self.token2idx = {token: int(idx) for idx, token in self.idx2token.items() if idx != 0}
        for token in self.eod_token_aliases:
            self.token2idx[token] = 0
        self.root = TRIE()
        for token, idx in self.token2idx.items():
            self.root.add(token, val=idx)
        self.vocab_size = len(self.idx2token)
        self.user_role = user_role
        self.assistant_role = assistant_role
        self.system_role = system_role
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.prompt_prefix = prompt_prefix
        self.space_after_roles = space_after_roles

    def encode_bytes(self, src: bytes):
        idx = 0
        token_ids = []
        while idx < len(src):
            prev_idx = idx
            idx, values = self.root.find_longest(src, idx)
            assert idx != prev_idx
            token_ids.append(next(iter(values)))
        return token_ids

    def decode_bytes(self, token_ids):
        return b"".join(self.idx2token[token_id] for token_id in token_ids)

    def encode(self, src: str, **kwargs):
        return self.encode_bytes(src.encode("utf-8"))

    def decode(self, token_ids, utf8_errors: str = "strict", **kwargs):
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        return self.decode_bytes(token_ids).decode("utf-8", errors=utf8_errors)

    @classmethod
    def canonicalize_state_cache_token_ids(cls, token_ids: list[int] | tuple[int, ...]) -> list[int]:
        canonical_ids: list[int] = []
        for token_id in token_ids:
            canonical_ids.append(int(token_id))
            if len(canonical_ids) < 2:
                continue
            for source, replacement in cls.state_cache_id_replacements:
                if tuple(canonical_ids[-2:]) == source:
                    canonical_ids[-2:] = replacement
                    break
        return canonical_ids

    def get_default_stop_token_seqs(self) -> tuple[tuple[int, ...], ...]:
        return tuple(tuple(seq) for seq in self.stop_token_seqs_by_eos_text.get(self.eos_token, ()))

    @staticmethod
    def _message_field(message, key: str):
        if isinstance(message, dict):
            return message.get(key)
        return getattr(message, key)

    def _normalize_chat_role(self, role: str) -> str:
        if role == "user":
            return self.user_role
        if role == "assistant":
            return self.assistant_role
        if role == "system":
            return self.system_role
        return role

    @staticmethod
    def _coerce_chat_content(content) -> str:
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            chunks: list[str] = []
            for part in content:
                if isinstance(part, dict):
                    part_type = part.get("type")
                    part_text = part.get("text")
                else:
                    part_type = getattr(part, "type", None)
                    part_text = getattr(part, "text", None)
                if part_type != "text":
                    raise ValueError(f"Unsupported chat content part type: {part_type!r}")
                chunks.append(part_text or "")
            return "".join(chunks)
        return str(content)

    @staticmethod
    def _collapse_chat_newlines(text: str) -> str:
        text = text.replace("\r\n", "\n")
        while "\n\n" in text:
            text = text.replace("\n\n", "\n")
        return text

    @staticmethod
    def format_role_line(
        role: str,
        content: str | None = None,
        *,
        bos_token: str = "",
        space_after_role: bool = True,
    ) -> str:
        if content is None:
            return f"{bos_token}{role}:"
        separator = ": " if space_after_role else ":"
        return f"{bos_token}{role}{separator}{content}"

    def apply_chat_template(
        self,
        messages,
        tokenize: bool = False,
        add_generation_prompt: bool = True,
    ):
        rendered_messages: list[tuple[str, str]] = []
        for message in messages:
            raw_role = self._message_field(message, "role")
            raw_content = self._message_field(message, "content")
            role = self._normalize_chat_role(str(raw_role or "user"))
            content = self._coerce_chat_content(raw_content)
            if role in {self.user_role, "User", self.system_role, "System"}:
                content = self._collapse_chat_newlines(content)
            rendered_messages.append((role, content))

        prompt_text = self.prompt_prefix
        for index, (role, content) in enumerate(rendered_messages):
            prompt_text += self.format_role_line(
                role,
                content,
                bos_token=self.bos_token,
                space_after_role=self.space_after_roles,
            )
            if index != len(rendered_messages) - 1:
                prompt_text += self.eos_token

        if rendered_messages and add_generation_prompt:
            prompt_text += self.eos_token
            last_role, _ = rendered_messages[-1]
            if last_role == self.user_role:
                prompt_text += self.format_role_line(
                    self.assistant_role,
                    bos_token=self.bos_token,
                )
            elif last_role == self.assistant_role:
                prompt_text += self.format_role_line(
                    self.user_role,
                    bos_token=self.bos_token,
                )

        if tokenize:
            return self.encode(prompt_text)
        return prompt_text


@lru_cache(maxsize=1)
def get_rwkv_tokenizer():
    return RWKVTokenizer()
