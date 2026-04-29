import re


_FENCE_RE = re.compile(r"^\s*(```+|~~~+)")
_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+")
_LIST_RE = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+")
_TABLE_SEPARATOR_RE = re.compile(r"^\s*\|?(?:\s*:?-+:?\s*\|)+\s*:?-+:?\s*\|?\s*$")


class StreamingMarkdownRestorer:
    def __init__(self):
        self._line_buffer = ""
        self._current_line_kind = None
        self._pending_table_line = None
        self._current_block_kind = None
        self._has_output = False
        self._last_output_ended_with_newline = False
        self._last_emitted_blank_line = False
        self._inside_fence = False
        self._fence_marker = ""
        self._inside_display_math = False

    def parse(self, delta: str) -> str:
        if not delta:
            return ""

        parts: list[str] = []
        self._consume_text(delta, parts)
        return "".join(parts)

    def flush(self) -> str:
        parts: list[str] = []
        if self._pending_table_line is not None:
            pending_line = self._pending_table_line
            self._pending_table_line = None
            pending_kind = "table" if self._current_block_kind == "table" else "paragraph"
            self._emit_line(pending_line, pending_kind, parts)

        if self._line_buffer:
            if self._current_line_kind is None:
                self._emit_line(self._line_buffer, self._resolve_flush_kind(self._line_buffer), parts)
            else:
                self._apply_line_end_state(self._line_buffer)
            self._line_buffer = ""
            self._current_line_kind = None
        return "".join(parts)

    def reset(self) -> None:
        self.__init__()

    def _consume_text(self, text: str, parts: list[str]) -> None:
        for char in text:
            self._consume_char(char, parts)

    def _consume_char(self, char: str, parts: list[str]) -> None:
        self._line_buffer += char

        if self._pending_table_line is not None:
            if char == "\n":
                self._resolve_pending_table(parts)
            return

        if self._current_line_kind is None:
            resolved_kind = self._try_resolve_line_kind(self._line_buffer)
            if resolved_kind is not None:
                self._emit_line(self._line_buffer, resolved_kind, parts)
                return

            if char == "\n":
                if self._is_blank_line(self._line_buffer):
                    self._emit_blank_line(parts)
                    self._line_buffer = ""
                    self._current_line_kind = None
                    return
                if self._is_table_candidate(self._line_buffer):
                    self._pending_table_line = self._line_buffer
                    self._line_buffer = ""
                    self._current_line_kind = None
                    return
                self._emit_line(self._line_buffer, self._resolve_flush_kind(self._line_buffer), parts)
            return

        parts.append(char)
        self._has_output = True
        self._last_output_ended_with_newline = char == "\n"
        self._last_emitted_blank_line = False
        if char == "\n":
            self._apply_line_end_state(self._line_buffer)
            self._line_buffer = ""
            self._current_line_kind = None

    def _resolve_pending_table(self, parts: list[str]) -> None:
        next_line = self._line_buffer
        pending_line = self._pending_table_line
        self._pending_table_line = None
        self._line_buffer = ""
        self._current_line_kind = None

        if self._current_block_kind == "table":
            if self._is_table_separator(next_line.strip()):
                self._emit_line(pending_line, "table", parts, force_new_block=True)
                self._emit_line(next_line, "table", parts, force_new_block=False)
                return

            self._emit_line(pending_line, "table", parts, force_new_block=False)
            self._consume_text(next_line, parts)
            return

        if self._is_table_separator(next_line.strip()):
            self._emit_line(pending_line, "table", parts)
            self._emit_line(next_line, "table", parts, force_new_block=False)
            return

        self._emit_line(pending_line, "paragraph", parts)
        self._consume_text(next_line, parts)

    def _emit_blank_line(self, parts: list[str]) -> None:
        if not self._has_output or self._last_emitted_blank_line:
            self._current_block_kind = None
            return
        parts.append("\n" if self._last_output_ended_with_newline else "\n\n")
        self._has_output = True
        self._last_output_ended_with_newline = True
        self._last_emitted_blank_line = True
        self._current_block_kind = None

    def _emit_line(
        self,
        line: str,
        kind: str,
        parts: list[str],
        force_new_block: bool | None = None,
    ) -> None:
        if kind == "blank":
            self._emit_blank_line(parts)
            self._line_buffer = ""
            self._current_line_kind = None
            return

        starts_new_block = self._starts_new_block(kind, line) if force_new_block is None else force_new_block
        if starts_new_block and self._has_output and not self._last_emitted_blank_line:
            self._emit_blank_line(parts)

        parts.append(line)
        self._has_output = True
        self._last_output_ended_with_newline = line.endswith("\n")
        self._last_emitted_blank_line = False
        self._current_block_kind = kind
        self._current_line_kind = kind

        if line.endswith("\n"):
            self._apply_line_end_state(line)
            self._line_buffer = ""
            self._current_line_kind = None

    def _starts_new_block(self, kind: str, line: str) -> bool:
        if kind in {"heading", "paragraph"}:
            return True
        if kind == "code_fence":
            return not self._inside_fence and self._get_fence_marker(line) is not None
        if kind == "display_math":
            return not self._inside_display_math and self._is_display_math_delimiter(line)
        return self._current_block_kind != kind

    def _try_resolve_line_kind(self, line: str) -> str | None:
        if self._inside_fence:
            return "code_fence"

        if self._inside_display_math:
            return "display_math"

        fence_marker = self._get_fence_marker(line)
        if fence_marker is not None:
            return "code_fence"

        if self._is_potential_display_math_delimiter(line):
            return None

        if self._is_heading_line(line):
            return "heading"

        if self._is_table_candidate(line):
            return None

        if self._is_blockquote_line(line):
            return "blockquote"

        if self._is_list_item_line(line):
            return "list"

        if self._can_resolve_paragraph(line):
            return "paragraph"
        return None

    def _resolve_flush_kind(self, line: str) -> str:
        if self._is_blank_line(line):
            return "blank"
        if self._is_display_math_delimiter(line):
            return "display_math"
        if self._is_table_candidate(line):
            return "paragraph"
        kind = self._try_resolve_line_kind(line)
        return "paragraph" if kind is None else kind

    def _apply_line_end_state(self, line: str) -> None:
        if self._inside_fence:
            if self._is_matching_fence_close(line):
                self._inside_fence = False
            return

        if self._inside_display_math:
            if self._is_display_math_delimiter(line):
                self._inside_display_math = False
            return

        fence_marker = self._get_fence_marker(line)
        if fence_marker is not None:
            self._inside_fence = True
            self._fence_marker = fence_marker
            return

        if self._is_display_math_delimiter(line):
            self._inside_display_math = True

    @staticmethod
    def _can_resolve_paragraph(line: str) -> bool:
        stripped = line.lstrip()
        return bool(stripped) and stripped[0] not in {"#", "-", "*", "+", ">", "|", "$", "~", "`"}

    @staticmethod
    def _is_blank_line(line: str) -> bool:
        return line.strip() == ""

    @staticmethod
    def _get_fence_marker(line: str) -> str | None:
        match = _FENCE_RE.match(line)
        if match is None:
            return None
        return match.group(1)

    def _is_matching_fence_close(self, line: str) -> bool:
        return line.lstrip().startswith(self._fence_marker)

    @staticmethod
    def _is_display_math_delimiter(line: str) -> bool:
        return line.strip() == "$$"

    @staticmethod
    def _is_potential_display_math_delimiter(line: str) -> bool:
        stripped = line.strip()
        return stripped in {"$", "$$"}

    @staticmethod
    def _is_heading_line(line: str) -> bool:
        return _HEADING_RE.match(line) is not None

    @staticmethod
    def _is_blockquote_line(line: str) -> bool:
        return line.lstrip().startswith(">")

    @staticmethod
    def _is_list_item_line(line: str) -> bool:
        return _LIST_RE.match(line) is not None

    @staticmethod
    def _is_table_candidate(line: str) -> bool:
        stripped = line.strip()
        return stripped.startswith("|") and stripped.count("|") >= 2

    @staticmethod
    def _is_table_separator(line: str) -> bool:
        return _TABLE_SEPARATOR_RE.match(line.strip()) is not None
