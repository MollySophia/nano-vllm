from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple

from collections import defaultdict


RuleCondition = Callable[["ParserContext"], bool]
RuleAction = Callable[["ParserContext"], None]


@dataclass(slots=True)
class ParserContext:
    flags: dict[str, bool] = field(default_factory=dict)
    counters: dict[str, int] = field(default_factory=dict)
    data: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TransitionRule:
    current_state: str
    trigger: str
    next_state: str
    keep_trigger: bool | str = False
    condition: RuleCondition | None = None
    action: RuleAction | None = None


class TrieNode:
    __slots__ = ["children", "rules"]

    def __init__(self):
        self.children: Dict[str, TrieNode] = {}
        self.rules: list[TransitionRule] = []


class StreamingStringParser:
    def __init__(
        self,
        rules: List[Tuple[str, str, str, bool | str] | TransitionRule] = None,
        start_state: str = "content",
        tries: Dict[str, TrieNode] | None = None,
        context: ParserContext | None = None,
    ):
        """
        初始化解析器

        :param rules: 状态转移规则列表，每个规则为元组 (current_state, trigger, next_state, keep_trigger)
                      - current_state: 当前状态
                      - trigger: 触发字符串
                      - next_state: 目标状态
                      - keep_trigger: 是否保留触发字符串，False表示不保留触发字符串，"left"表示保留为转移之前的状态，"right"表示保留为转移之后的状态

        :param start_state: 初始状态，默认为"content"
        """
        self.current_state = start_state
        self.buffer = ""
        self.context = context if context is not None else ParserContext()

        # rules 和 tires 有且只有一个不为None
        assert tries is not None or rules is not None, "tries or rules must be provided"

        if tries is not None:
            self.tries = tries
        else:
            self.tries = self.build_trie(rules)

        # 设置初始状态的当前节点
        self.current_node = self.tries.get(self.current_state, TrieNode())

    @staticmethod
    def build_trie(
        rules: List[Tuple[str, str, str, bool | str] | TransitionRule],
    ) -> Dict[str, TrieNode]:
        """
        构建Trie树

        :param rules: 状态转移规则列表，每个规则为元组 (current_state, trigger, next_state, keep_trigger)
                      - current_state: 当前状态
                      - trigger: 触发字符串
                      - next_state: 目标状态
                      - keep_trigger: 是否保留触发字符串，False表示不保留触发字符串，"left"表示保留为转移之前
                      "right"表示保留为转移之后

        """
        rules_by_state = defaultdict(list)
        for rule in rules:
            normalized_rule = StreamingStringParser._normalize_rule(rule)
            rules_by_state[normalized_rule.current_state].append(normalized_rule)

        tries = {}
        for state, rule_list in rules_by_state.items():
            root = tries[state] if state in tries else TrieNode()
            for rule in rule_list:
                node = root
                for char in rule.trigger:
                    if char not in node.children:
                        node.children[char] = TrieNode()
                    node = node.children[char]
                assert rule.keep_trigger in [
                    False,
                    "left",
                    "right",
                ], "keep_trigger must be False, 'left' or 'right'"
                node.rules.append(rule)
            tries[state] = root
        return tries

    @staticmethod
    def _normalize_rule(
        rule: Tuple[str, str, str, bool | str] | TransitionRule,
    ) -> TransitionRule:
        if isinstance(rule, TransitionRule):
            return rule
        current_state, trigger, next_state, keep_trigger = rule
        return TransitionRule(
            current_state=current_state,
            trigger=trigger,
            next_state=next_state,
            keep_trigger=keep_trigger,
        )

    def parse(self, delta: str) -> List[Tuple[str, str]]:
        """
        流式解析字符串片段

        :param delta: 输入的字符串片段
        :return: 元组列表 [(output, state), ...]
        """
        outputs = []
        i = 0
        n = len(delta)

        while i < n:
            char = delta[i]

            # 检查当前字符是否在Trie路径中
            if char in self.current_node.children:
                self.buffer += char
                self.current_node = self.current_node.children[char]
                i += 1

                # 检查是否到达完整触发字符串
                if self.current_node.rules:
                    selected_rule = None
                    for rule in self.current_node.rules:
                        if rule.condition is None or rule.condition(self.context):
                            selected_rule = rule
                            break

                    if selected_rule is None:
                        outputs.append((self.buffer, self.current_state))
                        self.buffer = ""
                        self.current_node = self.tries.get(self.current_state, TrieNode())
                        continue

                    next_state = selected_rule.next_state
                    keep_trigger = selected_rule.keep_trigger
                    trigger_str = self.buffer

                    if keep_trigger:
                        outputs.append(
                            (
                                trigger_str,
                                (self.current_state if keep_trigger == "left" else next_state),
                            )
                        )
                    if selected_rule.action is not None:
                        selected_rule.action(self.context)
                    # 执行状态转移
                    self.current_state = next_state
                    self.current_node = self.tries.get(next_state, TrieNode())
                    self.buffer = ""
                    # 继续处理后续字符
            else:
                # 匹配失败：处理缓冲区中的部分匹配内容
                if self.buffer:
                    outputs.append((self.buffer, self.current_state))
                    self.buffer = ""
                    # 重置到当前状态的Trie根节点
                    self.current_node = self.tries.get(self.current_state, TrieNode())
                    # 不增加i，重新处理当前字符
                else:
                    # 无缓冲区内容：直接输出当前字符
                    outputs.append((char, self.current_state))
                    i += 1

        groups = []
        current_key = None
        current_chars = []
        for char, key in outputs:
            if key == current_key:
                current_chars.append(char)
            else:
                if current_key is not None:
                    groups.append(("".join(current_chars), current_key))
                current_key = key
                current_chars = [char]
        if current_chars:
            groups.append(("".join(current_chars), current_key))
        return groups

    def flush(self) -> List[Tuple[str, str]]:
        if not self.buffer:
            return []
        buffered = self.buffer
        self.buffer = ""
        self.current_node = self.tries.get(self.current_state, TrieNode())
        return [(buffered, self.current_state)]


def _flag_unset(name: str) -> RuleCondition:
    return lambda context: not context.flags.get(name, False)


def _set_flag(name: str) -> RuleAction:
    return lambda context: context.flags.__setitem__(name, True)


TRIE_THINK_NO_TRIGGER = StreamingStringParser.build_trie(
    [
        TransitionRule(
            current_state="content",
            trigger="<think>",
            next_state="reasoning_content",
            keep_trigger=False,
            condition=_flag_unset("think_seen"),
            action=_set_flag("think_seen"),
        ),
        ("reasoning_content", "</think>", "content", False),
        ("content", "\n\n", "end", "right"),
    ]
)

TRIE_THINK_KEEP_TRIGGER = StreamingStringParser.build_trie(
    [
        ("content", "<think>", "reasoning_content", "right"),
        ("reasoning_content", "</think>", "content", "left"),
        ("content", "\n\n", "end", "right"),
    ]
)
