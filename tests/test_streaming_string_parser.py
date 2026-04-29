import unittest

from nanovllm.entrypoints.openai.streaming_string_parser import (
    StreamingStringParser,
    TRIE_THINK_NO_TRIGGER,
)


class StreamingStringParserTest(unittest.TestCase):
    def test_parse_preserves_partial_triggers_across_chunks(self):
        parser = StreamingStringParser(tries=TRIE_THINK_NO_TRIGGER)

        self.assertEqual(parser.parse("<think"), [])
        self.assertEqual(parser.parse(">abc</"),[("abc", "reasoning_content")])
        self.assertEqual(parser.parse("think>abc"),[("abc", "content")])
        self.assertEqual(parser.parse("def\n"),[("def", "content")])
        self.assertEqual(parser.parse("\nUser"), [('\n\nUser', 'end')])

    def test_parse_only_enters_think_state_once(self):
        parser = StreamingStringParser(tries=TRIE_THINK_NO_TRIGGER)

        self.assertEqual(parser.parse("<think>abc</think><th"), [("abc", "reasoning_content")])
        self.assertEqual(parser.parse("ink>xyz"), [("<think>xyz", "content")])

    def test_flush_releases_unmatched_partial_trigger(self):
        parser = StreamingStringParser(tries=TRIE_THINK_NO_TRIGGER)

        self.assertEqual(parser.parse("<thi"), [])
        self.assertEqual(parser.flush(), [("<thi", "content")])


if __name__ == "__main__":
    unittest.main()