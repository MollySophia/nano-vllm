import textwrap
import unittest

from nanovllm.entrypoints.openai.streaming_markdown_restorer import StreamingMarkdownRestorer


class StreamingMarkdownRestorerTest(unittest.TestCase):
    def test_parse_emits_new_block_prefix_before_line_end(self):
        restorer = StreamingMarkdownRestorer()

        self.assertEqual(restorer.parse("Intro"), "Intro")
        self.assertEqual(restorer.parse("\n"), "\n")
        self.assertEqual(restorer.parse("##"), "")
        self.assertEqual(restorer.parse(" inline"), "\n## inline")
        self.assertEqual(restorer.parse(" test"), " test")
        self.assertEqual(restorer.parse("\n"), "\n")
        self.assertEqual(restorer.parse("-"), "")
        self.assertEqual(restorer.parse(" item"), "\n- item")

    def test_parse_inserts_blank_line_between_adjacent_headings(self):
        restorer = StreamingMarkdownRestorer()

        self.assertEqual(restorer.parse("#"), "")
        self.assertEqual(restorer.parse(" title"), "# title")
        self.assertEqual(restorer.parse("\n"), "\n")
        self.assertEqual(restorer.parse("##"), "")
        self.assertEqual(restorer.parse(" child"), "\n## child")

    def test_parse_inserts_blank_line_between_adjacent_paragraph_lines(self):
        restorer = StreamingMarkdownRestorer()

        self.assertEqual(restorer.parse("paragraph 1"), "paragraph 1")
        self.assertEqual(restorer.parse("\n"), "\n")
        self.assertEqual(restorer.parse("paragraph 2"), "\nparagraph 2")
        self.assertEqual(restorer.parse("\n"), "\n")
        self.assertEqual(restorer.parse("paragraph 3"), "\nparagraph 3")
        self.assertEqual(restorer.parse(" test"), " test")

    def test_parse_buffers_single_table_line_until_separator_arrives(self):
        restorer = StreamingMarkdownRestorer()

        self.assertEqual(restorer.parse("before"), "before")
        self.assertEqual(restorer.parse("\n"), "\n")
        self.assertEqual(restorer.parse("| h1 | h2 |\n"), "")
        self.assertEqual(restorer.parse("| -- | -- |\n"), "\n| h1 | h2 |\n| -- | -- |\n")

    def test_parse_keeps_table_rows_streaming_without_extra_blank_lines(self):
        restorer = StreamingMarkdownRestorer()

        self.assertEqual(restorer.parse("| h1 | h2 |\n"), "")
        self.assertEqual(restorer.parse("| -- | -- |\n"), "| h1 | h2 |\n| -- | -- |\n")
        self.assertEqual(restorer.parse("| r1 | r2 |\n"), "")
        self.assertEqual(restorer.parse("| r3 | r4 |\n"), "| r1 | r2 |\n")
        self.assertEqual(restorer.flush(), "| r3 | r4 |\n")

    def test_parse_emits_code_fence_prefix_before_line_end(self):
        restorer = StreamingMarkdownRestorer()

        self.assertEqual(restorer.parse("text"), "text")
        self.assertEqual(restorer.parse("\n"), "\n")
        self.assertEqual(restorer.parse("```"), "\n```")
        self.assertEqual(restorer.parse("python"), "python")
        self.assertEqual(restorer.parse("\n"), "\n")
        self.assertEqual(restorer.parse("x"), "x")

    def test_parse_emits_display_math_prefix_before_line_end(self):
        restorer = StreamingMarkdownRestorer()

        self.assertEqual(restorer.parse("text"), "text")
        self.assertEqual(restorer.parse("\n"), "\n")
        self.assertEqual(restorer.parse("$"), "")
        self.assertEqual(restorer.parse("$"), "")
        self.assertEqual(restorer.parse("\n"), "\n$$\n")
        self.assertEqual(restorer.parse("x"), "x")

    def test_parse_inserts_blank_line_between_independent_code_fences(self):
        restorer = StreamingMarkdownRestorer()

        self.assertEqual(restorer.parse("```"), "```")
        self.assertEqual(restorer.parse("\ncode\n```\n"), "\ncode\n```\n")
        self.assertEqual(restorer.parse("```"), "\n```")

    def test_parse_inserts_blank_line_between_independent_display_math_blocks(self):
        restorer = StreamingMarkdownRestorer()

        self.assertEqual(restorer.parse("$$\n"), "$$\n")
        self.assertEqual(restorer.parse("x\n$$\n"), "x\n$$\n")
        self.assertEqual(restorer.parse("$$\n"), "\n$$\n")

    def test_parse_restores_block_spacing_across_streaming_chunks(self):
        source = textwrap.dedent(
            """\
            # Web RWKV Markdown render test
            ## inline
            inline `code` inline strong **strong**
            new paragraph
            ## list
            list test
            - a1 `b1` c1 **d1**
            - a2 `b2` c2 **d2**
            - a3 `b3` c3 **d3**
            ## code
            ```python
            def xxx(sdwds):
                asdfghjklasdfghjklasdfghjkl.asdwdsa()
                asdfghjklasdfghjklasdfghjkl.asdwdsa()
                
            xxx(xxasd)
            ```
            ## table
            | h1 | h2 | h3 |
            | -- | -- | -- |
            | r1c1 | r1c2 | r1c3 |
            | r2c1 | r2c2 | r2c3 |
            | h1 | h2   | h3       |
            | ---- | ------ | ---------- |
            | 1    | 1-*2*  | 1-_3_      |
            | 2    | 2-**2**  | 2-__3__      |
            | 3    | 3-***2***  | 3-___3___      |
            ## blockquote
            > q1
            > > q2
            > + l1
            > + l2 `code` 
            ## katex
            inline $f(x)=x+y$
            $$
            % \\f is defined as #1f(#2) using the macro
            \\f\\relax{x} = \\int_{-\\infty}^\\infty
                \\f\\hat\\xi\\,e^{2 \\pi i \\xi x}
                \\,d\\xi
            $$
            test
            """
        )
        expected = textwrap.dedent(
            """\
            # Web RWKV Markdown render test

            ## inline

            inline `code` inline strong **strong**

            new paragraph

            ## list

            list test

            - a1 `b1` c1 **d1**
            - a2 `b2` c2 **d2**
            - a3 `b3` c3 **d3**

            ## code

            ```python
            def xxx(sdwds):
                asdfghjklasdfghjklasdfghjkl.asdwdsa()
                asdfghjklasdfghjklasdfghjkl.asdwdsa()

            xxx(xxasd)
            ```

            ## table

            | h1 | h2 | h3 |
            | -- | -- | -- |
            | r1c1 | r1c2 | r1c3 |
            | r2c1 | r2c2 | r2c3 |

            | h1 | h2   | h3       |
            | ---- | ------ | ---------- |
            | 1    | 1-*2*  | 1-_3_      |
            | 2    | 2-**2**  | 2-__3__      |
            | 3    | 3-***2***  | 3-___3___      |

            ## blockquote

            > q1
            > > q2
            > + l1
            > + l2 `code` 

            ## katex

            inline $f(x)=x+y$

            $$
            % \\f is defined as #1f(#2) using the macro
            \\f\\relax{x} = \\int_{-\\infty}^\\infty
                \\f\\hat\\xi\\,e^{2 \\pi i \\xi x}
                \\,d\\xi
            $$

            test
            """
        )

        restorer = StreamingMarkdownRestorer()
        step = 1
        chunks = [source[index : index + step] for index in range(0, len(source), step)]
        rendered = "".join(restorer.parse(chunk) for chunk in chunks) + restorer.flush()

        self.assertEqual(rendered, expected)

    def test_flush_releases_trailing_incomplete_line(self):
        restorer = StreamingMarkdownRestorer()

        self.assertEqual(restorer.parse("##"), "")
        self.assertEqual(restorer.parse(" heading"), "## heading")
        self.assertEqual(restorer.parse("\nvalue"), "\n\nvalue")
        self.assertEqual(restorer.flush(), "")


if __name__ == "__main__":
    unittest.main()