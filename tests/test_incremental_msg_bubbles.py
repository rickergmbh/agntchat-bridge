"""Incremental <msg> bubble emission for human-like messaging.

The bridge posts each <msg>…</msg> bubble the instant its closer arrives during
streaming, instead of generating the whole reply then splitting/staggering it.
These cover the parsing primitives that drive that.
"""

from agent_bridge import IncrementalMsgEmitter, _extract_msg_bubbles


class TestExtractMsgBubbles:
    def test_no_tags_returns_whole_text_as_remainder(self):
        bubbles, remainder = _extract_msg_bubbles("just a normal reply")
        assert bubbles == []
        assert remainder == "just a normal reply"

    def test_multiple_bubbles(self):
        bubbles, remainder = _extract_msg_bubbles(
            "<msg>Hey</msg><msg>so about that</msg><msg>here's the plan</msg>"
        )
        assert bubbles == ["Hey", "so about that", "here's the plan"]
        assert remainder == ""

    def test_trailing_remainder_after_last_close(self):
        bubbles, remainder = _extract_msg_bubbles("<msg>first</msg> and a tail line")
        assert bubbles == ["first"]
        assert remainder == "and a tail line"

    def test_final_unclosed_msg_becomes_remainder(self):
        # Model dropped the closing </msg> — don't lose the text.
        bubbles, remainder = _extract_msg_bubbles("<msg>done</msg><msg>almost the")
        assert bubbles == ["done"]
        assert remainder == "almost the"

    def test_blank_bubbles_skipped(self):
        bubbles, _ = _extract_msg_bubbles("<msg></msg><msg>real</msg>")
        assert bubbles == ["real"]


class TestIncrementalMsgEmitter:
    def test_emits_each_bubble_once_across_growing_buffer(self):
        em = IncrementalMsgEmitter()
        # Cumulative stream: closer for bubble 1 not yet arrived.
        assert em.take_closed("<msg>Hey</m") == []
        assert em.emitted_count == 0
        # Bubble 1 closes.
        assert em.take_closed("<msg>Hey</msg><msg>so ab") == ["Hey"]
        assert em.emitted_count == 1
        # Bubble 2 closes; bubble 1 not re-emitted.
        assert em.take_closed("<msg>Hey</msg><msg>so about X</msg>") == ["so about X"]
        assert em.emitted_count == 2
        # No new closes.
        assert em.take_closed("<msg>Hey</msg><msg>so about X</msg> tail") == []

    def test_no_msg_tags_emits_nothing(self):
        em = IncrementalMsgEmitter()
        assert em.take_closed("plain streaming text") == []
        assert em.emitted_count == 0
