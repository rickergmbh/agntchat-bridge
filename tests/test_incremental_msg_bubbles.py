"""Human-like bubble splitting + pacing + peer-mention routing.

The bridge takes one completed reply and posts it as several human-paced chat
bubbles. A bubble that addresses a peer agent goes through the full send path
so the peer is added + woken; the rest are turn-neutral continuations.
"""

from agent_bridge import (
    _split_reply_into_bubbles,
    _reply_mentions_agent,
    _bubble_read_pause_s,
    _bubble_write_pause_s,
    _human_expects_reply,
)


class TestHumanExpectsReply:
    one_human_one_agent = [
        {"type": "human", "displayName": "James"},
        {"type": "agent", "displayName": "Tim"},
    ]
    two_humans = [
        {"type": "human", "displayName": "James"},
        {"type": "human", "displayName": "Sam"},
        {"type": "agent", "displayName": "Tim"},
    ]

    def test_single_human_conversation_human_sender(self):
        # Onboarding: 1 human + 1 agent, human sent → agent must reply.
        assert _human_expects_reply({}, self.one_human_one_agent, True) is True

    def test_explicit_address_always_expects(self):
        assert _human_expects_reply({"agentAddressed": True}, self.two_humans, True) is True

    def test_multi_human_unaddressed_does_not_expect(self):
        # 2 humans, not addressed → silence is legitimate.
        assert _human_expects_reply({}, self.two_humans, True) is False

    def test_agent_sender_does_not_expect(self):
        # An agent (not a human) sent the message → no forced reply.
        assert _human_expects_reply({}, self.one_human_one_agent, False) is False


class TestSplitReplyIntoBubbles:
    def test_no_tags_is_single_bubble(self):
        assert _split_reply_into_bubbles("just a normal reply") == ["just a normal reply"]

    def test_blank_is_empty(self):
        assert _split_reply_into_bubbles("   ") == []
        assert _split_reply_into_bubbles("") == []

    def test_multiple_msg_bubbles_in_order(self):
        out = _split_reply_into_bubbles(
            "<msg>Hey</msg><msg>so about that</msg><msg>here's the plan</msg>"
        )
        assert out == ["Hey", "so about that", "here's the plan"]

    def test_prose_outside_tags_kept_in_order(self):
        out = _split_reply_into_bubbles("Intro line <msg>bubble one</msg> tail line")
        assert out == ["Intro line", "bubble one", "tail line"]

    def test_blank_bubbles_skipped(self):
        assert _split_reply_into_bubbles("<msg></msg><msg>real</msg>") == ["real"]


class TestReplyMentionsAgent:
    members = [
        {"type": "agent", "displayName": "Trtiw"},
        {"type": "agent", "displayName": "Pip"},
        {"type": "human", "displayName": "James"},
    ]

    def test_at_mention(self):
        assert _reply_mentions_agent("come help @Pip", self.members, "Trtiw") is True

    def test_bare_name(self):
        assert _reply_mentions_agent("Pip, take it from here?", self.members, "Trtiw") is True

    def test_human_mention_does_not_count(self):
        assert _reply_mentions_agent("thanks @James", self.members, "Trtiw") is False

    def test_self_name_excluded(self):
        # "Pip here" from Pip must not count as addressing a peer.
        assert _reply_mentions_agent("Pip here!", self.members, "Pip") is False

    def test_plain_text(self):
        assert _reply_mentions_agent("here is the plan", self.members, "Trtiw") is False


class TestBubblePause:
    def test_read_pause_grows_with_landed_length_and_caps(self):
        short = _bubble_read_pause_s("ok")
        long = _bubble_read_pause_s("x" * 500)
        assert short < long
        assert long <= 4.0
        assert short >= 0.5

    def test_write_pause_grows_with_next_length_and_caps(self):
        short = _bubble_write_pause_s("ok")
        long = _bubble_write_pause_s("x" * 500)
        assert short < long
        assert long <= 5.0
        assert short >= 0.8

    def test_write_beat_longer_than_read_beat_for_same_text(self):
        # Composing reads as slower than skimming: the "writing" beat for a
        # bubble should exceed the "reading" beat for the same-length text.
        text = "x" * 120
        assert _bubble_write_pause_s(text) > _bubble_read_pause_s(text)
