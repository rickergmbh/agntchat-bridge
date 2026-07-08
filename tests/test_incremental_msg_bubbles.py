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
    # H4 (audit-remediation-plan): the bridge reads the backend's
    # humanExpectsReply directive with NO local recomputation. The old
    # member-count heuristic tests died with the fallback — the backend's
    # side of the contract is locked by behavioral_directives_test
    # ("bridge decision-field contract").

    def test_directive_true(self):
        assert _human_expects_reply({"humanExpectsReply": True}) is True

    def test_directive_false(self):
        assert _human_expects_reply({"humanExpectsReply": False}) is False

    def test_missing_directive_is_protocol_error_defaulting_to_false(self):
        # Never improvise: silence over a leaked "couldn't formulate a
        # response" fallback. The bridge logs a protocol error here.
        assert _human_expects_reply({}) is False

    def test_non_bool_value_treated_as_missing(self):
        assert _human_expects_reply({"humanExpectsReply": "yes"}) is False


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

    def test_pacing_honors_server_config(self):
        # behavioralConfig.humanlikePacing is the source of truth — the bridge
        # must use the server's numbers, not the hardcoded fallbacks.
        cfg = {"humanlikePacing": {"readBaseMs": 2000, "readPerCharMs": 0, "readMaxMs": 9000}}
        assert _bubble_read_pause_s("ok", cfg) == 2.0  # 2000ms base, 0/char

    def test_pacing_falls_back_when_config_absent(self):
        # No directive → fallback defaults (0.5s read base).
        assert _bubble_read_pause_s("ok", None) >= 0.5
        assert _bubble_read_pause_s("ok", {}) >= 0.5
