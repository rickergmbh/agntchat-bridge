"""Tests for data model serialization."""

from agentchat.models import Conversation, Message, Participant


class TestParticipant:
    def test_from_dict(self, sample_participant_dict):
        p = Participant.from_dict(sample_participant_dict)
        assert p.id == "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
        assert p.type == "agent"
        assert p.display_name == "TestBot"
        assert p.avatar_url == "https://example.com/avatar.png"
        assert p.capabilities == ["chat", "tasks"]
        assert p.description == "A test agent"
        assert p.status == "active"

    def test_from_dict_minimal(self):
        p = Participant.from_dict({"id": "abc", "type": "human"})
        assert p.id == "abc"
        assert p.display_name == ""
        assert p.capabilities == []
        assert p.avatar_url is None

    def test_round_trip(self, sample_participant_dict):
        p = Participant.from_dict(sample_participant_dict)
        d = p.to_dict()
        assert d["id"] == sample_participant_dict["id"]
        assert d["displayName"] == sample_participant_dict["displayName"]
        assert d["avatarUrl"] == sample_participant_dict["avatarUrl"]

    def test_to_dict_camel_case(self):
        p = Participant(id="x", type="agent", display_name="Bot", avatar_url="http://a")
        d = p.to_dict()
        assert "displayName" in d
        assert "avatarUrl" in d
        assert "display_name" not in d


class TestMessage:
    def test_from_dict(self, sample_message_dict):
        m = Message.from_dict(sample_message_dict)
        assert m.id == "msg-1111"
        assert m.conversation_id == "conv-2222"
        assert m.sender_id == "sender-3333"
        assert m.content == "Hello, world!"
        assert m.content_type == "text"
        assert m.metadata == {"key": "value"}
        assert m.sender is not None
        assert m.sender.display_name == "TestBot"
        assert m.inserted_at == "2025-01-01T00:00:00Z"

    def test_from_dict_no_sender(self):
        m = Message.from_dict({
            "id": "m1",
            "conversationId": "c1",
            "senderId": "s1",
            "content": "hi",
        })
        assert m.sender is None
        assert m.content_type == "text"

    def test_round_trip(self, sample_message_dict):
        m = Message.from_dict(sample_message_dict)
        d = m.to_dict()
        assert d["conversationId"] == "conv-2222"
        assert d["senderId"] == "sender-3333"
        assert d["contentType"] == "text"
        assert "sender" in d

    def test_to_dict_omits_none_optional(self):
        m = Message(id="m1", conversation_id="c1", sender_id="s1", content="hi")
        d = m.to_dict()
        assert "parentMessageId" not in d
        assert "sender" not in d
        assert "insertedAt" not in d


class TestConversation:
    def test_from_dict(self, sample_conversation_dict):
        c = Conversation.from_dict(sample_conversation_dict)
        assert c.id == "conv-2222"
        assert c.type == "group"
        assert c.title == "Test Chat"
        assert c.created_by == "creator-4444"
        assert len(c.members) == 1

    def test_from_dict_minimal(self):
        c = Conversation.from_dict({"id": "c1", "type": "direct"})
        assert c.title is None
        assert c.members == []
        assert c.metadata == {}

    def test_round_trip(self, sample_conversation_dict):
        c = Conversation.from_dict(sample_conversation_dict)
        d = c.to_dict()
        assert d["id"] == "conv-2222"
        assert d["type"] == "group"
        assert d["title"] == "Test Chat"
        assert d["createdBy"] == "creator-4444"
