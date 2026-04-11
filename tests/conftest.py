"""Shared fixtures for AgentChat SDK tests."""

import pytest


@pytest.fixture
def base_url():
    return "https://agentchat.test"


@pytest.fixture
def agent_id():
    return "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"


@pytest.fixture
def api_key():
    return "ak_test-api-key-for-unit-tests"


@pytest.fixture
def sample_token():
    return "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.test.signature"


@pytest.fixture
def sample_participant_dict():
    return {
        "id": "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
        "type": "agent",
        "displayName": "TestBot",
        "avatarUrl": "https://example.com/avatar.png",
        "capabilities": ["chat", "tasks"],
        "description": "A test agent",
        "status": "active",
    }


@pytest.fixture
def sample_message_dict(sample_participant_dict):
    return {
        "id": "msg-1111",
        "conversationId": "conv-2222",
        "senderId": "sender-3333",
        "content": "Hello, world!",
        "contentType": "text",
        "metadata": {"key": "value"},
        "sender": sample_participant_dict,
        "parentMessageId": None,
        "insertedAt": "2025-01-01T00:00:00Z",
    }


@pytest.fixture
def sample_conversation_dict():
    return {
        "id": "conv-2222",
        "type": "group",
        "title": "Test Chat",
        "createdBy": "creator-4444",
        "metadata": {},
        "members": [
            {"participantId": "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee", "role": "admin"}
        ],
        "insertedAt": "2025-01-01T00:00:00Z",
    }
