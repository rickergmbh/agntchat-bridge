"""Tests for the uniform attachment block contract: every backend's
translator must surface the same information (filename, attachment_id,
read_attachment hint) so a model can reach the file regardless of
whether the adapter handles it natively or falls back to text."""

import pytest

from agentchat.backends import ChatMessage
from agentchat.backends import _attachment as att
from agentchat.backends.anthropic import _translate_attachments
from agentchat.backends.openai import _translate_content
from agentchat.backends.openclaw import _flatten_to_text


def _block(**overrides):
    base = {
        "type": "attachment",
        "filename": "Infos.pdf",
        "content_type": "application/pdf",
        "attachment_id": "abc-123",
        "url": "https://signed.example/Infos.pdf",
        "label": "[Human: Tom] shared a file: Infos.pdf",
    }
    base.update(overrides)
    return base


# --- shared helper ----------------------------------------------------------


class TestSharedAttachmentHelper:
    def test_image_predicate(self):
        assert att.is_image_attachment(_block(content_type="image/png"))
        assert not att.is_image_attachment(_block(content_type="application/pdf"))
        assert not att.is_image_attachment({"content_type": None})

    def test_pdf_predicate(self):
        assert att.is_pdf_attachment(_block())
        assert not att.is_pdf_attachment(_block(content_type="text/plain"))

    def test_label_always_includes_attachment_id_when_present(self):
        label = att.attachment_label(_block())
        assert "abc-123" in label
        assert "Infos.pdf" in label

    def test_label_skips_id_when_absent(self):
        label = att.attachment_label(_block(attachment_id=None))
        assert "abc-123" not in label
        assert "[attachment_id" not in label

    def test_fallback_text_includes_read_hint(self):
        text = att.fallback_text(_block())
        assert "read_attachment" in text
        assert "abc-123" in text


# --- Anthropic --------------------------------------------------------------


class TestAnthropicTranslation:
    def test_pdf_with_url_emits_document_block_and_text_label_with_id(self):
        msgs = _translate_attachments([
            ChatMessage(role="user", content=[_block()])
        ])
        blocks = msgs[0].content
        assert blocks[0] == {
            "type": "document",
            "source": {"type": "url", "url": "https://signed.example/Infos.pdf"},
        }
        assert blocks[1]["type"] == "text"
        # attachment_id present even when document block handles the file natively.
        assert "abc-123" in blocks[1]["text"]

    def test_image_with_url_emits_image_block(self):
        msgs = _translate_attachments([
            ChatMessage(role="user", content=[_block(content_type="image/png",
                                                     filename="photo.png")])
        ])
        blocks = msgs[0].content
        assert blocks[0]["type"] == "image"
        assert blocks[0]["source"]["url"] == "https://signed.example/Infos.pdf"

    def test_missing_url_falls_through_to_text_with_hint(self):
        msgs = _translate_attachments([
            ChatMessage(role="user", content=[_block(url=None)])
        ])
        blocks = msgs[0].content
        assert len(blocks) == 1
        assert blocks[0]["type"] == "text"
        assert "read_attachment" in blocks[0]["text"]
        assert "abc-123" in blocks[0]["text"]

    def test_unsupported_mime_uses_text_fallback(self):
        msgs = _translate_attachments([
            ChatMessage(role="user", content=[_block(content_type="application/zip",
                                                     filename="archive.zip")])
        ])
        blocks = msgs[0].content
        assert blocks[0]["type"] == "text"
        assert "read_attachment" in blocks[0]["text"]

    def test_pass_through_for_non_attachment_content(self):
        msgs = _translate_attachments([
            ChatMessage(role="user", content="plain string"),
            ChatMessage(role="assistant", content=[{"type": "text", "text": "hi"}]),
        ])
        assert msgs[0].content == "plain string"
        assert msgs[1].content == [{"type": "text", "text": "hi"}]


# --- OpenAI -----------------------------------------------------------------


class TestOpenAITranslation:
    def test_image_attachment_to_image_url_block(self):
        out = _translate_content([_block(content_type="image/jpeg", filename="x.jpg")])
        assert {"type": "image_url",
                "image_url": {"url": "https://signed.example/Infos.pdf"}} in out

    def test_non_image_attachment_collapses_to_text_string(self):
        out = _translate_content([_block()])
        # OpenAI collapse-when-all-text: PDF fallback is text → returns string.
        assert isinstance(out, str)
        assert "read_attachment" in out
        assert "abc-123" in out


# --- OpenClaw ---------------------------------------------------------------


class TestOpenClawFlatten:
    def test_attachment_becomes_text_with_read_hint(self):
        out = _flatten_to_text([_block()])
        assert "read_attachment" in out
        assert "abc-123" in out

    def test_plain_string_passes_through(self):
        assert _flatten_to_text("hi") == "hi"
