"""Shared helpers for translating the bridge's internal `attachment`
content block into backend-specific shapes.

The bridge's `messages_to_chat_history` emits a uniform block:

    {"type": "attachment",
     "filename": str,
     "content_type": str,   # MIME
     "attachment_id": str | None,
     "url": str | None,     # presigned download URL
     "label": str}          # human-readable "X shared a file: foo.pdf"

Each backend adapter decides WHAT to do with that block (Anthropic
emits native image/document blocks, Claude CLI downloads to a temp
path so its Read tool can open it, etc.). The pieces that are
identical across all backends live here:

  * `is_image_attachment`  — content-type predicate
  * `is_pdf_attachment`    — content-type predicate
  * `attachment_label`     — always includes attachment_id so the
                             model can call `read_attachment` even
                             when it also has the file inline
  * `fallback_text`        — text to emit when no native handling is
                             possible (or available) for this MIME
"""

from __future__ import annotations

from typing import Any


# Non-image files at or above this size skip the raw download + native Read
# path: the model is pointed at `read_attachment` (server-extracted text,
# capped at 512 KB) plus the brief, so a multi-MB Read doesn't echo the whole
# file back into context. Matches the backend's read_attachment inline cap —
# below it, native Read is at most as large; above it, the capped path wins.
# Images always render natively (vision needs the bytes).
LARGE_ATTACHMENT_BYTES = 512 * 1024


def is_image_attachment(block: dict[str, Any]) -> bool:
    return (block.get("content_type") or "").startswith("image/")


def is_large_attachment(block: dict[str, Any]) -> bool:
    """True when the file is big enough to prefer the capped read path."""
    size = block.get("size_bytes")
    return isinstance(size, int) and size >= LARGE_ATTACHMENT_BYTES


def should_use_capped_path(block: dict[str, Any]) -> bool:
    """Route large non-image files through read_attachment, not raw Read.

    Images are excluded — a model needs the actual bytes to see them, and
    they don't produce the giant text echo a document Read does.
    """
    return is_large_attachment(block) and not is_image_attachment(block)


def capped_pointer_text(block: dict[str, Any]) -> str:
    """Pointer for a large file: brief + read_attachment hint + size.

    Deliberately inlines NO raw bytes. The model gets the server brief (if
    ready) and is told to call `read_attachment` for the full (capped)
    extracted text, or use the download URL for raw bytes.
    """
    parts = [attachment_label(block)]

    summary = block.get("summary")
    if summary and block.get("summary_status") == "completed":
        parts.append(f"Summary: {summary}")

    size = block.get("size_bytes")
    if isinstance(size, int) and size > 0:
        parts.append(f"({size} bytes — large file)")

    if block.get("attachment_id"):
        parts.append(
            "Use read_attachment for the full server-extracted text (capped); "
            "the download URL has the raw bytes if you truly need them."
        )
    return " ".join(parts)


def is_pdf_attachment(block: dict[str, Any]) -> bool:
    return (block.get("content_type") or "") == "application/pdf"


def attachment_label(block: dict[str, Any]) -> str:
    """Human-readable label including the attachment_id when known.

    Surfacing the id on every path — not just the fallback — is
    deliberate: a model that sees a PDF rendered inline may still
    want to call `read_attachment` for the columnar `pdftotext` body
    or for pages beyond Anthropic's PDF rendering caps.
    """
    base = block.get("label") or f"shared a file: {block.get('filename') or 'file'}"
    attachment_id = block.get("attachment_id")
    if attachment_id:
        return f"{base} [attachment_id={attachment_id}]"
    return base


def fallback_text(block: dict[str, Any]) -> str:
    """Plain-text reference + read_attachment hint for non-native paths."""
    label = attachment_label(block)
    if block.get("attachment_id"):
        return f"{label} (use read_attachment to read it)"
    return label
