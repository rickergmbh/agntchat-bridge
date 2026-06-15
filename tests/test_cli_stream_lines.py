"""`iter_event_lines` must survive over-long CLI stream lines.

The Claude/Codex CLIs emit newline-delimited JSON. When the CLI's `Read`
tool echoes a large uploaded document back as a single `tool_result` line,
`StreamReader.readline()` raised `LimitOverrunError` ("Separator is not
found, and chunk exceed the limit") and killed the whole turn (prod task
b4706955). `iter_event_lines` drains-and-drops such lines instead, while
yielding every line at or under the cap intact — and the bridge only ever
consumes the small token-delta / result lines anyway.
"""

import asyncio

import pytest

from agentchat.backends._cli_utils import iter_event_lines


def _reader(data: bytes) -> asyncio.StreamReader:
    reader = asyncio.StreamReader()
    reader.feed_data(data)
    reader.feed_eof()
    return reader


async def _collect(reader: asyncio.StreamReader, *, max_line: int) -> list[bytes]:
    return [
        line
        async for line in iter_event_lines(reader, timeout=5, max_line=max_line)
    ]


@pytest.mark.asyncio
async def test_yields_normal_lines_intact():
    reader = _reader(b'{"a":1}\n{"b":2}\n{"c":3}\n')
    lines = await _collect(reader, max_line=1024)
    assert lines == [b'{"a":1}', b'{"b":2}', b'{"c":3}']


@pytest.mark.asyncio
async def test_flushes_trailing_line_without_newline():
    reader = _reader(b'{"a":1}\n{"b":2}')
    lines = await _collect(reader, max_line=1024)
    assert lines == [b'{"a":1}', b'{"b":2}']


@pytest.mark.asyncio
async def test_oversized_line_is_skipped_not_raised():
    """The exact failure mode: a multi-cap line between two normal ones.

    Without the fix this is where `readline()` blew up; here the giant line
    is dropped and the surrounding events still arrive.
    """
    giant = b"x" * (4 * 1024)  # well over the 1KB cap
    payload = b'{"type":"result"}\n' + giant + b"\n" + b'{"type":"done"}\n'
    reader = _reader(payload)
    lines = await _collect(reader, max_line=1024)
    assert lines == [b'{"type":"result"}', b'{"type":"done"}']


@pytest.mark.asyncio
async def test_oversized_line_spanning_many_reads_is_skipped():
    """A line larger than a single `_READ_CHUNK` is drained across reads.

    The real tool_result echo is many MB — far bigger than one read — so the
    skip path must hold across chunk boundaries and not leak memory or stall.
    """
    giant = b"y" * (3 * 1024 * 1024)  # > _READ_CHUNK (256KB), > max_line
    payload = b'{"first":1}\n' + giant + b"\n" + b'{"last":1}\n'
    reader = _reader(payload)
    lines = await _collect(reader, max_line=64 * 1024)
    assert lines == [b'{"first":1}', b'{"last":1}']


@pytest.mark.asyncio
async def test_line_exactly_at_cap_is_kept():
    body = b"z" * 1024
    reader = _reader(body + b"\n")
    lines = await _collect(reader, max_line=1024)
    assert lines == [body]
