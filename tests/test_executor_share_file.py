"""Tests for share_file in ExecutorClient.

share_file is the bridge-local write path for conversation files: the
agent created a real file on this machine, so the bridge reads the bytes
and pushes them through upload-url → PUT → confirm. These tests mock
_post and the storage PUT so no network calls are made.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentchat.errors import AgentChatError
from agentchat.executor import ExecutorClient

CONV_ID = "11111111-2222-3333-4444-555555555555"


@pytest.fixture
def executor(base_url, agent_id, api_key):
    return ExecutorClient(base_url, agent_id, api_key, "test-executor")


@pytest.fixture
def local_file(tmp_path):
    f = tmp_path / "report.xlsx"
    f.write_bytes(b"PK\x03\x04 fake xlsx bytes")
    return f


def _mock_post(upload_url="https://storage.test/signed-put"):
    """AsyncMock for _post that answers upload-url then confirm."""

    async def route(path, json=None, **kwargs):
        if path.endswith("/upload-url"):
            return {
                "uploadUrl": upload_url,
                "storageKey": f"conversations/{CONV_ID}/uuid/report.xlsx",
            }
        if path.endswith("/files/confirm"):
            return {
                "id": "msg-9999",
                "fileAttachments": [{"id": "att-1234", "filename": "report.xlsx"}],
            }
        raise AssertionError(f"unexpected _post path: {path}")

    return AsyncMock(side_effect=route)


def _mock_storage_client(status_code=200):
    client = MagicMock()
    resp = MagicMock()
    resp.status_code = status_code
    resp.text = "err body"
    client.put = AsyncMock(return_value=resp)
    return client


class TestShareFile:
    @pytest.mark.asyncio
    async def test_happy_path(self, executor, local_file):
        storage = _mock_storage_client()
        executor._api_client = storage
        with patch.object(executor, "_post", new=_mock_post()) as mock_post:
            result = await executor.share_file(
                str(local_file), conversation_id=CONV_ID, caption="the workbook"
            )

        assert result["attachment_id"] == "att-1234"
        assert result["message_id"] == "msg-9999"
        assert result["filename"] == "report.xlsx"
        assert result["size_bytes"] == local_file.stat().st_size

        # upload-url request carries filename/contentType/sizeBytes
        upload_call = mock_post.call_args_list[0]
        assert upload_call.args[0] == f"/api/conversations/{CONV_ID}/upload-url"
        body = upload_call.kwargs["json"]
        assert body["filename"] == "report.xlsx"
        assert body["sizeBytes"] == local_file.stat().st_size

        # bytes were PUT to the signed URL
        put_call = storage.put.call_args
        assert put_call.args[0] == "https://storage.test/signed-put"
        assert put_call.kwargs["content"] == local_file.read_bytes()

        # confirm carries the storage key + caption
        confirm_call = mock_post.call_args_list[1]
        assert confirm_call.args[0] == f"/api/conversations/{CONV_ID}/files/confirm"
        confirm_body = confirm_call.kwargs["json"]
        assert confirm_body["storageKey"].startswith(f"conversations/{CONV_ID}/")
        assert confirm_body["caption"] == "the workbook"

    @pytest.mark.asyncio
    async def test_filename_override(self, executor, local_file):
        executor._api_client = _mock_storage_client()
        with patch.object(executor, "_post", new=_mock_post()) as mock_post:
            result = await executor.share_file(
                str(local_file), conversation_id=CONV_ID, filename="counter_offer.xlsx"
            )
        assert result["filename"] == "counter_offer.xlsx"
        body = mock_post.call_args_list[0].kwargs["json"]
        assert body["filename"] == "counter_offer.xlsx"

    @pytest.mark.asyncio
    async def test_missing_conversation_id(self, executor, local_file):
        with pytest.raises(AgentChatError, match="conversation_id"):
            await executor.share_file(str(local_file))

    @pytest.mark.asyncio
    async def test_missing_file(self, executor, tmp_path):
        with pytest.raises(AgentChatError, match="no such file"):
            await executor.share_file(
                str(tmp_path / "nope.txt"), conversation_id=CONV_ID
            )

    @pytest.mark.asyncio
    async def test_empty_file(self, executor, tmp_path):
        f = tmp_path / "empty.txt"
        f.write_bytes(b"")
        with pytest.raises(AgentChatError, match="empty"):
            await executor.share_file(str(f), conversation_id=CONV_ID)

    @pytest.mark.asyncio
    async def test_oversize_file(self, executor, tmp_path):
        f = tmp_path / "big.bin"
        f.write_bytes(b"x")
        with patch("os.path.getsize", return_value=26 * 1024 * 1024):
            with pytest.raises(AgentChatError, match="25MB"):
                await executor.share_file(str(f), conversation_id=CONV_ID)

    @pytest.mark.asyncio
    async def test_storage_put_failure_surfaces(self, executor, local_file):
        executor._api_client = _mock_storage_client(status_code=500)
        with patch.object(executor, "_post", new=_mock_post()):
            with pytest.raises(AgentChatError, match="storage PUT failed"):
                await executor.share_file(str(local_file), conversation_id=CONV_ID)
