"""SessionManager on-disk persistence and optional encryption."""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from cecli.helpers import crypto as session_crypto
from cecli.io import InputOutput
from cecli.sessions import SessionManager


def _prepare_workspace(coder, tmp_path) -> Path:
    root = Path(tmp_path)
    coder.abs_root_path.side_effect = lambda x: str(root / x)
    (root / ".cecli" / "sessions").mkdir(parents=True, exist_ok=True)
    (root / "file1.py").write_text("", encoding="utf-8")
    return root


@pytest.fixture
def mock_coder(monkeypatch):
    main_model = MagicMock()
    main_model.name = "test_model"
    main_model.weak_model.name = "weak"
    main_model.editor_model.name = "editor"
    main_model.agent_model.name = "agent"
    main_model.editor_edit_format = "editor-diff"
    main_model.retries = 0
    main_model.debug = False

    conv_manager = MagicMock()
    conv_manager.get_messages_dict.return_value = []
    files_manager = MagicMock()
    monkeypatch.setattr(
        "cecli.sessions.ConversationService.get_manager",
        lambda _coder: conv_manager,
    )
    monkeypatch.setattr(
        "cecli.sessions.ConversationService.get_files",
        lambda _coder: files_manager,
    )
    monkeypatch.setattr(
        "cecli.sessions.models.Model",
        lambda *args, **kwargs: main_model,
    )

    coder = MagicMock()
    coder.abs_fnames = set()
    coder.abs_read_only_fnames = set()
    coder.abs_read_only_stubs_fnames = set()
    coder.auto_commits = True
    coder.auto_lint = True
    coder.auto_test = False
    coder.total_tokens_sent = 0
    coder.total_tokens_received = 0
    coder.total_cached_tokens = 0
    coder.total_cost = 0.0
    coder.edit_format = "diff"
    coder.format_chat_chunks = MagicMock()
    coder.get_rel_fname.side_effect = lambda x: os.path.basename(x)
    coder.local_agent_folder.side_effect = lambda x: f".cecli/{x}"
    coder.io = MagicMock(spec=InputOutput)
    coder.agent_config = {}
    coder.mcp_manager = None
    coder.skills_manager = None
    coder.main_model = main_model
    coder.args = SimpleNamespace(
        model="test_model",
        weak_model="weak",
        editor_model="editor",
        agent_model="agent",
        editor_edit_format="editor-diff",
        verbose=False,
        session_encrypt=False,
        session_key_file=None,
    )
    return coder


@pytest.fixture
def session_manager(mock_coder):
    return SessionManager(mock_coder, mock_coder.io)


@pytest.fixture
def encrypt_coder(mock_coder, session_key_env):
    mock_coder.args = SimpleNamespace(
        model="test_model",
        weak_model="weak",
        editor_model="editor",
        agent_model="agent",
        editor_edit_format="editor-diff",
        verbose=False,
        session_encrypt=True,
        session_key_file=None,
    )
    return mock_coder


def test_save_plaintext_json(session_manager, mock_coder, tmp_path):
    root = _prepare_workspace(mock_coder, tmp_path)
    assert session_manager.save_session("plain", output=False)
    path = root / ".cecli" / "sessions" / "plain.json"
    raw = path.read_bytes()
    assert raw.startswith(b"{")
    data = json.loads(raw.decode("utf-8"))
    assert data["session_name"] == "plain"
    assert data["version"] == 1


def test_save_encrypted_blob(encrypt_coder, session_key32, tmp_path):
    manager = SessionManager(encrypt_coder, encrypt_coder.io)
    root = _prepare_workspace(encrypt_coder, tmp_path)
    assert manager.save_session("secret", output=False)
    path = root / ".cecli" / "sessions" / "secret.json"
    raw = path.read_bytes()
    assert session_crypto.is_encrypted_payload(raw)
    assert session_crypto.decrypt_session_bytes(raw, session_key32)["session_name"] == "secret"


def test_save_encrypt_without_key_fails(mock_coder, monkeypatch, tmp_path):
    monkeypatch.delenv(session_crypto.KEY_ENV, raising=False)
    _prepare_workspace(mock_coder, tmp_path)
    mock_coder.args = SimpleNamespace(
        model="test_model",
        weak_model="weak",
        editor_model="editor",
        agent_model="agent",
        editor_edit_format="editor-diff",
        verbose=False,
        session_encrypt=True,
        session_key_file=None,
    )
    assert SessionManager(mock_coder, mock_coder.io).save_session("nope", output=False) is False


def test_list_encrypted_with_key(encrypt_coder, tmp_path):
    manager = SessionManager(encrypt_coder, encrypt_coder.io)
    _prepare_workspace(encrypt_coder, tmp_path)
    manager.save_session("listed", output=False)
    rows = manager.list_sessions()
    assert len(rows) == 1
    assert rows[0]["name"] == "listed"
    assert rows[0].get("encrypted") is True
    assert rows[0]["model"] == "test_model"


def test_list_encrypted_placeholder_without_key(encrypt_coder, monkeypatch, tmp_path):
    manager = SessionManager(encrypt_coder, encrypt_coder.io)
    _prepare_workspace(encrypt_coder, tmp_path)
    manager.save_session("locked", output=False)
    monkeypatch.delenv(session_crypto.KEY_ENV, raising=False)
    encrypt_coder.args = SimpleNamespace(
        model="test_model",
        weak_model="weak",
        editor_model="editor",
        agent_model="agent",
        editor_edit_format="editor-diff",
        verbose=False,
        session_encrypt=False,
        session_key_file=None,
    )
    rows = manager.list_sessions()
    assert rows[0]["encrypted"] is True
    assert rows[0]["model"] == "encrypted"


def test_read_legacy_plaintext_when_encrypt_enabled(encrypt_coder, tmp_path):
    manager = SessionManager(encrypt_coder, encrypt_coder.io)
    root = _prepare_workspace(encrypt_coder, tmp_path)
    legacy = root / ".cecli" / "sessions" / "legacy.json"
    legacy.write_text(
        json.dumps({"version": 1, "session_name": "legacy", "model": "test_model"}),
        encoding="utf-8",
    )
    data = manager._read_session_file(legacy)
    assert data is not None
    assert data["session_name"] == "legacy"


@pytest.mark.asyncio
async def test_load_encrypted_without_switch(encrypt_coder, session_key32, tmp_path):
    manager = SessionManager(encrypt_coder, encrypt_coder.io)
    root = _prepare_workspace(encrypt_coder, tmp_path)
    encrypt_coder.edit_format = "ask"
    assert manager.save_session("enc", output=False)
    encrypt_coder.edit_format = "diff"
    path = root / ".cecli" / "sessions" / "enc.json"
    assert await manager.load_session(str(path), switch=False) is True
    loaded = session_crypto.decrypt_session_bytes(path.read_bytes(), session_key32)
    assert loaded["edit_format"] == "ask"


@pytest.mark.asyncio
async def test_load_encrypted_using_env_key_only(encrypt_coder, session_key_env, tmp_path):
    manager = SessionManager(encrypt_coder, encrypt_coder.io)
    root = _prepare_workspace(encrypt_coder, tmp_path)
    encrypt_coder.edit_format = "architect"
    manager.save_session("env", output=False)
    encrypt_coder.args = SimpleNamespace(
        model="test_model",
        weak_model="weak",
        editor_model="editor",
        agent_model="agent",
        editor_edit_format="editor-diff",
        verbose=False,
        session_encrypt=False,
        session_key_file=None,
    )
    path = root / ".cecli" / "sessions" / "env.json"
    assert await manager.load_session(str(path), switch=False) is True
    loaded = session_crypto.decrypt_session_bytes(path.read_bytes(), session_key_env)
    assert loaded["edit_format"] == "architect"
