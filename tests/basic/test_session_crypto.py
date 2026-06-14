"""Unit tests for cecli.session_crypto."""

import base64
import json
import os

import pytest

from cecli.helpers import crypto as session_crypto


def test_roundtrip_encrypted(session_key32):
    data = {"version": 1, "session_name": "t", "model": "gpt-4"}
    blob = session_crypto.encrypt_session_dict(data, session_key32)
    assert session_crypto.is_encrypted_payload(blob)
    assert session_crypto.decrypt_session_bytes(blob, session_key32) == data


def test_plaintext_json_rejected_by_decrypt(session_key32):
    """decrypt_session_bytes is single-purpose; callers check is_encrypted_payload first."""
    raw = json.dumps({"version": 1}).encode("utf-8")
    assert not session_crypto.is_encrypted_payload(raw)
    with pytest.raises(session_crypto.SessionCryptoError):
        session_crypto.decrypt_session_bytes(raw, session_key32)


def test_wrong_key_fails(session_key32):
    blob = session_crypto.encrypt_session_dict({"version": 1}, session_key32)
    with pytest.raises(session_crypto.SessionCryptoError):
        session_crypto.decrypt_session_bytes(blob, os.urandom(32))


def test_invalid_key_length_rejected():
    with pytest.raises(session_crypto.SessionCryptoError):
        session_crypto.encrypt_session_dict({"version": 1}, b"short")


def test_resolve_key_from_env(session_key_env, session_key32):
    assert session_crypto.resolve_key() == session_key32


def test_resolve_key_from_file(tmp_path, session_key32):
    path = tmp_path / "key.txt"
    path.write_text(base64.urlsafe_b64encode(session_key32).decode(), encoding="utf-8")
    assert session_crypto.resolve_key(key_file=path) == session_key32


def test_resolve_key_missing_returns_none(monkeypatch):
    monkeypatch.delenv(session_crypto.KEY_ENV, raising=False)
    assert session_crypto.resolve_key() is None


def test_resolve_key_rejects_bad_env(monkeypatch):
    monkeypatch.setenv(session_crypto.KEY_ENV, "not-valid-key-material")
    assert session_crypto.resolve_key() is None


def test_magic_prefix_constant():
    assert session_crypto.MAGIC.startswith(b"CECLI_ENCRYPTED_SESSION")


def test_corrupt_ciphertext_raises(session_key32):
    blob = session_crypto.MAGIC + b"not-valid-base64!!!\n"
    with pytest.raises(session_crypto.SessionCryptoError):
        session_crypto.decrypt_session_bytes(blob, session_key32)


def test_empty_encrypted_body_raises(session_key32):
    blob = session_crypto.MAGIC + b"\n"
    with pytest.raises(session_crypto.SessionCryptoError):
        session_crypto.decrypt_session_bytes(blob, session_key32)


def test_encrypted_file_roundtrip_on_disk(tmp_path, session_key32):
    path = tmp_path / "sess.json"
    payload = {
        "version": 1,
        "session_name": "disk",
        "chat_history": {"done_messages": [], "cur_messages": []},
    }
    path.write_bytes(session_crypto.encrypt_session_dict(payload, session_key32))
    raw = path.read_bytes()
    assert session_crypto.is_encrypted_payload(raw)
    assert session_crypto.decrypt_session_bytes(raw, session_key32) == payload


def test_unicode_roundtrip(session_key32):
    payload = {"version": 1, "session_name": "t", "todo_list": "— fix café naïve"}
    blob = session_crypto.encrypt_session_dict(payload, session_key32)
    assert session_crypto.decrypt_session_bytes(blob, session_key32) == payload


def test_cryptography_import_error(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "cryptography.hazmat.primitives.ciphers.aead":
            raise ImportError("blocked for test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(session_crypto.SessionCryptoError, match="cryptography"):
        session_crypto.encrypt_session_dict({"version": 1}, os.urandom(32))
