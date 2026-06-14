"""Simplified AES-256-GCM encryption for cecli session files.

Key improvements over PR #533's session_crypto.py:
  1. Raw binary format instead of base64-encoded payloads
     (eliminates base64 import, padding logic, and ascii encode/decode)
  2. decrypt_session_bytes is single-purpose (encrypted data only)
  3. Callers check is_encrypted_payload() first, then decrypt
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

MAGIC = b"CECLI_ENCRYPTED_SESSION_v1\n"
KEY_ENV = "CECLI_SESSION_KEY"
KEY_BYTES = 32


class SessionCryptoError(Exception):
    """Session encrypt/decrypt failed."""


def is_encrypted_payload(data: bytes) -> bool:
    """Check whether *data* starts with the magic header."""
    return data.startswith(MAGIC)


def resolve_key(*, key_file: str | Path | None = None) -> bytes | None:
    """Load a 32-byte key from CECLI_SESSION_KEY (urlsafe base64) or a key file.

    Returns None when no key is configured or the value is invalid.
    """
    raw = os.environ.get(KEY_ENV, "").strip()
    if raw:
        key = _decode_key_b64(raw)
        if key is not None:
            return key

    if key_file:
        path = Path(key_file).expanduser()
        if path.is_file():
            key = _decode_key_b64(path.read_text(encoding="utf-8").strip())
            if key is not None:
                return key

    return None


def _decode_key_b64(text: str) -> bytes | None:
    """Decode a urlsafe-base64 32-byte key, tolerating missing padding."""
    try:
        import base64

        # Python's b64decode accepts excess padding, so "==" always works.
        key = base64.urlsafe_b64decode(text + "==")
    except (ValueError, UnicodeEncodeError):
        return None
    if len(key) != KEY_BYTES:
        return None
    return key


def encrypt_session_dict(session_data: dict[str, Any], key: bytes) -> bytes:
    """Encrypt *session_data* and return bytes ready to write to disk.

    Format: CECLI_ENCRYPTED_SESSION_v1\n || 12-byte nonce || AES-256-GCM ciphertext
    """
    if len(key) != KEY_BYTES:
        raise SessionCryptoError(f"Session key must be {KEY_BYTES} bytes.")

    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    except ImportError as err:
        raise SessionCryptoError(
            "Session encryption requires the cryptography package" " (pip install cryptography)."
        ) from err

    plaintext = json.dumps(session_data, ensure_ascii=False).encode("utf-8")
    nonce = os.urandom(12)
    ciphertext = AESGCM(key).encrypt(nonce, plaintext, None)

    return MAGIC + nonce + ciphertext


def decrypt_session_bytes(data: bytes, key: bytes) -> dict[str, Any]:
    """Decrypt a previously encrypted session blob.

    Raises SessionCryptoError on any failure (wrong key, corrupted data,
    invalid format).  Callers MUST check *is_encrypted_payload* first.
    """
    if len(key) != KEY_BYTES:
        raise SessionCryptoError(f"Session key must be {KEY_BYTES} bytes.")

    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    except ImportError as err:
        raise SessionCryptoError(
            "Session encryption requires the cryptography package" " (pip install cryptography)."
        ) from err

    body = data[len(MAGIC) :]
    if len(body) < 13:
        raise SessionCryptoError("Encrypted session payload is too short.")

    nonce, ciphertext = body[:12], body[12:]

    try:
        plaintext = AESGCM(key).decrypt(nonce, ciphertext, None)
    except Exception as err:
        raise SessionCryptoError(
            "Could not decrypt session (wrong key or corrupted file)."
        ) from err

    try:
        parsed = json.loads(plaintext.decode("utf-8"))
    except json.JSONDecodeError as err:
        raise SessionCryptoError("Decrypted session is not valid JSON.") from err

    if not isinstance(parsed, dict):
        raise SessionCryptoError("Invalid session format.")

    return parsed
