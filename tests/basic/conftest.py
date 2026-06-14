"""Shared fixtures for cecli basic tests."""

import base64
import os

import pytest

from cecli.helpers import crypto as session_crypto


@pytest.fixture
def session_key32():
    return os.urandom(session_crypto.KEY_BYTES)


@pytest.fixture
def session_key_b64(session_key32):
    return base64.urlsafe_b64encode(session_key32).decode().rstrip("=")


@pytest.fixture
def session_key_env(monkeypatch, session_key32, session_key_b64):
    monkeypatch.setenv(session_crypto.KEY_ENV, session_key_b64)
    return session_key32
