"""E2E regression tests for normal (non-interrupt) TUI operation.

These tests require a running cecli TUI instance and Playwright.
They are designed to be run manually or in a CI environment with
appropriate setup.

Test Cases Covered:
  - TC-INTERRUPT-005: Normal (non-interrupt) regression
  - TC-INTERRUPT-010: Non-TUI (headless) regression
"""

import os
import subprocess
import time

import pytest

# Skip E2E tests by default unless --run-e2e flag is passed
pytestmark = pytest.mark.skipif(
    not os.environ.get("RUN_E2E_TESTS"),
    reason="E2E tests require running cecli TUI; set RUN_E2E_TESTS=1 to enable",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _launch_cecli_tui() -> subprocess.Popen:
    """Launch cecli in TUI mode with a mock LLM provider."""
    env = os.environ.copy()
    env.setdefault("CE CLI_MODEL", "mock/slow")

    proc = subprocess.Popen(
        ["python", "-m", "cecli", "--gui"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        text=True,
    )
    time.sleep(2)
    return proc


def _send_message(proc: subprocess.Popen, message: str) -> None:
    """Send a message to cecli via stdin."""
    if proc.stdin:
        proc.stdin.write(message + "\n")
        proc.stdin.flush()


def _wait_for_responsive(proc: subprocess.Popen, timeout: float = 5.0) -> bool:
    """Wait for the TUI to become responsive."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        poll = proc.poll()
        if poll is not None:
            return False
        time.sleep(0.5)
    return proc.poll() is None


def _launch_cecli_headless() -> subprocess.Popen:
    """Launch cecli in non-TUI (headless) mode."""
    env = os.environ.copy()
    env.setdefault("CE CLI_MODEL", "mock/slow")

    proc = subprocess.Popen(
        ["python", "-m", "cecli", "--message", "test"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        text=True,
    )
    time.sleep(1)
    return proc


# ---------------------------------------------------------------------------
# Test Cases
# ---------------------------------------------------------------------------


class TestTUIRegressionE2E:
    """End-to-end regression tests for the cecli TUI."""

    def test_normal_operation_regression(self):
        """TC-INTERRUPT-005: Normal (non-interrupt) regression.

        Start cecli, send multiple messages without interrupts.
        Expected: All messages process normally, no change in behavior.
        """
        proc = _launch_cecli_tui()
        try:
            # Send message 1
            _send_message(proc, "Write a hello world function in Python")
            time.sleep(2)
            responsive = _wait_for_responsive(proc, timeout=5.0)
            assert responsive, "TUI hung after message 1"

            # Send message 2
            _send_message(proc, "What is 2+2?")
            time.sleep(2)
            responsive = _wait_for_responsive(proc, timeout=5.0)
            assert responsive, "TUI hung after message 2"

            # Send message 3
            _send_message(proc, "What is the capital of France?")
            time.sleep(2)
            responsive = _wait_for_responsive(proc, timeout=5.0)
            assert responsive, "TUI hung after message 3"

        finally:
            proc.terminate()
            proc.wait(timeout=5)

    def test_non_tui_headless_regression(self):
        """TC-INTERRUPT-010: Non-TUI (headless) regression.

        Start cecli in non-TUI mode, send a message, verify it completes.
        Expected: Non-TUI operation is unchanged; no regression.
        """
        proc = _launch_cecli_headless()
        try:
            # Wait for the process to complete
            try:
                stdout, stderr = proc.communicate(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                stdout, stderr = proc.communicate()
                pytest.fail("Non-TUI cecli process timed out")

            # Verify process exited cleanly
            assert proc.returncode == 0, (
                f"Non-TUI cecli exited with code {proc.returncode}:\n" f"stderr: {stderr}"
            )

        finally:
            if proc.poll() is None:
                proc.terminate()
                proc.wait(timeout=5)
