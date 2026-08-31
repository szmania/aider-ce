import asyncio
import json
from unittest.mock import MagicMock, patch

import pytest

from cecli.llm import litellm
from cecli.models import Model


def test_extract_gemini_retry_delay_valid():
    model = Model("gemini/gemini-2.5-flash")

    payload = {
        "error": {
            "code": 429,
            "message": "Quota exceeded for metric ... Please retry in 15.2s.",
            "status": "RESOURCE_EXHAUSTED",
            "details": [
                {
                    "@type": "://googleapis.com",
                    "violations": [
                        {
                            "subject": "client_id:your_api_key_or_project",
                            "description": "Rate limit exceeded.",
                        }
                    ],
                },
                {"@type": "://googleapis.com", "retryDelay": "15.2s"},
            ],
        }
    }

    # Exception with response object
    err = Exception("Rate limit error")
    err.status_code = 429
    mock_resp = MagicMock()
    mock_resp.json.return_value = payload
    err.response = mock_resp

    delay = model._extract_gemini_retry_delay(err)
    assert delay == 15.2


def test_extract_gemini_retry_delay_json_in_message():
    model = Model("gemini/gemini-2.5-flash")

    payload = {
        "error": {
            "code": 429,
            "message": "Quota exceeded",
            "details": [{"retryDelay": "8.5s"}],
        }
    }

    err = Exception(f"APIError: 429 {json.dumps(payload)}")
    delay = model._extract_gemini_retry_delay(err)
    assert delay == 8.5


def test_extract_gemini_retry_delay_non_429():
    model = Model("gemini/gemini-2.5-flash")

    payload = {
        "error": {
            "code": 500,
            "message": "Internal error",
            "details": [{"retryDelay": "15.2s"}],
        }
    }

    err = Exception("Internal Error")
    err.status_code = 500
    mock_resp = MagicMock()
    mock_resp.json.return_value = payload
    err.response = mock_resp

    delay = model._extract_gemini_retry_delay(err)
    assert delay is None


def test_extract_gemini_retry_delay_missing_details():
    model = Model("gemini/gemini-2.5-flash")

    payload = {
        "error": {
            "code": 429,
            "message": "Quota exceeded",
        }
    }

    err = Exception("Quota exceeded")
    err.status_code = 429
    mock_resp = MagicMock()
    mock_resp.json.return_value = payload
    err.response = mock_resp

    delay = model._extract_gemini_retry_delay(err)
    assert delay is None


def test_retry_fallback_to_unilateral_backoff_when_no_retry_delay():
    async def run_test():
        model = Model("gemini/gemini-2.5-flash")

        rate_limit_err = litellm.RateLimitError(
            message="Rate limit exceeded",
            response=MagicMock(
                json=lambda: {"error": {"code": 429, "message": "Rate limit exceeded"}}
            ),
            model="gemini/gemini-2.5-flash",
            llm_provider="gemini",
        )

        call_count = 0
        slept_delays = []

        async def mock_acompletion(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise rate_limit_err
            return MagicMock(choices=[MagicMock(message=MagicMock(content="ok"))])

        async def mock_sleep(delay):
            slept_delays.append(delay)

        with (
            patch("cecli.llm.litellm.acompletion", side_effect=mock_acompletion),
            patch("asyncio.sleep", side_effect=mock_sleep),
        ):
            await model.send_completion(
                messages=[{"role": "user", "content": "hi"}], functions=None, stream=False
            )

        # Initial retry_delay is 0.125, multiplied by retry_backoff_factor (1.5) = 0.1875
        assert len(slept_delays) == 1
        assert pytest.approx(slept_delays[0]) == 0.125 * 1.5

    asyncio.run(run_test())
