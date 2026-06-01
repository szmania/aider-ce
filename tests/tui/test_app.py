from unittest.mock import MagicMock, patch

import pytest
from textual import events

# Assuming TUI is in cecli.tui.app
from cecli.tui.app import TUI


@pytest.fixture
def tui_instance(monkeypatch):
    """A pytest fixture to create a mocked TUI instance."""
    monkeypatch.setattr("cecli.tui.app.TUI.__init__", lambda *args, **kwargs: None)
    tui = TUI(coder_worker=None, output_queue=None, input_queue=None, args=None)
    tui._mouse_hold_timer = None
    tui._currently_generating = False
    return tui


def test_on_mouse_move_windows(tui_instance):
    """
    Test that on_mouse_move stops the event on Windows.
    """
    with patch("platform.system", return_value="Windows"):
        mock_event = MagicMock(spec=events.MouseMove)
        tui_instance.on_mouse_move(mock_event)
        mock_event.stop.assert_called_once()


def test_on_mouse_move_linux(tui_instance):
    """
    Test that on_mouse_move does not stop the event on Linux.
    """
    with patch("cecli.tui.app.IS_WINDOWS", False):
        mock_event = MagicMock(spec=events.MouseMove)
        tui_instance.on_mouse_move(mock_event)
        mock_event.stop.assert_not_called()


def test_handle_output_message_spinner_with_agent_name(tui_instance, monkeypatch):
    """
    Test that spinner status messages display the agent name prefix
    when a sub-agent is active.
    """
    # Mock query_one to return mock widgets for all lookup types
    mock_footer = MagicMock()
    mock_footer.spinner_suffix = ""
    mock_status_bar = MagicMock()
    mock_input_area = MagicMock()
    mock_input_container = MagicMock()
    mock_output_container = MagicMock()

    def mock_query_one(selector, *args):
        # query_one may be called with class or string selector
        if isinstance(selector, type):
            name = selector.__name__
        else:
            # String selector - could be CSS like "#input, InputArea"
            if "," in selector or "#" in selector:
                return mock_input_area
            name = "MainFooter"  # Default fallback for footer lookup

        mapping = {
            "MainFooter": mock_footer,
            "StatusBar": mock_status_bar,
            "InputContainer": mock_input_container,
            "InputArea": mock_input_area,
            "OutputContainer": mock_output_container,
        }
        return mapping.get(name, mock_footer)

    tui_instance.query_one = mock_query_one

    # Mock coder worker for agent service lookups
    mock_coder = MagicMock()
    mock_coder.uuid = "primary_uuid"
    tui_instance.worker = MagicMock()
    tui_instance.worker.coder = mock_coder

    # Mock AgentService so _resolve_agent_name works
    mock_agent_service = MagicMock()
    mock_agent_info = MagicMock()
    mock_agent_info.name = "researcher"
    mock_agent_info.coder = MagicMock()
    mock_agent_info.coder.uuid = "some_uuid"
    mock_agent_service.sub_agents = {"some_uuid": mock_agent_info}
    mock_agent_service.coder = mock_coder

    monkeypatch.setattr(
        "cecli.helpers.agents.service.AgentService.get_instance",
        lambda *args: mock_agent_service,
    )

    # Test: sub-agent spinner should include agent_name="researcher"
    msg = {
        "type": "spinner",
        "action": "start",
        "text": "Thinking...",
        "coder_uuid": "some_uuid",
    }
    tui_instance.handle_output_message(msg)
    mock_footer.start_spinner.assert_called_once_with("Thinking...", agent_name="researcher")

    # Test: primary agent spinner should have agent_name=None
    mock_footer.reset_mock()
    msg["coder_uuid"] = "primary_uuid"
    tui_instance.handle_output_message(msg)
    mock_footer.start_spinner.assert_called_once_with("Thinking...", agent_name=None)


def test_handle_output_message_confirmation_with_agent_name(tui_instance, monkeypatch):
    """
    Test that confirmation status messages display the agent name prefix.
    """
    mock_footer = MagicMock()
    mock_footer.spinner_suffix = ""
    mock_status_bar = MagicMock()
    mock_input_area = MagicMock()
    mock_input_container = MagicMock()
    mock_output_container = MagicMock()

    def mock_query_one(selector, *args):
        if isinstance(selector, type):
            name = selector.__name__
        else:
            if selector == "#input" or selector == "#input, InputArea":
                return mock_input_area
            elif selector == "#status-bar" or selector == "#status-bar, StatusBar":
                return mock_status_bar
            name = "MainFooter"  # Default fallback

        mapping = {
            "MainFooter": mock_footer,
            "StatusBar": mock_status_bar,
            "InputContainer": mock_input_container,
            "InputArea": mock_input_area,
            "OutputContainer": mock_output_container,
        }
        return mapping.get(name, mock_footer)

    tui_instance.query_one = mock_query_one

    # Mock coder worker for agent service lookups
    mock_coder = MagicMock()
    mock_coder.uuid = "primary_uuid"
    tui_instance.worker = MagicMock()
    tui_instance.worker.coder = mock_coder

    # Stub status_bar reference
    tui_instance.status_bar = mock_status_bar

    # Mock AgentService
    mock_agent_service = MagicMock()
    mock_agent_info = MagicMock()
    mock_agent_info.name = "researcher"
    mock_agent_info.coder = MagicMock()
    mock_agent_info.coder.uuid = "some_uuid"
    mock_agent_service.sub_agents = {"some_uuid": mock_agent_info}
    mock_agent_service.coder = mock_coder

    monkeypatch.setattr(
        "cecli.helpers.agents.service.AgentService.get_instance",
        lambda *args: mock_agent_service,
    )

    # Test: sub-agent confirmation should include agent_name="researcher"
    msg = {
        "type": "confirmation",
        "question": "Are you sure?",
        "options": {},
        "coder_uuid": "some_uuid",
    }
    tui_instance.handle_output_message(msg)
    mock_status_bar.show_confirm.assert_called_once_with(
        "Are you sure?",
        show_all=False,
        allow_tweak=False,
        allow_never=False,
        default="y",
        explicit_yes_required=False,
        agent_name="researcher",
    )


def test_handle_output_message_error_with_agent_name(tui_instance, monkeypatch):
    """
    Test that error status messages display the agent name prefix.
    """
    mock_footer = MagicMock()
    mock_footer.spinner_suffix = ""
    mock_status_bar = MagicMock()
    mock_input_area = MagicMock()
    mock_input_container = MagicMock()
    mock_output_container = MagicMock()

    def mock_query_one(selector, *args):
        if isinstance(selector, type):
            name = selector.__name__
        else:
            if "," in selector or "#" in selector:
                return mock_input_area
            return mock_footer
        mapping = {
            "MainFooter": mock_footer,
            "StatusBar": mock_status_bar,
            "InputContainer": mock_input_container,
            "InputArea": mock_input_area,
            "OutputContainer": mock_output_container,
        }
        return mapping.get(name, mock_footer)

    tui_instance.query_one = mock_query_one

    # Mock coder worker for agent service lookups
    mock_coder = MagicMock()
    mock_coder.uuid = "primary_uuid"
    tui_instance.worker = MagicMock()
    tui_instance.worker.coder = mock_coder

    # Stub status_bar reference
    tui_instance.status_bar = mock_status_bar

    # Mock AgentService - unknown UUID should return None (no prefix)
    monkeypatch.setattr(
        "cecli.helpers.agents.service.AgentService.get_instance",
        lambda *args: MagicMock(sub_agents={}, coder=mock_coder),
    )

    # Test: error message for unknown agent should have agent_name=None
    msg = {
        "type": "error",
        "message": "Something went wrong!",
        "coder_uuid": "unknown_uuid",
    }
    tui_instance.handle_output_message(msg)
    mock_status_bar.show_notification.assert_called_once_with(
        "Something went wrong!",
        severity="error",
        timeout=5,
        agent_name=None,
    )
