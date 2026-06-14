"""SubAgentPills widget - displays active sub-agents as clickable pills.

DEPRECATED: This widget is not currently mounted in any TUI compose method.
The sub-agent pill display is handled inline via InputContainer.update_mode().
Kept for reference should TUI integration be desired in the future.
"""

from typing import Any

from textual.containers import Horizontal
from textual.message import Message
from textual.reactive import reactive
from textual.widgets import Static


class SubAgentPills(Horizontal):
    """Horizontal bar of sub-agent pills showing active agents.

    Each pill shows the agent name. The primary agent is shown as
    "primary". Active/selected sub-agents are highlighted.

    State is derived from AgentService via ``self.app.worker.coder``
    rather than maintained internally.  Uses a ``reactive`` attribute
    with ``recompose=True`` so Textual's built-in lifecycle manages
    mounting / removing child widgets.
    """

    DEFAULT_CSS = """
    SubAgentPills {
        height: 1;
        width: 1fr;
        margin: 0 1 0 1;
        padding: 0 0 0 0;
        overflow-x: hidden;
        overflow-y: hidden;
    }

    SubAgentPills > .pill {
        color: $accent;
        padding: 0 1 0 1;
        margin: 0 0 0 1;
        text-style: bold;
        width: auto;
        height: 100%;
    }

    SubAgentPills > .pill.active {
        color: $accent;
        text-style: bold;
        width: auto;
        height: 100%;
    }

    SubAgentPills > .pill.primary {
        color: $accent;
        text-style: bold;
        width: auto;
        height: 100%;
    }
    """

    class PillSelected(Message):
        """Emitted when a pill is clicked."""

        def __init__(self, agent_uuid: str) -> None:
            self.agent_uuid = agent_uuid
            super().__init__()

    # Reactive data — Textual will auto-recompose when this changes
    _pill_data: reactive[list[dict[str, Any]]] = reactive([], recompose=True)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _get_service(self):
        """Get the AgentService from the primary coder via the TUI app."""
        try:
            from cecli.helpers.agents.service import AgentService

            return AgentService.get_instance(self.app.worker.coder)
        except Exception:
            return None

    def compose(self):
        """Yield a pill ``Static`` for every entry in ``_pill_data``."""
        for pill_info in self._pill_data:
            yield Static(
                pill_info["name"],
                id=f"pill-{pill_info['uuid']}",
                classes=pill_info["classes"],
            )

    def sync(self) -> None:
        """
        Sync pills with the AgentService state.
        """
        service = self._get_service()
        if service is None:
            self._pill_data = []
            self.display = False
            return

        # Hide the pill bar when there are no sub-agents
        if not service.sub_agents:
            self.display = False
            self._pill_data = []
            return

        self.display = True

        # Determine active UUID (None → primary is active)
        primary_uuid = service.coder.uuid
        active_uuid = service.foreground_uuid
        if active_uuid is None and primary_uuid is not None:
            active_uuid = primary_uuid

        pills: list[dict] = []

        # Primary-agent pill
        if primary_uuid:
            classes = "pill"
            if active_uuid == primary_uuid:
                classes += " active"
            pills.append(
                {
                    "uuid": primary_uuid,
                    "name": "● primary" if active_uuid == primary_uuid else "○ primary",
                    "classes": classes,
                }
            )

        # Sub-agent pills
        for uuid_key, info in service.sub_agents.items():
            coder_uuid = str(info.coder.uuid)
            classes = "pill"
            if coder_uuid == active_uuid:
                classes += " active"
            pills.append(
                {
                    "uuid": coder_uuid,
                    "name": (
                        f"\u25cf {info.name}"
                        if coder_uuid == active_uuid
                        else f"\u25cb {info.name}"
                    ),
                    "classes": classes,
                }
            )
        # Let the reactive recompose system call compose() to rebuild children
        self._pill_data = pills

    def on_click(self, event) -> None:
        """Handle click events to identify which pill was clicked."""
        target = event.widget
        while target is not None and not isinstance(target, Static):
            target = target.parent

        if target is None:
            return

        widget_id = target.id or ""
        if widget_id.startswith("pill-"):
            uuid = widget_id[5:]
            self.post_message(self.PillSelected(uuid))
