"""SubAgentCoder - a Coder variant for sub-agents.

Extends AgentCoder but excludes the Delegate tool from its tool schemas
so sub-agents cannot spawn further sub-agents.
"""

import logging

from cecli.coders.agent_coder import AgentCoder

logger = logging.getLogger(__name__)


class SubAgentCoder(AgentCoder):
    """Coder for sub-agents that disallows spawning further sub-agents."""

    edit_format = "subagent"
    prompt_format = "subagent"

    def post_init(self):
        super().post_init()
        if not self.agent_config.get("allow_nested_delegation", False):
            self.registered_tools["excluded"].add("delegate")
