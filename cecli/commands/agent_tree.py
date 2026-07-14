from typing import List

from cecli.commands.utils.base_command import BaseCommand
from cecli.helpers.agents.service import AgentService, SubAgentInfo, SubAgentStatus


class TreeNode:
    """A node in the agent hierarchy tree."""

    def __init__(self, agent_info: SubAgentInfo):
        self.agent_info = agent_info
        self.children: List[TreeNode] = []

    def add_child(self, child_node: "TreeNode"):
        self.children.append(child_node)


class AgentTreeCommand(BaseCommand):
    """Command to display the hierarchy of active agents."""

    NORM_NAME = "agent-tree"
    DESCRIPTION = "Display the hierarchy of active agents."

    @classmethod
    async def execute(cls, io, coder, args, **kwargs):
        """Execute the /agent-tree command."""
        agent_service = AgentService.get_instance(coder)
        if not agent_service:
            io.tool_output("Agent service not available.")
            return

        # Collect all agents: primary coder + sub-agents from sub_agents dict
        all_infos: List[SubAgentInfo] = []

        # Primary agent as a synthetic SubAgentInfo
        primary_coder = agent_service.coder
        primary_info = SubAgentInfo(
            name="primary",
            coder=primary_coder,
            parent_uuid=primary_coder.parent_uuid or "",
            status=SubAgentStatus.RUNNING,
        )
        all_infos.append(primary_info)

        # Sub-agents from the service's sub_agents dict
        for info in agent_service.sub_agents.values():
            if info and info.coder:
                all_infos.append(info)

        if len(all_infos) <= 1:
            io.tool_output("No active sub-agents found.")
            return

        root_nodes = cls._build_tree(all_infos)
        tree_output = cls._render_tree(root_nodes)
        io.tool_output(tree_output)

    @classmethod
    def _build_tree(cls, all_agents: List[SubAgentInfo]) -> List[TreeNode]:
        """Build the agent tree from a flat list of agents."""
        nodes = {agent.coder.uuid: TreeNode(agent) for agent in all_agents}
        root_nodes = []

        for agent in all_agents:
            parent_uuid = agent.coder.parent_uuid
            if parent_uuid and parent_uuid in nodes:
                parent_node = nodes[parent_uuid]
                child_node = nodes[agent.coder.uuid]
                parent_node.add_child(child_node)
            else:
                root_nodes.append(nodes[agent.coder.uuid])

        return root_nodes

    @classmethod
    def _render_tree(cls, nodes: List[TreeNode]) -> str:
        """Render the agent tree to a string."""
        output = []

        def render_node(node: TreeNode, prefix: str = "", is_last: bool = True):
            connector = "└── " if is_last else "├── "
            output.append(
                f"{prefix}{connector}{node.agent_info.name}"
                f" ({node.agent_info.status.value})"
                f" - {node.agent_info.coder.uuid}"
            )

            child_prefix = prefix + ("    " if is_last else "│   ")
            for i, child in enumerate(node.children):
                render_node(child, child_prefix, i == len(node.children) - 1)

        for i, node in enumerate(nodes):
            render_node(node, is_last=i == len(nodes) - 1)

        return "\n".join(output)
