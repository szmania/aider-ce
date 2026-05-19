---
parent: Configuration
nav_order: 40
description: Sub-agents enable autonomous delegation of specialized tasks to dedicated LLM sessions within the same TUI session.
---

# Sub-Agents

Sub-agents allow the primary coding agent to delegate specialized sub-tasks to dedicated child agent sessions. Each sub-agent runs its own LLM loop with its own tools, conversation history, and system prompt — all within the same TUI session. This enables parallel and sequential task decomposition without leaving your workflow.

Sub-agents can be used for:

- **Code review** — have a dedicated reviewer analyze changes in parallel
- **Testing** — delegate test writing to a specialist agent
- **Research** — explore documentation or codebase structure while the primary agent works on other tasks
- **Multi-perspective analysis** — get feedback from agents with different model backends or system prompts

## Configuration

### Defining Sub-Agents

Sub-agents are defined using Markdown files (`.md`) with YAML front matter. The front matter specifies the agent's name and optional model override, while the body content becomes the agent's system prompt.

Sub-agent definition files can be placed in any directory. You can configure which directories cecli scans using the `subagent_paths` option.

### Sub-Agent File Format

```markdown
---
name: reviewer
model: deepseek/deepseek-v4-pro
---
You are a code review specialist. Your job is to analyze code changes,
identify bugs, security issues, and style problems. Be thorough but
constructive in your feedback. Always provide specific line numbers
and suggestions for improvement.
```

#### Front Matter Fields

| Field | Required | Description |
|-------|----------|-------------|
| `name` | Yes | Unique name used to reference the sub-agent in commands and the Delegate tool |
| `model` | No | Model override for this sub-agent. If omitted, inherits the parent agent's model |

#### System Prompt

Any content after the closing `---` of the front matter becomes the sub-agent's system prompt. This replaces the default main system prompt for that agent. You can use this to define the sub-agent's role, behavior, and constraints.

### Configuration File

Add sub-agent paths to your YAML configuration file:

```yaml
# .cecli.conf.yml or ~/.cecli.conf.yml
agent-config:
    max_sub_agents: 3  # Maximum concurrent sub-agents (default: 3)
    subagent_paths:
        - ".cecli/subagents"  # Default path
        - "~/team-agents"     # Custom path for shared agent definitions
```

## Usage

### Available Commands

| Command | Description |
|---------|-------------|
| `/invoke-agent <name> <prompt>` | Invoke a sub-agent with a prompt (blocking — waits for completion) |
| `/spawn-agent <name>` | Spawn a sub-agent without a prompt (non-blocking — waits for user input) |
| `/reap-agent` | Force destroy the currently active sub-agent |

> **Tip**: Both `/invoke-agent` and `/spawn-agent` support tab completion of sub-agent names.

### Invoking a Sub-Agent (Blocking)

The most common way to use sub-agents. The primary agent waits for the sub-agent to finish:

```
/invoke-agent reviewer Can you review the changes in editblock_func_coder.py?
```

This sends the prompt to the reviewer sub-agent, which works autonomously and returns a summary when done.

### Delegating from the Primary Agent

The primary agent can also delegate work using the `Delegate` tool. This enables the autonomous workflow:

1. The primary agent analyzes a task
2. It decomposes the work into sub-tasks
3. It delegates each sub-task to the appropriate sub-agent
4. Sub-agents work independently and return their summaries
5. The primary agent synthesizes the results

### Spawning a Sub-Agent (Non-Blocking)

Creates a sub-agent that waits for you to interact with it directly:

```
/spawn-agent tester
```

Once spawned, you can switch to it and type messages directly.

### Reaping a Sub-Agent

Forcefully destroy the currently active sub-agent and reclaim its resources:

```
/reap-agent
```

This is useful if a sub-agent is stuck, misbehaving, or you no longer need its work.

## TUI Integration

### Switching Between Agents

When sub-agents are active, the TUI shows agent pills in the input container's border title, displaying each agent with status icons:

```
┌─ agent: ○ primary  ◆ reviewer  ○ tester ─────────────────┐
```

- **Keyboard**: Use `Ctrl+Alt+Left` / `Ctrl+Alt+Right` to cycle through agents. Use `Ctrl+Alt+Up` to return to the primary agent.

### Container Routing

Each agent gets its own output container. When you switch agents:

1. The active container is shown; all others are hidden
2. Your input is routed to the active agent
3. Tool output, streaming responses, and task notifications are displayed in the correct container
4. Agent pills in the border title highlight the active agent

## Lifecycle and Limits

### Max Sub-Agents

The `max_sub_agents` setting (default: 3) limits how many concurrent sub-agents can exist. This prevents resource exhaustion.

When the limit is reached:

- If any sub-agents have **finished**, the oldest finished one is automatically reaped to make room
- If all sub-agents are still **running**, a `RuntimeError` is raised. You must wait for one to finish or use `/reap-agent` to free resources.

### Cleanup

- **Normal completion**: A sub-agent calls `Finished(summary="...")` which marks it as finished. Its container remains visible but its resources are eligible for lazy cleanup.
- **Session end**: When the parent session ends, all sub-agents are automatically cleaned up.
- **Force cleanup**: Use `/reap-agent` to immediately destroy a sub-agent and reclaim all resources.

## Restrictions

- **No nested sub-agents by default**: Sub-agents cannot spawn further sub-agents. The `Delegate` tool is excluded from sub-agent tool schemas by default. To enable nested delegation, set `allow_nested_delegation: true` in the agent configuration.
- **TUI-dependent**: Sub-agent container switching and the reap command depend on the TUI. Running in headless or non-TUI modes may not support these features.

## Examples

### Example 1: Code Review Workflow

```yaml
# .cecli/subagents/reviewer.md
---
name: reviewer
model: deepseek/deepseek-v4-pro
description: A sub agent for reviewing edited code
---
You are a code review specialist. Your job is to analyze code changes,
identify bugs, security issues, and style problems. Be thorough but
constructive in your feedback. Always provide specific line numbers
and suggestions for improvement.
```

```
/invoke-agent reviewer Please review the last 5 commits in this branch
```

### Example 2: Test Writing Workflow

```yaml
# .cecli/subagents/tester.md
---
name: tester
model: gemini/gemini-3-flash-preview
description: A sub agent for running tests and interpreting results
---
You are a testing specialist. Your job is to write comprehensive tests
for code changes. You should cover edge cases, error conditions, and
happy paths. Use the project's existing testing patterns and conventions.
```

```
/invoke-agent tester Write unit tests for the new AgentService.invoke() method
```

### Example 3: Multi-Agent Review

By defining multiple sub-agents, you can get different perspectives on the same code:

1. Delegate to a **reviewer** to analyze security concerns
2. Delegate to a **tester** to identify test gaps
3. The primary agent synthesizes both reports into an action plan

## See Also

- [Agent Mode](/config/agent-mode)
- [Custom Commands](/config/custom-commands)
- [Custom System Prompts](/config/custom-system-prompts)
- [Hooks](/config/hooks)