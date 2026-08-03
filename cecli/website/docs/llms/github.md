---
parent: Connecting to LLMs
nav_order: 510
---

# GitHub Copilot

Cecli can connect to GitHub Copilot’s LLMs because Copilot exposes a standard **OpenAI-style** endpoint at:

```
https://api.githubcopilot.com
```

First, install cecli:

```bash
uv tool install cecli-dev
```

---

When you specify a github copilot model on start up (e.g. `github_copilot/gpt-5-mini`), [litellm](https://github.com/BerriAI/litellm) will enter a github auth workflow wherein you will connect to your github account with the provided auth code to grant the system access. Further details can be found [here](https://docs.litellm.ai/docs/providers/github_copilot).