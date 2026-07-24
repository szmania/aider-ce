---
parent: Configuration
nav_order: 42
description: WebSocket API for connecting external tools and UIs to cecli's event stream.
---

# WebSocket API
{: .no_toc }

cecli includes an optional WebSocket server that broadcasts real-time events
and accepts input from external clients. This enables custom UIs, dashboards,
and integrations to observe and interact with cecli sessions programmatically.

## Activation

The server is controlled by two command-line flags:

```
cecli --ws-port 23254 --ws-host 127.0.0.1
```

| Flag | Default | Description |
|------|---------|-------------|
| `--ws-port` | `23254` | Port to listen on. Set to `0` to disable the server. |
| `--ws-host` | `127.0.0.1` | Host interface to bind to. |

Both can also be set in your config file:

```yaml
# ~/.cecli/config.yml
ws-port: 23254
ws-host: "127.0.0.1"
```

## Connecting

Point any WebSocket client to `ws://{host}:{port}`:

```bash
# Example with websocat
websocat ws://127.0.0.1:23254
```

## Server → Client Events

The server broadcasts JSON messages for every major IO event. Each message
has an `event` field identifying the type and a `data` dictionary with
payload fields:

```json
{"event": "tool_output", "data": {"text": "...", "coder_uuid": "..."}}
```

| Event | Payload | When |
|-------|---------|------|
| `tool_output` | `text`, `coder_uuid` | Assistant tool output |
| `tool_call` | `lines`, `coder_uuid` | Assistant issued a tool call |
| `tool_result` | `text`, `coder_uuid` | Tool execution result |
| `stream_chunk` | `text`, `coder_uuid` | Streaming response chunk |
| `start_response` | `coder_uuid` | Assistant starts responding |
| `end_response` | `coder_uuid` | Assistant finishes responding |
| `spinner` | `action`, `text`, `coder_uuid` | Spinner state change |
| `start_task` | `task_id`, `title`, `task_type`, `coder_uuid` | Background task started |
| `cost_update` | `cost`, `coder_uuid` | Token cost update |
| `error` | `message`, `coder_uuid` | An error occurred |
| `ready_for_input` | `files`, `commands`, `chat_files`, `coder_uuid` | Ready for user input |
| `confirmation` | `question`, `subject`, `options`, `coder_uuid` | Awaiting user confirmation |

## Client → Server Messages

Clients can send JSON messages to provide input back to cecli:

### User input

```json
{"type": "user_input", "text": "your message", "coder_uuid": "..."}
```

### Confirmation response

```json
{"type": "confirmation", "confirmed": true, "coder_uuid": "..."}
```

The `coder_uuid` field is optional. When omitted, the message is routed to the
primary coder via blinker signals.

## Example

Using a simple Python script to listen for events:

```python
import asyncio
import json
from websockets.asyncio.client import connect

async def listen():
    async with connect("ws://127.0.0.1:23254") as ws:
        async for msg in ws:
            data = json.loads(msg)
            print(f"[{data['event']}] {data['data']}")

asyncio.run(listen())
```