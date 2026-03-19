# Medical Follow-up Agent

A voice AI medical follow-up coordinator built with LiveKit Agents (Python). The agent conducts recovery check-ins in natural Hinglish, assesses status, and generates a structured JSON summary at the end of each conversation.

## What this project does

- Runs a real-time voice agent on LiveKit.
- Greets patients and collects symptom/recovery updates conversationally.
- Uses multilingual speech and turn detection optimized for mixed-language conversations.
- Produces a structured outcome (`recovered`, `needed`, `physical-visit`, or `emergency`).
- Saves call summaries as timestamped JSON files in the `data/` folder.

## Tech stack

- Python 3.10+
- [LiveKit Agents SDK](https://docs.livekit.io/agents/)
- STT: Deepgram `nova-3` (`multi` language)
- LLM: OpenAI `gpt-4.1-mini` (main dialogue)
- Summary LLM: Gemini `gemini-2.5-flash`
- TTS: Cartesia `sonic-3`
- VAD / turn detection: Silero + LiveKit multilingual turn detector
- Package/tooling: `uv`, `pytest`, `ruff`

## Project layout

```text
src/
   agent.py          # Agent entrypoint and core behavior
data/
   *.json            # Generated call summaries
pyproject.toml      # Dependencies and tooling config
AGENTS.md           # Contributor/agent coding guidance
```

## Prerequisites

- Python 3.10 or newer
- `uv` installed
- LiveKit Cloud project credentials
- Provider credentials for the configured STT/LLM/TTS models

## Setup

1. Install dependencies:

    ```bash
    uv sync
    ```

2. Configure environment variables:

    ```bash
    copy .env.example .env.local
    ```

    Fill required keys in `.env.local` (for example `LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET`, and model provider credentials).

3. Download required local model artifacts:

    ```bash
    uv run python src/agent.py download-files
    ```

## Run the agent

- Local console mode:

   ```bash
   uv run python src/agent.py console
   ```

- Development worker mode:

   ```bash
   uv run python src/agent.py dev
   ```

## Conversation summary output

At call end, the `get_summary` tool builds and returns JSON, and persists the same payload to `data/<timestamp>.json`.

Expected shape:

```json
{
   "follow_up_needed": "needed | recovered | physical-visit | emergency",
   "conversation_summary": "Brief summary",
   "conversation": [
      { "speaker": "agent", "text": "..." },
      { "speaker": "user", "text": "..." }
   ]
}
```

## Notes

- Entrypoint remains `src/agent.py`.
- Conversation summary files are written in UTC timestamp format.
- Do not commit `.env.local` or secrets.
