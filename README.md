# llmsh

A terminal chat and repo summarization interface for LLMs using [PIG](https://github.com/hosamsh/pig) inference gateway.

The interactive `chat` and `ask` flows enable direct interactions with the models servred through PIG. `summarize` analyzes and reduces large files or entire folders based on user instructions.

## Install

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/hosamsh/llmsh
cd llmsh
uv sync
uv tool install .
```

## Usage

Run `llmsh` to open the interactive chat. First run walks you through setup.

- `llmsh` interactive chat tui for day-to-day work with PIG models.
- `llmsh ask` one-shot path for quick prompts in scripts or shell workflows.
- `llmsh summarize` for analyzing files, folders, and codebases with map-reduce style summarization.

### CLI commands

```
llmsh                  Interactive chat (default)
llmsh ask <prompt>     One-shot prompt, prints response to stdout
llmsh summarize <path> <instruction>   Analyze files or folders with map-reduce summarization
llmsh doctor           Run diagnostics on current config
llmsh profile list|add|use|set-model|remove   Manage profiles
llmsh endpoint list|add|remove                Manage API endpoints
llmsh session list|show|delete                Manage saved sessions
```

`ask` options: `--stdin`, `--json`, `--profile <name>`, `--system <text>`

### Slash commands (inside the chat)

```
/help                  Show command reference
/clear                 Clear the transcript
/retry                 Re-send the last message
/copy                  Copy last assistant message
/save                  Save current session
/load [id]             List or load a session
/session               Session management (list/save/load/delete)
/profile               Manage profiles (list/add/use/set-model/remove)
/endpoint              Manage endpoints (list/add/remove)
/doctor                Run diagnostics
/summarize <path> <instruction>   Analyze files or folders with map-reduce summarization
/cancel                Cancel active flow or request
```

## Configuration

Config file: `~/.config/llmsh/config.toml`

Repository example: `config.example.toml`

Set up interactively via `llmsh` (onboarding) or `llmsh endpoint add` / `llmsh profile add` from the command line. Or edit the TOML directly:

```toml
current_profile = "default"

[endpoints.local]
base_url = "http://localhost:8006/v1"
auth_mode = "none"

[profiles.default]
endpoint = "local"
model = "your-model-name"
max_tokens = 1024
```

For endpoints requiring API keys:

```toml
[endpoints.openai]
base_url = "https://api.openai.com/v1"
auth_mode = "api_key"
api_key_env = "OPENAI_API_KEY"
```

## Development

```bash
uv sync --extra dev
uv run pytest
uv run ruff check src/ tests/
uv run ty check src/
```

## License

MIT — see [LICENSE](LICENSE).
