# Setup Guide

This guide is for first-time setup on a new machine.

## 1) Install

From the repo root:

```bash
python3 -m pip install -e .
```

Run from this repo during development:

```bash
./usage-hud
```

Run globally after install:

```bash
usage-hud
```

## 2) Provider Configuration

### Claude Code

1. Install Claude Code CLI.
2. Authenticate:

```bash
claude login
```

3. Verify credentials exist in macOS Keychain:

```bash
security find-generic-password -s "Claude Code-credentials" -w >/dev/null && echo "Claude auth OK"
```

The HUD fetches Claude usage from Anthropic's usage API using this keychain item.

### OpenAI Codex

1. Install and authenticate the Codex CLI.
2. Run at least one Codex session so local logs exist under `~/.codex/sessions`.
3. Verify logs exist:

```bash
find ~/.codex/sessions -name '*.jsonl' | head
```

The HUD reads local `token_count` events from these files.

### Gemini

1. Install and authenticate Gemini CLI.
2. Run at least one Gemini session so logs exist under `~/.gemini/tmp`.
3. Verify logs exist:

```bash
find ~/.gemini/tmp -name 'session-*.json' | head
```

The HUD estimates Gemini request usage from local session files.

## 3) Quick Health Check

Use one-shot JSON mode (terminal mode required):

```bash
./usage-hud --once --json --no-always-on-top
```

You should see:
- `claude.status` as `Live usage` (or a clear auth/network error)
- `codex.status` as `Local usage` if Codex logs exist
- `gemini.status` as `Local usage` if Gemini logs exist

## 4) Common Runtime Modes

Default on macOS:

```bash
./usage-hud
```

Terminal mode:

```bash
./usage-hud --no-always-on-top
```

Adjust topmost window:

```bash
./usage-hud --always-on-top-font-size 7 --always-on-top-geometry 320x130+40+40
```

## 5) Troubleshooting

`Could not initialize topmost HUD window: missing Tk bindings`:

```bash
brew install python-tk@3.14
exec zsh
```

Then retry `./usage-hud`.

If bars look too dashed in your font renderer:

```bash
./usage-hud --bar-style solid
```
