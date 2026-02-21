# model-usage-hud

Unified terminal HUD for monitoring usage across:
- Claude Code subscription windows (5h, 7d, 7d-opus)
- OpenAI Codex local rate-limit snapshots from `~/.codex/sessions/*.jsonl`
- Gemini local request + token usage from `~/.gemini/tmp/*/chats/session-*.json`

## Features

- Single command: `usage-hud`
- Minimal two-line provider badges on the left (no text labels):
  - Claude: /CC\ and \CC/ in brown
  - OpenAI: /OA\ and \OA/ in white
  - Gemini: /GM\ and \GM/ in cyan
- Claude-HUD style pace bars with expected-usage marker
- Cleaner mini rows: delta + target in parentheses (for example `-36 (43%)`)
- Defaults to your preferred mode: `--mini --force`
- Default Codex view shows only the main `codex` bucket
- Optional `--all-limits` to also show model-specific buckets (like `CODEX_BENGALFOX`)
- macOS default: always-on-top frameless PiP-style window (font 7.5, geometry `320x130+40+40`)
- Gemini mini bars use configurable request limits (defaults: `120/min` and `1500/day`)
- JSON output mode for scripting
- Uses local Codex logs (no OpenAI API key required)

## Quick Start

```bash
cd ~/Code/model-usage-hud
./usage-hud
```

Run one snapshot:

```bash
./usage-hud --once --no-always-on-top
```

Show all Codex buckets:

```bash
./usage-hud --once --all-limits --no-always-on-top
```

Run in always-on-top mode explicitly:

```bash
./usage-hud --always-on-top
```

Run in terminal mode (opt out of topmost default):

```bash
./usage-hud --no-always-on-top
```

Run topmost with title bar:

```bash
./usage-hud --always-on-top-framed
```

Run topmost with smaller text:

```bash
./usage-hud --always-on-top-font-size 7
```

Run topmost without macOS title bar (frameless):

```bash
./usage-hud --always-on-top-geometry 320x130+40+40
```

Run topmost in monochrome:

```bash
./usage-hud --no-color
```

Show only one provider (example: Codex only):

```bash
./usage-hud --providers codex
```

JSON output:

```bash
./usage-hud --json --once
```

## Install as Global Command

From the repo root:

```bash
python3 -m pip install -e .
```

Then run anywhere:

```bash
usage-hud
```

## Documentation

- Setup and provider configuration: `docs/SETUP.md`
- Privacy and safe commits: `docs/PRIVACY.md`

## Reading the Mini Bars

Each window line shows:
- `%` actual utilization
- bar with a vertical marker `│` for expected utilization at this point in the window
- signed delta (`+/-`) versus expected pace
- target utilization in parentheses (for example `(43%)`)

Window labels:
- Claude/Codex: `S` = short window, `W` = week window
- Gemini: `M` = minute request window, `D` = daily request window

Interpretation:
- red delta: spending faster than steady pace
- green delta: spending slower than steady pace
- marker near actual fill: on pace

## Options

```bash
usage-hud --help
```

Notable options:
- `--mini` compact view
- `--force` replace existing HUD lock
- `--interval 15` refresh every 15 seconds
- `--no-alt-screen` keep scrollback
- `--all-limits` show all Codex limit buckets
- `--providers codex,gemini` show only selected providers (`claude`, `codex`, `gemini`; default: all)
- `--always-on-top` force topmost HUD window
- `--no-always-on-top` force terminal rendering
- `--always-on-top-font-size 7` smaller/larger text size in topmost mode (default: 7.5)
- `--always-on-top-geometry 320x130+40+40` set initial topmost window size/position (default: `320x130+40+40`)
- `--always-on-top-frameless` hide title bar in topmost mode (default on macOS)
- `--always-on-top-framed` show title bar in topmost mode
- `--bar-style solid|legacy|auto` choose bar glyph style (default `auto`: solid in topmost, legacy in terminal)
- `--codex-sessions-dir /path/to/sessions`
- `--gemini-tmp-dir /path/to/.gemini/tmp`
- `--gemini-minute-limit-requests 120`
- `--gemini-day-limit-requests 1500`
- `--no-color`

## Notes

- Claude usage is fetched from Anthropic OAuth usage API using your macOS Keychain `Claude Code-credentials` item.
- If Claude credentials are missing, the HUD still shows Codex data.
- Gemini request usage is estimated from local Gemini CLI session logs by counting Gemini responses as requests and comparing against request limits.
- Gemini token totals are still included in `--json` output for reference (`minute.used_tokens`, `day.used_tokens`, and per-model totals).
- Gemini defaults (`120/min`, `1500/day`) match Google AI Pro quotas as of February 21, 2026; override with `--gemini-minute-limit-requests` and `--gemini-day-limit-requests` if your account limits differ.
- Lock file default is `~/.usage-hud/usage-hud.lock`; if unavailable, it falls back to `/tmp/usage-hud.lock`.
- On macOS, topmost mode is enabled by default; use `--no-always-on-top` for terminal mode.
- `--always-on-top` is only supported on macOS and cannot be combined with `--json` or `--once`.
- If `--always-on-top` fails with `No module named '_tkinter'`, install Tk for your Python version (for example `brew install python-tk@3.14`) and start a new shell.
- Topmost mode preserves HUD color cues by default; use `--no-color` for monochrome.
- In `--always-on-top-frameless` mode, drag anywhere in the HUD to move it and use `Esc`, `Cmd+W`, or `q` to close.
