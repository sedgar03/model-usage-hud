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
./usage-hud --once
```

Show all Codex buckets:

```bash
./usage-hud --once --all-limits
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

## Reading the Mini Bars

Each window line shows:
- `%` actual utilization
- bar with a vertical marker `┊` for expected utilization at this point in the window
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
