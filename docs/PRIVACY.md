# Privacy and Repo Safety

This project reads local usage data to build a HUD. Use this checklist to avoid leaking personal files when contributing.

## What the HUD Reads

- Claude token from macOS Keychain item `Claude Code-credentials`
- Codex local logs from `~/.codex/sessions/*.jsonl`
- Gemini local logs from `~/.gemini/tmp/*/chats/session-*.json`

## What the HUD Sends Over Network

- Claude usage request to `https://api.anthropic.com/api/oauth/usage`

## What Is Kept Local

- Codex and Gemini local logs are parsed locally
- No Codex/Gemini log upload is performed by this tool

## Pre-Commit Safety Checklist

Run before commit:

```bash
git status --short
git diff --staged --stat
git diff --staged
```

Verify ignore behavior for local artifacts:

```bash
git check-ignore -v __pycache__/ .venv/ .vscode/ .idea/ .usage-hud/
```

If a sensitive/local file was accidentally staged:

```bash
git restore --staged <path>
```

If a sensitive/local file was already tracked and should stop being tracked:

```bash
git rm --cached <path>
```

Then add an ignore rule in `.gitignore` and commit that change.

## Recommended Commit Scope

Prefer explicit adds instead of `git add .`:

```bash
git add usage_hud.py usage-hud README.md docs/SETUP.md docs/PRIVACY.md .gitignore
```
