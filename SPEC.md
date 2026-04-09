# Model Usage HUD — Same-Repo CLI + App Migration Spec

## Overview

Evolve `model-usage-hud` into a small multi-entrypoint project without splitting
the repository and without breaking the current CLI contract.

The current repo already contains:

- shared provider-fetch logic
- shared pacing and burn-rate logic
- a terminal renderer
- a minimal always-on-top Tk window

The next step is not "replace the repo with a desktop app". The next step is:

1. extract a shared core contract
2. keep `usage-hud` working exactly as it does now
3. add a separate desktop app frontend on top of the same core

Future target: a PySide6 macOS app, optionally packaged with Briefcase.

## Goals

- Keep the existing `usage-hud` command stable for terminal and Tk usage.
- Stay in one repository.
- Avoid duplicating provider, pacing, and burn-rate logic.
- Create a clean seam for a future PySide6 UI.
- Add richer UI features only after the shared core contract exists.

## Non-Goals

- No new repository.
- No Electron/Tauri rewrite.
- No forced dependency on PySide6 for CLI users.
- No silent behavioral changes to `--once`, `--json`, or `--no-always-on-top`.

## Principles

- One product, multiple frontends.
- The CLI remains the source of truth until the app reaches feature parity.
- Data fetch, normalization, and rendering are separate concerns.
- UI code must consume structured view models, not ANSI text.
- Desktop refreshes must be single-flight: no overlapping fetch workers.

## Target Repo Layout

```text
model-usage-hud/
  usage_hud.py                 # existing CLI + Tk implementation, kept working
  usage-hud                    # existing local launcher, kept working
  model_usage_hud/
    __init__.py
    cli.py                     # package entrypoint; delegates to usage_hud.main()
    core/
      __init__.py
      models.py                # shared typed contracts for snapshots/view models
      builders.py              # future: normalize raw snapshots into UI-ready data
      state.py                 # future: UI state persistence helpers
    app/
      __init__.py
      main.py                  # future PySide6 entrypoint
      window.py                # future main window
      widgets/
        __init__.py
        provider_card.py
        button_bar.py
      styles.py
  pyproject.toml
  README.md
```

## Entrypoints

### Current

- `usage-hud`
  - Console-first entrypoint
  - Preserves current behavior
  - Supports terminal rendering, JSON mode, and Tk always-on-top mode

### Future

- `usage-hud-app`
  - Separate desktop-app entrypoint
  - Requires PySide6
  - Does not replace the CLI

This split is intentional. The desktop app should be additive, not a breaking
rename of the current command.

## Architecture

### Layer 1: Fetch

Raw provider access and caching stay close to the current implementation:

- Claude API fetch and backoff
- Codex local-session parsing
- Gemini API fetch and backoff
- single-instance lock semantics

This logic currently lives in `usage_hud.py` and should remain reusable.

### Layer 2: Normalize

Add a structured intermediate layer that turns raw snapshots into stable view
models. This is the missing seam in the current codebase.

The desktop UI must not parse:

- ANSI escape sequences
- padded strings
- terminal-specific spacing
- inline badge formatting

Instead, both CLI and app should consume a common model shaped roughly like:

- snapshot bundle
- provider section
- metric row
- provider status
- highest utilization
- stale/error metadata

### Layer 3: Render

Renderers become thin consumers of the normalized layer:

- CLI renderer: terminal text / ANSI
- Tk renderer: current text-based window
- PySide renderer: widgets and styles

## Required Refactor Before PySide6

Before adding real Qt widgets, implement a shared builder such as:

```python
build_provider_view_models(...)
```

It should return structured provider sections derived from the same logic that
currently powers:

- `render_claude_mini()`
- `render_codex_mini()`
- `render_gemini_mini()`
- burn-rate tracking
- stale-status handling

The CLI renderer can then continue to format those models as ANSI text, while
the PySide frontend renders the same data as widgets.

## Desktop App Scope

The first desktop version should be intentionally narrow:

- always-on-top window
- frameless option with drag-to-move
- provider cards for Claude, Codex, Gemini
- collapsible sections
- mute toggle buttons for Claude and Codex
- persisted UI state

It should not introduce unrelated product scope before parity exists.

## Window Behavior

### App Defaults

- top-left start position: `(40, 40)`
- dark theme matching the current HUD
- always-on-top on macOS
- frameless by default on macOS

### Geometry Rules

The app must preserve the current geometry logic rather than replacing it with a
hard-coded width:

- base width tracks current defaults
- speedometer mode may widen the window
- single-provider mode may shrink width
- collapsed/expanded state affects height

Saved window state must define precedence explicitly:

1. explicit CLI/app launch arguments
2. saved UI state
3. computed defaults

## Refresh Model

Desktop refreshes use a timer plus a worker thread, but the contract must be:

- only one active refresh at a time
- manual refresh requests are ignored or queued while a fetch is running
- cache state is not mutated concurrently from overlapping workers

This is required because current Claude and Gemini backoff state is stored in
module-level mutable dictionaries.

## Single-Instance Behavior

The desktop app must preserve the current single-instance intent.

Decide one of these behaviors and implement it explicitly:

1. second launch focuses existing window and exits
2. second launch errors unless `--force`
3. second launch replaces existing instance with `--force`

Do not drop lock semantics during the GUI migration.

## CLI Compatibility Requirements

The following behavior must remain intact:

- `usage-hud --once`
- `usage-hud --json`
- `usage-hud --no-always-on-top`
- `usage-hud --providers ...`
- `usage-hud --all-limits`
- `usage-hud --speedometer`
- current default macOS always-on-top Tk path

PySide6 must be optional. A CLI-only user should not need Qt installed.

## Mute Controls

### Claude

- Control sentinel path: `~/.claude/mute`
- Read current state from disk on refresh
- Toggle by creating or deleting the file

### Codex

- Control sentinel path: `~/.codex/mute`
- Same sentinel behavior as Claude
- UI may exist before a Codex sound hook exists, but the spec must call that out

### Contract

Use home-relative paths in code and docs. Do not encode machine-specific
absolute paths into the product contract.

## State Persistence

Persist UI-only state to:

`~/.usage-hud/ui-state.json`

Initial shape:

```json
{
  "collapsed": {
    "claude": false,
    "codex": false,
    "gemini": false
  },
  "window_position": [40, 40]
}
```

Future additions may include:

- width/height
- frameless preference
- last-selected providers

State-file corruption must fail soft and fall back to defaults.

## Dependencies

### Required

- no new required runtime dependencies for the CLI path

### Optional

- `PySide6>=6.6` as an optional `gui` dependency

Example direction for `pyproject.toml`:

```toml
[project.optional-dependencies]
gui = ["PySide6>=6.6"]

[project.scripts]
usage-hud = "model_usage_hud.cli:main"

# Add later, when the app is real:
# usage-hud-app = "model_usage_hud.app.main:main"
```

Do not make the GUI entrypoint the default `usage-hud` command.

## Phases

### Phase 0: Spec + Package Scaffold

- add a real Python package namespace
- keep `usage_hud.py` working
- route packaged CLI through a thin wrapper
- define shared contracts for future view models

### Phase 1: Shared View Models

- extract normalized provider/metric builders
- make CLI rendering consume shared models
- keep output visually equivalent

### Phase 2: Thin PySide App Shell

- add a minimal app window
- use structured provider models
- no extra features beyond parity-critical UI

### Phase 3: UX Features

- collapsible provider cards
- Claude/Codex mute buttons
- persisted UI state
- keyboard shortcuts

### Phase 4: Packaging

- add Briefcase config
- package a macOS `.app`
- later, if useful, produce a `.dmg`

## Acceptance Criteria

Before calling the architecture migration successful:

- packaged `usage-hud` still behaves like today
- direct `./usage-hud` still behaves like today
- CLI path works without PySide6 installed
- app frontend does not duplicate fetch/render business logic
- no overlapping refresh workers occur
- saved UI state restores safely
- second-launch behavior is explicit and tested

## Test Matrix

Minimum validation set:

- `usage-hud --once --no-always-on-top`
- `usage-hud --json --once`
- `usage-hud --providers codex`
- `usage-hud --all-limits --once --no-always-on-top`
- `usage-hud --speedometer --once --no-always-on-top`
- missing Claude credentials
- stale Claude/Gemini cache behavior
- Codex with no session events
- second launch with and without `--force`
- corrupted `~/.usage-hud/ui-state.json`

## Open Questions

1. Should the future desktop app focus an existing instance or honor the current
   lock-and-error model?
2. Do we want a menu bar app later, or is a floating HUD enough?
3. Should Codex mute ship before Codex has a sound hook, or should the control
   stay hidden until the hook exists?
