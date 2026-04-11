"""Helpers for loading and saving app-only UI state."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .models import ProviderName, UiState

DEFAULT_UI_STATE_PATH = Path.home() / ".usage-hud" / "ui-state.json"
PROVIDER_NAMES: tuple[ProviderName, ...] = ("claude", "codex", "gemini")


def _default_collapsed() -> dict[ProviderName, bool]:
    return UiState().collapsed.copy()


def _coerce_collapsed(value: Any) -> dict[ProviderName, bool]:
    collapsed = _default_collapsed()
    if not isinstance(value, dict):
        return collapsed

    for provider in PROVIDER_NAMES:
        provider_value = value.get(provider)
        if isinstance(provider_value, bool):
            collapsed[provider] = provider_value
    return collapsed


def _coerce_window_position(value: Any) -> tuple[int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return UiState().window_position

    try:
        return (int(value[0]), int(value[1]))
    except (TypeError, ValueError):
        return UiState().window_position


def load_ui_state(path: Path = DEFAULT_UI_STATE_PATH) -> UiState:
    """Load persisted UI state, falling back to defaults on any bad input."""

    state_path = Path(path)
    try:
        raw = json.loads(state_path.read_text())
    except (OSError, json.JSONDecodeError):
        return UiState()

    if not isinstance(raw, dict):
        return UiState()

    return UiState(
        collapsed=_coerce_collapsed(raw.get("collapsed")),
        window_position=_coerce_window_position(raw.get("window_position")),
    )


def save_ui_state(state: UiState, path: Path = DEFAULT_UI_STATE_PATH) -> bool:
    """Persist UI state and return whether the write succeeded."""

    state_path = Path(path)
    x, y = state.window_position
    payload = {
        "collapsed": {
            provider: bool(state.collapsed.get(provider, False))
            for provider in PROVIDER_NAMES
        },
        "window_position": [int(x), int(y)],
    }

    try:
        state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = state_path.with_name(f"{state_path.name}.tmp")
        tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        tmp_path.replace(state_path)
    except OSError:
        return False
    return True
