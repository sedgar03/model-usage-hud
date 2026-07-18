"""Helpers for loading and saving app-only UI state."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .models import ProviderName, UiState

DEFAULT_UI_STATE_PATH = Path.home() / ".usage-hud" / "ui-state.json"
PROVIDER_NAMES: tuple[ProviderName, ...] = ("claude", "codex", "gemini", "system")


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


def _coerce_font_size(value: Any) -> float | None:
    """Accept a positive number, reject anything else so a corrupt state
    file can never push the window into an unusable 0.1pt render."""

    if value is None:
        return None
    try:
        size = float(value)
    except (TypeError, ValueError):
        return None
    if size <= 0.0:
        return None
    return size


def _coerce_selected_providers(value: Any) -> set[ProviderName] | None:
    """Accept a non-empty provider list, reject bad or empty persisted state."""

    if not isinstance(value, (list, tuple, set)):
        return None

    selected = {
        provider
        for provider in PROVIDER_NAMES
        if provider in value
    }
    if not selected:
        return None
    return selected


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
        font_size=_coerce_font_size(raw.get("font_size")),
        selected_providers=_coerce_selected_providers(raw.get("selected_providers")),
    )


def save_ui_state(state: UiState, path: Path = DEFAULT_UI_STATE_PATH) -> bool:
    """Persist UI state and return whether the write succeeded."""

    state_path = Path(path)
    x, y = state.window_position
    payload: dict[str, Any] = {
        "collapsed": {
            provider: bool(state.collapsed.get(provider, False))
            for provider in PROVIDER_NAMES
        },
        "window_position": [int(x), int(y)],
    }
    if state.font_size is not None:
        payload["font_size"] = float(state.font_size)
    if state.selected_providers is not None:
        payload["selected_providers"] = [
            provider
            for provider in PROVIDER_NAMES
            if provider in state.selected_providers
        ]

    try:
        state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = state_path.with_name(f"{state_path.name}.tmp")
        tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        tmp_path.replace(state_path)
    except OSError:
        return False
    return True
