"""Typed contracts for the shared view-model layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

ProviderName = Literal["claude", "codex", "gemini", "kimi", "system"]
TextStyle = Literal[
    "plain",
    "bold",
    "dim",
    "red",
    "green",
    "yellow",
    "cyan",
    "white",
    "orange",
    "brown",
    "bold_red",
    "bold_green",
    "bold_white",
]

PaceBarCellKind = Literal["filled", "empty", "marker"]
PaceBarTone = Literal["cyan", "red", "green", "orange", "dim", "white"]


@dataclass(slots=True, frozen=True)
class PaceBarCell:
    """One cell of a pace bar, rendered identically by CLI ANSI and Qt painters.

    ``kind`` tells a renderer whether the cell is the expected-pace marker, a
    filled data cell, or empty space. ``tone`` is an abstract palette name the
    renderer maps to its own color system (ANSI code, Qt ``QColor``, CSS).
    """

    kind: PaceBarCellKind
    tone: PaceBarTone


@dataclass(slots=True)
class NoteLine:
    """Structured representation of one note/status line."""

    text: str
    style: TextStyle = "plain"


@dataclass(slots=True)
class MetricRow:
    """Structured representation of one utilization row."""

    label: str
    utilization: float | int | None
    expected_utilization: float | int | None
    delta: float | int | None
    display_mode: Literal["pace", "value_only"] = "pace"
    reset_at: str | int | None = None
    stale: bool = False
    prefix: str | None = None
    prefix_style: TextStyle = "dim"
    burn_rate_per_hour: float | None = None
    eta_hours: float | None = None
    # Free-form trailing context for gauge rows (no pace target), e.g.
    # "263 GB free" or "load 6.3". Rendered in the detail column with
    # ``detail_style``. Ignored for pace rows, which build their own detail
    # from delta/target/speedometer.
    detail: str | None = None
    detail_style: TextStyle = "dim"
    # Optional fill-color override for gauge rows. ``None`` colors the bar by
    # its own value (usage_style); a concrete style lets a caller drive the
    # color from a different health signal — e.g. MEM tinted by macOS memory
    # pressure rather than by used-%.
    gauge_style: TextStyle | None = None


@dataclass(slots=True)
class ProviderSection:
    """UI-ready representation of a provider block."""

    provider: ProviderName
    title: str
    status: str
    highest_utilization: float | int | None
    rows: tuple[MetricRow, ...] = ()
    notes: tuple[NoteLine, ...] = ()
    stale: bool = False
    accent: str = ""


@dataclass(slots=True)
class SnapshotBundle:
    """Raw snapshot payloads plus provider status strings."""

    generated_at: str
    selected_providers: tuple[ProviderName, ...]
    claude_snapshot: dict[str, Any] | None
    codex_snapshot: dict[str, Any] | None
    gemini_snapshot: dict[str, Any] | None
    claude_status: str
    codex_status: str
    gemini_status: str
    kimi_snapshot: dict[str, Any] | None = None
    kimi_status: str = "Disabled by --providers"
    system_snapshot: dict[str, Any] | None = None
    system_status: str = "Disabled by --providers"


@dataclass(slots=True)
class UiState:
    """Persisted app-only UI state."""

    collapsed: dict[ProviderName, bool] = field(
        default_factory=lambda: {
            "claude": False,
            "codex": False,
            "gemini": False,
            "kimi": False,
            "system": False,
        }
    )
    window_position: tuple[int, int] = (40, 40)
    # ``None`` = follow the ``--font-size`` CLI flag / its default. A
    # concrete value here means the user zoomed via ⌘+/⌘- and we should
    # restore that size on next launch, overriding the default.
    font_size: float | None = None
    # ``None`` = follow the ``--providers`` CLI flag / default. A concrete
    # set means the user changed provider visibility from the app top bar.
    selected_providers: set[ProviderName] | None = None
