"""Builders that normalize raw snapshot data into shared view models."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

from .models import (
    MetricRow,
    NoteLine,
    PaceBarCell,
    ProviderName,
    ProviderSection,
    TextStyle,
)

ORDERED_PROVIDERS: tuple[ProviderName, ...] = (
    "claude",
    "codex",
    "gemini",
    "kimi",
    "system",
)


def pace_bar_runs(
    actual_pct: float | int,
    expected_pct: float | int,
    width: int = 24,
    *,
    stale: bool = False,
) -> tuple[PaceBarCell, ...]:
    """Return a renderer-neutral description of a pace bar.

    Each element is one cell of the bar, left to right. Renderers (CLI ANSI
    text, Qt paint event, future web UI) decide how to draw ``kind`` and
    ``tone`` — this function owns the pacing logic so the CLI HUD and the
    PySide app cannot drift.

    Parameters mirror ``build_pace_bar``: ``actual_pct`` and ``expected_pct``
    are clamped to [0, 100] before conversion into integer cell counts.
    """

    actual_units = int(round((max(0.0, min(float(actual_pct), 100.0)) / 100.0) * width))
    expected_units = int(round((max(0.0, min(float(expected_pct), 100.0)) / 100.0) * width))
    marker_idx = max(0, min(width - 1, expected_units - 1 if expected_units > 0 else 0))

    cells: list[PaceBarCell] = []
    for i in range(width):
        pos = i + 1

        if i == marker_idx:
            if stale:
                marker_tone = "orange"
            elif abs(actual_units - expected_units) <= 1:
                marker_tone = "white"
            elif actual_units > expected_units:
                marker_tone = "red"
            else:
                marker_tone = "green"
            cells.append(PaceBarCell(kind="marker", tone=marker_tone))
            continue

        if stale:
            if pos <= actual_units:
                cells.append(PaceBarCell(kind="filled", tone="orange"))
            else:
                cells.append(PaceBarCell(kind="empty", tone="dim"))
            continue

        if pos <= min(actual_units, expected_units):
            cells.append(PaceBarCell(kind="filled", tone="cyan"))
        elif actual_units > expected_units and expected_units < pos <= actual_units:
            cells.append(PaceBarCell(kind="filled", tone="red"))
        elif expected_units > actual_units and actual_units < pos <= expected_units:
            cells.append(PaceBarCell(kind="filled", tone="green"))
        elif pos <= actual_units:
            cells.append(PaceBarCell(kind="filled", tone="cyan"))
        else:
            cells.append(PaceBarCell(kind="empty", tone="dim"))

    return tuple(cells)


def build_note_line(text: str, style: TextStyle = "plain") -> NoteLine:
    """Create a structured note/status line."""

    return NoteLine(text=text, style=style)


def build_window_metric_row(
    *,
    label: str,
    utilization: float | int | None,
    expected_utilization: float | int | None,
    display_mode: Literal["pace", "value_only"] = "pace",
    reset_at: str | int | None = None,
    stale: bool = False,
    prefix: str | None = None,
    prefix_style: TextStyle = "dim",
    burn_rate_per_hour: float | None = None,
    eta_hours: float | None = None,
    detail: str | None = None,
    detail_style: TextStyle = "dim",
    gauge_style: TextStyle | None = None,
) -> MetricRow:
    """Create a structured metric row for a provider section."""

    delta = None
    if utilization is not None and expected_utilization is not None:
        delta = utilization - expected_utilization

    return MetricRow(
        label=label,
        utilization=utilization,
        expected_utilization=expected_utilization,
        delta=delta,
        display_mode=display_mode,
        reset_at=reset_at,
        stale=stale,
        prefix=prefix,
        prefix_style=prefix_style,
        burn_rate_per_hour=burn_rate_per_hour,
        eta_hours=eta_hours,
        detail=detail,
        detail_style=detail_style,
        gauge_style=gauge_style,
    )


def build_provider_section(
    *,
    provider: ProviderName,
    title: str,
    status: str,
    rows: tuple[MetricRow, ...] = (),
    notes: tuple[NoteLine, ...] = (),
    stale: bool = False,
    accent: str = "",
) -> ProviderSection:
    """Create a provider section and derive its highest utilization."""

    highest_utilization = None
    for row in rows:
        if row.utilization is None:
            continue
        if highest_utilization is None or row.utilization > highest_utilization:
            highest_utilization = row.utilization

    return ProviderSection(
        provider=provider,
        title=title,
        status=status,
        highest_utilization=highest_utilization,
        rows=rows,
        notes=notes,
        stale=stale,
        accent=accent,
    )


def build_provider_view_models(
    *,
    selected_providers: set[str],
    sections_by_provider: Mapping[str, ProviderSection | None],
) -> tuple[ProviderSection, ...]:
    """Return ordered provider sections for the selected providers."""

    ordered: list[ProviderSection] = []
    for provider in ORDERED_PROVIDERS:
        if provider not in selected_providers:
            continue
        section = sections_by_provider.get(provider)
        if section is not None:
            ordered.append(section)
    return tuple(ordered)
