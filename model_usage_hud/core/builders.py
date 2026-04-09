"""Builders that normalize raw snapshot data into shared view models."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

from .models import MetricRow, NoteLine, ProviderName, ProviderSection, TextStyle

ORDERED_PROVIDERS: tuple[ProviderName, ...] = ("claude", "codex", "gemini")


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
