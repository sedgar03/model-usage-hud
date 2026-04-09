"""Stable package entrypoint for the existing CLI implementation."""

from __future__ import annotations

import usage_hud


def main() -> int:
    """Delegate to the existing CLI/Tk implementation."""
    return usage_hud.main()
