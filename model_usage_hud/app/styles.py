"""Shared styles for the minimal desktop app."""

from __future__ import annotations

from model_usage_hud.core.models import TextStyle

COLORS = {
    "bg": "#111315",
    "panel": "#171a1d",
    "border": "#2a2d30",
    "fg": "#e6edf3",
    "muted": "#8b949e",
    "red": "#f85149",
    "green": "#3fb950",
    "yellow": "#d29922",
    "cyan": "#58a6ff",
    "white": "#e6edf3",
    "orange": "#d18616",
    "brown": "#b87333",
}

STYLE_TO_COLOR: dict[TextStyle, str] = {
    "plain": COLORS["fg"],
    "bold": COLORS["fg"],
    "dim": COLORS["muted"],
    "red": COLORS["red"],
    "green": COLORS["green"],
    "yellow": COLORS["yellow"],
    "cyan": COLORS["cyan"],
    "white": COLORS["white"],
    "orange": COLORS["orange"],
    "brown": COLORS["brown"],
    "bold_red": COLORS["red"],
    "bold_green": COLORS["green"],
    "bold_white": COLORS["white"],
}


def color_for_style(style: TextStyle) -> str:
    return STYLE_TO_COLOR.get(style, COLORS["fg"])


def build_stylesheet(font_size: float) -> str:
    return f"""
QWidget {{
    background: {COLORS["bg"]};
    color: {COLORS["fg"]};
    font-family: Menlo;
    font-size: {font_size}pt;
}}

QFrame#root {{
    border: 1px solid {COLORS["border"]};
    border-radius: 10px;
}}

QFrame#card {{
    background: {COLORS["panel"]};
    border: 1px solid {COLORS["border"]};
    border-radius: 8px;
}}

QLabel#title {{
    font-weight: 600;
}}

QLabel#status {{
    color: {COLORS["muted"]};
}}

QLabel#headerMeta {{
    color: {COLORS["muted"]};
}}
"""
